#include "MovementController.h"

#include <Windows.h>
#include <setupapi.h>
#include <hidsdi.h>

#pragma comment(lib, "Setupapi.lib")
#pragma comment(lib, "Hid.lib")

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint8_t kTitanTwoPacketMagic = 0xA0;
constexpr std::size_t kTitanTwoPacketSize = 41;
constexpr std::size_t kTitanTwoSequenceOffset = 9;
constexpr std::size_t kTitanTwoSequenceSize = 4;
struct MovementCommand {
    bool hasTarget = false;
    bool active = false;
    int deltaX = 0;
    int deltaY = 0;
    std::int32_t stickXFixed = 0;
    std::int32_t stickYFixed = 0;
    int targetX = 0;
    int targetY = 0;
    int frameWidth = 0;
    int frameHeight = 0;
    int holdMs = 0;
    std::uint32_t sequence = 0;
};

struct MovementTransportTelemetry {
    bool ready = false;
    std::uint64_t dispatchCount = 0;
    std::uint64_t skipCount = 0;
    double lastSubmitMs = 0.0;
    int userIndex = -1;
};

class IMovementOutput {
public:
    virtual ~IMovementOutput() = default;

    virtual bool Initialize(const MovementSettings& settings, std::string& error) = 0;
    virtual bool Submit(const MovementCommand& command, std::string& error) = 0;
    virtual void Release() = 0;
    virtual MovementTransportTelemetry GetTelemetry() const = 0;
};

std::filesystem::path GetExecutableDirectory() {
    std::array<wchar_t, MAX_PATH> buffer{};
    const DWORD len = ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (len == 0 || len >= buffer.size()) {
        return {};
    }

    return std::filesystem::path(buffer.data()).parent_path();
}

std::filesystem::path ResolveMovementPath(const std::string& path) {
    const std::filesystem::path rawPath(path);
    if (rawPath.empty()) {
        return {};
    }
    if (rawPath.is_absolute()) {
        return rawPath;
    }
    return GetExecutableDirectory() / rawPath;
}

template <std::size_t N>
void WriteInt32BigEndian(std::array<std::uint8_t, N>& payload, std::size_t offset, std::int32_t value) {
    const std::uint32_t uvalue = static_cast<std::uint32_t>(value);
    payload[offset + 0] = static_cast<std::uint8_t>((uvalue >> 24) & 0xFFU);
    payload[offset + 1] = static_cast<std::uint8_t>((uvalue >> 16) & 0xFFU);
    payload[offset + 2] = static_cast<std::uint8_t>((uvalue >> 8) & 0xFFU);
    payload[offset + 3] = static_cast<std::uint8_t>(uvalue & 0xFFU);
}

std::int32_t EncodeTitanTwoStickPercent(double percent) {
    constexpr double kFixedScale = 65536.0;
    const double clamped = std::clamp(percent, -100.0, 100.0);
    return static_cast<std::int32_t>(std::llround(clamped * kFixedScale));
}

double ApplySignedCurve(double value, double exponent) {
    const double clamped = std::clamp(value, -1.0, 1.0);
    const double magnitude = std::pow(std::abs(clamped), exponent);
    return std::copysign(magnitude, clamped);
}

double ApplyMinimumSignedMagnitude(double value, double minimumMagnitude) {
    if (value == 0.0) {
        return 0.0;
    }

    const double magnitude = std::abs(value);
    return std::copysign((std::max)(magnitude, minimumMagnitude), value);
}

std::string Utf8Path(const std::filesystem::path& path) {
    return path.u8string();
}

int ComputeMovementKeepaliveMs(const MovementCommand& command) {
    if (!command.active || command.holdMs <= 0) {
        return 0;
    }

    const int halfHoldMs = (std::max)(1, command.holdMs / 2);
    return (std::min)(8, halfHoldMs);
}

std::string FormatWin32Error(DWORD errorCode) {
    LPSTR messageBuffer = nullptr;
    const DWORD chars = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        errorCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&messageBuffer),
        0,
        nullptr);

    std::string message;
    if (chars != 0 && messageBuffer != nullptr) {
        message.assign(messageBuffer, chars);
        while (!message.empty() && (message.back() == '\r' || message.back() == '\n' || message.back() == ' ')) {
            message.pop_back();
        }
    }
    if (messageBuffer != nullptr) {
        ::LocalFree(messageBuffer);
    }

    if (message.empty()) {
        std::ostringstream oss;
        oss << "Win32=" << errorCode;
        return oss.str();
    }

    std::ostringstream oss;
    oss << message << " (Win32=" << errorCode << ")";
    return oss.str();
}

double ClampDtSeconds(double dtSeconds) {
    if (!std::isfinite(dtSeconds) || dtSeconds <= 0.0) {
        return 1.0 / 60.0;
    }
    return std::clamp(dtSeconds, 1.0 / 240.0, 0.12);
}

class OneEuroFilter {
public:
    void Reset() {
        m_hasValue = false;
        m_value = 0.0;
        m_derivative = 0.0;
        m_lastRaw = 0.0;
    }

    double Filter(double value, double dtSeconds, double minCutoff, double beta, double derivativeCutoff) {
        dtSeconds = ClampDtSeconds(dtSeconds);
        if (!m_hasValue) {
            m_hasValue = true;
            m_value = value;
            m_derivative = 0.0;
            m_lastRaw = value;
            return value;
        }

        const double rawDerivative = (value - m_lastRaw) / dtSeconds;
        m_lastRaw = value;
        m_derivative = LowPass(rawDerivative, m_derivative, SmoothingFactor(derivativeCutoff, dtSeconds));

        const double cutoff = (std::max)(0.001, minCutoff + beta * std::abs(m_derivative));
        m_value = LowPass(value, m_value, SmoothingFactor(cutoff, dtSeconds));
        return m_value;
    }

private:
    static double SmoothingFactor(double cutoff, double dtSeconds) {
        constexpr double kPi = 3.14159265358979323846;
        const double safeCutoff = (std::max)(0.001, cutoff);
        const double tau = 1.0 / (2.0 * kPi * safeCutoff);
        return 1.0 / (1.0 + tau / dtSeconds);
    }

    static double LowPass(double value, double previous, double alpha) {
        return alpha * value + (1.0 - alpha) * previous;
    }

    bool m_hasValue = false;
    double m_value = 0.0;
    double m_derivative = 0.0;
    double m_lastRaw = 0.0;
};

struct AimMeasurement {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double prob = 0.0;
    int label = -1;
};

struct AimTrack {
    int id = -1;
    int label = -1;
    int age = 0;
    int hits = 0;
    int lost = 0;
    bool matched = false;
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double prob = 0.0;
    double velocityX = 0.0;
    double velocityY = 0.0;
    double lastMeasurementX = 0.0;
    double lastMeasurementY = 0.0;
    bool hasMeasurement = false;
};

struct SelectedAimTarget {
    bool found = false;
    int trackId = -1;
    int label = -1;
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double prob = 0.0;
    double score = 0.0;
    double velocityX = 0.0;
    double velocityY = 0.0;
};

struct AimAssociationCandidate {
    std::size_t trackIndex = 0;
    std::size_t measurementIndex = 0;
    double cost = 0.0;
};

struct MovementAimState {
    std::vector<AimTrack> tracks;
    std::vector<AimMeasurement> measurementsScratch;
    std::vector<AimAssociationCandidate> candidatesScratch;
    std::vector<std::uint8_t> trackUsedScratch;
    std::vector<std::uint8_t> measurementUsedScratch;
    int nextTrackId = 1;
    int lockedTrackId = -1;
    int filteredTrackId = -1;
    bool hasFrameTime = false;
    std::chrono::steady_clock::time_point lastFrameTime{};
    OneEuroFilter targetFilterX;
    OneEuroFilter targetFilterY;
    OneEuroFilter leadFilterX;
    OneEuroFilter leadFilterY;
    double lastFilteredTargetX = 0.0;
    double lastFilteredTargetY = 0.0;
    bool hasLastError = false;
    double lastErrorX = 0.0;
    double lastErrorY = 0.0;
    double lastStickXPercent = 0.0;
    double lastStickYPercent = 0.0;

    void ResetRuntime() {
        tracks.clear();
        measurementsScratch.clear();
        candidatesScratch.clear();
        trackUsedScratch.clear();
        measurementUsedScratch.clear();
        lockedTrackId = -1;
        filteredTrackId = -1;
        hasFrameTime = false;
        targetFilterX.Reset();
        targetFilterY.Reset();
        leadFilterX.Reset();
        leadFilterY.Reset();
        lastFilteredTargetX = 0.0;
        lastFilteredTargetY = 0.0;
        hasLastError = false;
        lastErrorX = 0.0;
        lastErrorY = 0.0;
        lastStickXPercent = 0.0;
        lastStickYPercent = 0.0;
    }
};

double AdvanceAimFrameTime(MovementAimState& aimState) {
    const auto now = std::chrono::steady_clock::now();
    double dtSeconds = 1.0 / 60.0;
    if (aimState.hasFrameTime) {
        dtSeconds = std::chrono::duration<double>(now - aimState.lastFrameTime).count();
    }
    aimState.lastFrameTime = now;
    aimState.hasFrameTime = true;
    return ClampDtSeconds(dtSeconds);
}

AimMeasurement MakeAimMeasurement(const DetectionObject& detection, double aimYFactor) {
    AimMeasurement measurement;
    measurement.width = (std::max)(0.0, static_cast<double>(detection.bbox.width));
    measurement.height = (std::max)(0.0, static_cast<double>(detection.bbox.height));
    measurement.x = static_cast<double>(detection.bbox.x) + measurement.width * 0.5;
    measurement.y = static_cast<double>(detection.bbox.y) + measurement.height * aimYFactor;
    measurement.prob = static_cast<double>(detection.prob);
    measurement.label = detection.label;
    return measurement;
}

double AssociationCost(const AimTrack& track, const AimMeasurement& measurement) {
    const double avgWidth = (std::max)(1.0, (track.width + measurement.width) * 0.5);
    const double avgHeight = (std::max)(1.0, (track.height + measurement.height) * 0.5);
    const double dx = (track.x - measurement.x) / (avgWidth * 1.5);
    const double dy = (track.y - measurement.y) / (avgHeight * 1.5);
    const double distanceCost = std::hypot(dx, dy);
    const double areaA = (std::max)(1.0, track.width * track.height);
    const double areaB = (std::max)(1.0, measurement.width * measurement.height);
    const double shapeCost = std::abs(areaA - areaB) / (std::max)(areaA, areaB);
    const double labelCost =
        (track.label >= 0 && measurement.label >= 0 && track.label != measurement.label) ? 2.0 : 0.0;
    return distanceCost + shapeCost * 0.15 + labelCost;
}

void UpdateAimTracks(
    MovementAimState& aimState,
    const MovementSettings& settings,
    const std::vector<DetectionObject>& detections,
    double aimYFactor,
    double dtSeconds) {
    dtSeconds = ClampDtSeconds(dtSeconds);
    for (auto& track : aimState.tracks) {
        track.x += track.velocityX * dtSeconds;
        track.y += track.velocityY * dtSeconds;
        track.matched = false;
        ++track.age;
    }

    std::vector<AimMeasurement>& measurements = aimState.measurementsScratch;
    measurements.clear();
    measurements.reserve(detections.size());
    for (const auto& detection : detections) {
        AimMeasurement measurement = MakeAimMeasurement(detection, aimYFactor);
        if (measurement.width > 0.0 && measurement.height > 0.0) {
            measurements.push_back(measurement);
        }
    }

    std::vector<AimAssociationCandidate>& candidates = aimState.candidatesScratch;
    candidates.clear();
    candidates.reserve(aimState.tracks.size() * measurements.size());
    for (std::size_t i = 0; i < aimState.tracks.size(); ++i) {
        for (std::size_t j = 0; j < measurements.size(); ++j) {
            candidates.push_back({ i, j, AssociationCost(aimState.tracks[i], measurements[j]) });
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const AimAssociationCandidate& lhs, const AimAssociationCandidate& rhs) {
        return lhs.cost < rhs.cost;
    });

    std::vector<std::uint8_t>& trackUsed = aimState.trackUsedScratch;
    std::vector<std::uint8_t>& measurementUsed = aimState.measurementUsedScratch;
    trackUsed.assign(aimState.tracks.size(), 0);
    measurementUsed.assign(measurements.size(), 0);
    const double maxCost = (std::max)(0.05, settings.aimTrackMatchMaxCost);
    for (const auto& candidate : candidates) {
        if (candidate.cost > maxCost ||
            trackUsed[candidate.trackIndex] ||
            measurementUsed[candidate.measurementIndex]) {
            continue;
        }

        AimTrack& track = aimState.tracks[candidate.trackIndex];
        const AimMeasurement& measurement = measurements[candidate.measurementIndex];
        double measuredVelocityX = 0.0;
        double measuredVelocityY = 0.0;
        if (track.hasMeasurement) {
            measuredVelocityX = (measurement.x - track.lastMeasurementX) / dtSeconds;
            measuredVelocityY = (measurement.y - track.lastMeasurementY) / dtSeconds;
        }
        track.velocityX = track.hasMeasurement
            ? track.velocityX * 0.75 + measuredVelocityX * 0.25
            : 0.0;
        track.velocityY = track.hasMeasurement
            ? track.velocityY * 0.75 + measuredVelocityY * 0.25
            : 0.0;
        track.x = measurement.x;
        track.y = measurement.y;
        track.width = measurement.width;
        track.height = measurement.height;
        track.prob = measurement.prob;
        track.label = measurement.label;
        track.lastMeasurementX = measurement.x;
        track.lastMeasurementY = measurement.y;
        track.hasMeasurement = true;
        track.matched = true;
        track.lost = 0;
        ++track.hits;
        trackUsed[candidate.trackIndex] = 1;
        measurementUsed[candidate.measurementIndex] = 1;
    }

    for (auto& track : aimState.tracks) {
        if (!track.matched) {
            ++track.lost;
        }
    }

    for (auto& track : aimState.tracks) {
        if (!track.matched && track.lost > 0) {
            const double decayFactor = 1.0 / (1.0 + track.lost * 0.5);
            track.velocityX *= decayFactor;
            track.velocityY *= decayFactor;
        }
    }

    const int maxLost = (std::max)(0, settings.aimTrackLostFrames);
    aimState.tracks.erase(
        std::remove_if(
            aimState.tracks.begin(),
            aimState.tracks.end(),
            [maxLost](const AimTrack& track) {
                return track.lost > maxLost;
            }),
        aimState.tracks.end());

    for (std::size_t i = 0; i < measurements.size(); ++i) {
        if (measurementUsed[i] != 0) {
            continue;
        }
        const AimMeasurement& measurement = measurements[i];
        AimTrack track;
        track.id = aimState.nextTrackId++;
        track.label = measurement.label;
        track.age = 1;
        track.hits = 1;
        track.lost = 0;
        track.matched = true;
        track.x = measurement.x;
        track.y = measurement.y;
        track.width = measurement.width;
        track.height = measurement.height;
        track.prob = measurement.prob;
        track.lastMeasurementX = measurement.x;
        track.lastMeasurementY = measurement.y;
        track.hasMeasurement = true;
        aimState.tracks.push_back(track);
    }
}

double AimTrackScore(
    const AimTrack& track,
    int frameWidth,
    int frameHeight,
    double centerX,
    double centerY,
    const MovementSettings& settings) {
    const double maxDistance = (std::max)(1.0, std::hypot(centerX, centerY));
    const double distance = std::hypot(track.x - centerX, track.y - centerY);
    const double normalizedDistance = distance / maxDistance;
    const double distanceWeight = 1.0 - normalizedDistance * normalizedDistance;
    const double sizeScore = std::clamp(track.height / (std::max)(1.0, static_cast<double>(frameHeight)), 0.0, 0.35);
    const double ageBonus = std::clamp(static_cast<double>(track.hits) * 0.04, 0.0, 0.3);
    double score = track.prob * 0.8
        + distanceWeight * 3.5
        + sizeScore * 0.5
        + ageBonus * 1.5
        - static_cast<double>(track.lost) * 0.35;
    if (settings.fovRadius > 0.0 && distance > settings.fovRadius) {
        score -= 10.0;
    }
    (void)frameWidth;
    return score;
}

SelectedAimTarget SelectTrackedAimTarget(
    MovementAimState& aimState,
    const MovementSettings& settings,
    int frameWidth,
    int frameHeight,
    double centerX,
    double centerY) {
    const int confirmFrames = (std::max)(1, settings.aimTrackConfirmFrames);
    const AimTrack* locked = nullptr;
    const AimTrack* best = nullptr;
    double bestScore = -std::numeric_limits<double>::infinity();
    double lockedScore = -std::numeric_limits<double>::infinity();
    double selectedScore = -std::numeric_limits<double>::infinity();

    for (const auto& track : aimState.tracks) {
        const bool confirmed = track.hits >= confirmFrames;
        const bool isLocked = track.id == aimState.lockedTrackId;
        if (!confirmed && !isLocked) {
            continue;
        }

        double score = AimTrackScore(track, frameWidth, frameHeight, centerX, centerY, settings);
        if (isLocked) {
            locked = &track;
            score += settings.aimTargetLockBonus;
            lockedScore = score;
        }
        if (score > bestScore) {
            bestScore = score;
            best = &track;
        }
    }

    const AimTrack* selected = nullptr;
    if (best == nullptr) {
        aimState.lockedTrackId = -1;
        return {};
    }
    if (locked != nullptr && best->id != locked->id) {
        const double switchThreshold = settings.aimTargetSwitchMargin + settings.aimTargetLockBonus * 0.5;
        if (bestScore <= lockedScore + switchThreshold) {
            selected = locked;
            selectedScore = lockedScore;
        } else {
            selected = best;
            selectedScore = bestScore;
        }
    } else {
        selected = best;
        selectedScore = bestScore;
    }

    aimState.lockedTrackId = selected->id;
    SelectedAimTarget target;
    target.found = true;
    target.trackId = selected->id;
    target.label = selected->label;
    target.x = selected->x;
    target.y = selected->y;
    target.width = selected->width;
    target.height = selected->height;
    target.prob = selected->prob;
    target.score = selectedScore;
    target.velocityX = selected->velocityX;
    target.velocityY = selected->velocityY;
    return target;
}

SelectedAimTarget SelectDirectAimTarget(
    const MovementSettings& settings,
    int frameWidth,
    int frameHeight,
    const std::vector<DetectionObject>& detections,
    double centerX,
    double centerY,
    double aimYFactor) {
    SelectedAimTarget target;
    const double maxDistance = (std::max)(1.0, std::hypot(centerX, centerY));
    double bestScore = -std::numeric_limits<double>::infinity();
    for (const auto& detection : detections) {
        AimMeasurement measurement = MakeAimMeasurement(detection, aimYFactor);
        const double distance = std::hypot(measurement.x - centerX, measurement.y - centerY);
        const double normalizedDistance = distance / maxDistance;
        const double score = measurement.prob * 2.0 - normalizedDistance;
        if (!target.found || score > bestScore) {
            target.found = true;
            target.trackId = -1;
            target.label = measurement.label;
            target.x = measurement.x;
            target.y = measurement.y;
            target.width = measurement.width;
            target.height = measurement.height;
            target.prob = measurement.prob;
            target.score = score;
            bestScore = score;
        }
    }
    (void)settings;
    (void)frameWidth;
    (void)frameHeight;
    return target;
}

double ApplySlewLimit(double previous, double target, double maxChange) {
    if (maxChange <= 0.0 || !std::isfinite(maxChange)) {
        return target;
    }
    return previous + std::clamp(target - previous, -maxChange, maxChange);
}

class NullMovementOutput final : public IMovementOutput {
public:
    bool Initialize(const MovementSettings&, std::string& error) override {
        error.clear();
        return true;
    }

    bool Submit(const MovementCommand&, std::string& error) override {
        error.clear();
        return true;
    }

    void Release() override {
    }

    MovementTransportTelemetry GetTelemetry() const override {
        return {};
    }
};

class TitanTwoGcvMovementOutput final : public IMovementOutput {
public:
    bool Initialize(const MovementSettings& settings, std::string& error) override {
        error.clear();
        Release();

        m_path = ResolveMovementPath(settings.titanTwoGcvPath);
        if (m_path.empty()) {
            error = "Titan Two GCV movement path is empty.";
            return false;
        }

        std::error_code ec;
        const auto parent = m_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
        }

        m_handle = ::CreateFileW(
            m_path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (m_handle == INVALID_HANDLE_VALUE) {
            std::ostringstream oss;
            oss << "CreateFileW failed for Titan Two GCV path '" << Utf8Path(m_path)
                << "' (" << FormatWin32Error(::GetLastError()) << ")";
            error = oss.str();
            return false;
        }

        m_hasLastPayload = false;
        m_lastWriteTime = std::chrono::steady_clock::time_point{};
        m_dispatchCount = 0;
        m_skipCount = 0;
        m_lastSubmitMs = 0.0;
        return true;
    }

    bool Submit(const MovementCommand& command, std::string& error) override {
        error.clear();
        if (m_handle == INVALID_HANDLE_VALUE) {
            error = "Titan Two GCV movement output is not initialized.";
            return false;
        }

        const int keepaliveMs = ComputeMovementKeepaliveMs(command);
        return WritePayload(BuildPayload(command), keepaliveMs, false, error);
    }

    void Release() override {
        if (m_handle != INVALID_HANDLE_VALUE) {
            std::string ignoredError;
            WritePayload(BuildPayload(MovementCommand{}), 0, true, ignoredError);
            ::CloseHandle(m_handle);
            m_handle = INVALID_HANDLE_VALUE;
        }
        m_path.clear();
        m_hasLastPayload = false;
        m_lastWriteTime = std::chrono::steady_clock::time_point{};
        m_dispatchCount = 0;
        m_skipCount = 0;
        m_lastSubmitMs = 0.0;
    }

    ~TitanTwoGcvMovementOutput() override {
        Release();
    }

    MovementTransportTelemetry GetTelemetry() const override {
        MovementTransportTelemetry telemetry;
        telemetry.ready = (m_handle != INVALID_HANDLE_VALUE);
        telemetry.dispatchCount = m_dispatchCount;
        telemetry.skipCount = m_skipCount;
        telemetry.lastSubmitMs = m_lastSubmitMs;
        return telemetry;
    }

private:
    static std::array<std::uint8_t, kTitanTwoPacketSize> BuildPayload(const MovementCommand& command) {
        std::array<std::uint8_t, kTitanTwoPacketSize> payload{};
        payload[0] = static_cast<std::uint8_t>(
            kTitanTwoPacketMagic |
            (command.hasTarget ? 0x01 : 0x00) |
            (command.active ? 0x02 : 0x00));
        WriteInt32BigEndian(payload, 1, command.deltaX);
        WriteInt32BigEndian(payload, 5, command.deltaY);
        WriteInt32BigEndian(payload, 9, static_cast<std::int32_t>(command.sequence));
        WriteInt32BigEndian(payload, 13, command.targetX);
        WriteInt32BigEndian(payload, 17, command.targetY);
        WriteInt32BigEndian(payload, 21, command.frameWidth);
        WriteInt32BigEndian(payload, 25, command.frameHeight);
        WriteInt32BigEndian(payload, 29, command.holdMs);
        WriteInt32BigEndian(payload, 33, command.stickXFixed);
        WriteInt32BigEndian(payload, 37, command.stickYFixed);
        return payload;
    }

    bool PayloadEquivalentIgnoringSequence(
        const std::array<std::uint8_t, kTitanTwoPacketSize>& lhs,
        const std::array<std::uint8_t, kTitanTwoPacketSize>& rhs) const {
        return std::equal(
                   lhs.begin(),
                   lhs.begin() + static_cast<std::ptrdiff_t>(kTitanTwoSequenceOffset),
                   rhs.begin()) &&
            std::equal(
                lhs.begin() + static_cast<std::ptrdiff_t>(kTitanTwoSequenceOffset + kTitanTwoSequenceSize),
                lhs.end(),
                rhs.begin() + static_cast<std::ptrdiff_t>(kTitanTwoSequenceOffset + kTitanTwoSequenceSize));
    }

    bool ShouldSkipWrite(
        const std::array<std::uint8_t, kTitanTwoPacketSize>& payload,
        int keepaliveMs) const {
        if (!m_hasLastPayload) {
            return false;
        }

        if (!PayloadEquivalentIgnoringSequence(payload, m_lastPayload)) {
            return false;
        }

        if (keepaliveMs <= 0) {
            return true;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastWriteTime).count();
        return elapsedMs < keepaliveMs;
    }

    bool WritePayload(
        const std::array<std::uint8_t, kTitanTwoPacketSize>& payload,
        int keepaliveMs,
        bool force,
        std::string& error) {
        error.clear();
        if (!force && ShouldSkipWrite(payload, keepaliveMs)) {
            ++m_skipCount;
            m_lastSubmitMs = 0.0;
            return true;
        }

        const auto submitStart = std::chrono::steady_clock::now();

        LARGE_INTEGER pos{};
        if (!::SetFilePointerEx(m_handle, pos, nullptr, FILE_BEGIN)) {
            error = "SetFilePointerEx failed (" + FormatWin32Error(::GetLastError()) + ")";
            return false;
        }
        DWORD written = 0;
        if (!::WriteFile(m_handle, payload.data(),
                         static_cast<DWORD>(kTitanTwoPacketSize), &written, nullptr) ||
            written != kTitanTwoPacketSize) {
            error = "WriteFile failed (" + FormatWin32Error(::GetLastError()) + ")";
            return false;
        }

        m_lastPayload = payload;
        m_hasLastPayload = true;
        m_lastWriteTime = std::chrono::steady_clock::now();
        ++m_dispatchCount;
        m_lastSubmitMs =
            std::chrono::duration<double, std::milli>(m_lastWriteTime - submitStart).count();
        return true;
    }

    HANDLE m_handle = INVALID_HANDLE_VALUE;
    std::filesystem::path m_path;
    std::array<std::uint8_t, kTitanTwoPacketSize> m_lastPayload{};
    bool m_hasLastPayload = false;
    std::chrono::steady_clock::time_point m_lastWriteTime{};
    std::uint64_t m_dispatchCount = 0;
    std::uint64_t m_skipCount = 0;
    double m_lastSubmitMs = 0.0;
};

class TitanTwoHidDirectMovementOutput final : public IMovementOutput {
public:
    bool Initialize(const MovementSettings& settings, std::string& error) override {
        error.clear();
        Release();

        std::cout << "[WARNING] TitanTwoHidDirect is EXPERIMENTAL. "
            << "TT2 GPC scripts using gcv_ready()/gcv_read() cannot consume raw HID writes. "
            << "This backend requires a compatible TT2 firmware/script that reads PROG port HID data."
            << std::endl;

        m_vendorId = static_cast<USHORT>(settings.titanTwoHidVendorId);
        m_productId = static_cast<USHORT>(settings.titanTwoHidProductId);
        m_reportId = settings.titanTwoHidReportId;
        m_payloadOffset = settings.titanTwoHidPayloadOffset;

        GUID hidGuid{};
        ::HidD_GetHidGuid(&hidGuid);
        HDEVINFO deviceInfo = ::SetupDiGetClassDevsW(
            &hidGuid, nullptr, nullptr, DIGCF_DEVICEINTERFACE | DIGCF_PRESENT);
        if (deviceInfo == INVALID_HANDLE_VALUE) {
            error = "SetupDiGetClassDevsW failed (" + FormatWin32Error(::GetLastError()) + ")";
            return false;
        }

        bool foundVidPid = false;
        DWORD interfaceIndex = 0;
        for (;;) {
            SP_DEVICE_INTERFACE_DATA interfaceData{};
            interfaceData.cbSize = sizeof(interfaceData);
            if (!::SetupDiEnumDeviceInterfaces(deviceInfo, nullptr, &hidGuid, interfaceIndex, &interfaceData)) {
                break;
            }
            ++interfaceIndex;

            DWORD requiredSize = 0;
            ::SetupDiGetDeviceInterfaceDetailW(deviceInfo, &interfaceData, nullptr, 0, &requiredSize, nullptr);
            if (requiredSize == 0) continue;

            auto detailBuffer = std::make_unique<std::uint8_t[]>(requiredSize);
            auto* detailData = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(detailBuffer.get());
            detailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
            if (!::SetupDiGetDeviceInterfaceDetailW(deviceInfo, &interfaceData, detailData, requiredSize, nullptr, nullptr)) {
                continue;
            }

            HANDLE candidate = ::CreateFileW(
                detailData->DevicePath,
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (candidate == INVALID_HANDLE_VALUE) continue;

            HIDD_ATTRIBUTES attributes{};
            attributes.Size = sizeof(attributes);
            if (!::HidD_GetAttributes(candidate, &attributes)) {
                ::CloseHandle(candidate);
                continue;
            }
            if (attributes.VendorID != m_vendorId || attributes.ProductID != m_productId) {
                ::CloseHandle(candidate);
                continue;
            }
            foundVidPid = true;

            PHIDP_PREPARSED_DATA preparsedData = nullptr;
            if (!::HidD_GetPreparsedData(candidate, &preparsedData)) {
                ::CloseHandle(candidate);
                continue;
            }
            HIDP_CAPS caps{};
            const NTSTATUS capsStatus = ::HidP_GetCaps(preparsedData, &caps);
            ::HidD_FreePreparsedData(preparsedData);
            if (capsStatus != HIDP_STATUS_SUCCESS) {
                ::CloseHandle(candidate);
                continue;
            }

            m_reportSize = settings.titanTwoHidReportSize > 0
                ? settings.titanTwoHidReportSize
                : static_cast<int>(caps.OutputReportByteLength);
            if (m_reportSize <= 0 ||
                m_payloadOffset + static_cast<int>(kTitanTwoPacketSize) > m_reportSize) {
                ::CloseHandle(candidate);
                continue;
            }

            m_handle = candidate;
            ::SetupDiDestroyDeviceInfoList(deviceInfo);
            m_dispatchCount = 0;
            m_skipCount = 0;
            m_lastSubmitMs = 0.0;
            m_hasLastPayload = false;
            return true;
        }

        ::SetupDiDestroyDeviceInfoList(deviceInfo);
        if (!foundVidPid) {
            std::ostringstream oss;
            oss << "No TT2 HID device found (VID=0x"
                << std::hex << std::uppercase << m_vendorId
                << " PID=0x" << m_productId << ")";
            error = oss.str();
        } else {
            error = "Found TT2 VID/PID but no writable HID interface.";
        }
        return false;
    }

    bool Submit(const MovementCommand& command, std::string& error) override {
        error.clear();
        if (m_handle == INVALID_HANDLE_VALUE) {
            error = "TT2 HID not initialized.";
            return false;
        }

        const auto payload = BuildPayload(command);
        const int keepaliveMs = ComputeMovementKeepaliveMs(command);
        if (m_hasLastPayload && PayloadEquivalent(payload, m_lastPayload) && keepaliveMs > 0) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastWriteTime).count();
            if (elapsedMs < keepaliveMs) {
                ++m_skipCount;
                m_lastSubmitMs = 0.0;
                return true;
            }
        }

        std::vector<std::uint8_t> report(static_cast<std::size_t>(m_reportSize), 0);
        report[0] = static_cast<std::uint8_t>(m_reportId);
        std::copy(payload.begin(), payload.end(),
                  report.begin() + static_cast<std::ptrdiff_t>(m_payloadOffset));

        const auto submitStart = std::chrono::steady_clock::now();
        DWORD bytesWritten = 0;
        if (!::WriteFile(m_handle, report.data(), static_cast<DWORD>(report.size()), &bytesWritten, nullptr) ||
            bytesWritten != report.size()) {
            error = "TT2 HID WriteFile failed (" + FormatWin32Error(::GetLastError()) + ")";
            return false;
        }
        m_lastWriteTime = std::chrono::steady_clock::now();
        m_lastSubmitMs = std::chrono::duration<double, std::milli>(m_lastWriteTime - submitStart).count();
        m_lastPayload = payload;
        m_hasLastPayload = true;
        ++m_dispatchCount;
        return true;
    }

    void Release() override {
        if (m_handle != INVALID_HANDLE_VALUE) {
            std::string ignoredError;
            MovementCommand neutral{};
            Submit(neutral, ignoredError);
            ::CloseHandle(m_handle);
            m_handle = INVALID_HANDLE_VALUE;
        }
    }

    ~TitanTwoHidDirectMovementOutput() override { Release(); }

    MovementTransportTelemetry GetTelemetry() const override {
        MovementTransportTelemetry t;
        t.ready = (m_handle != INVALID_HANDLE_VALUE);
        t.dispatchCount = m_dispatchCount;
        t.skipCount = m_skipCount;
        t.lastSubmitMs = m_lastSubmitMs;
        return t;
    }

private:
    static std::array<std::uint8_t, kTitanTwoPacketSize> BuildPayload(const MovementCommand& command) {
        std::array<std::uint8_t, kTitanTwoPacketSize> payload{};
        payload[0] = static_cast<std::uint8_t>(
            kTitanTwoPacketMagic |
            (command.hasTarget ? 0x01 : 0x00) |
            (command.active ? 0x02 : 0x00));
        WriteInt32BigEndian(payload, 1, command.deltaX);
        WriteInt32BigEndian(payload, 5, command.deltaY);
        WriteInt32BigEndian(payload, 9, static_cast<std::int32_t>(command.sequence));
        WriteInt32BigEndian(payload, 13, command.targetX);
        WriteInt32BigEndian(payload, 17, command.targetY);
        WriteInt32BigEndian(payload, 21, command.frameWidth);
        WriteInt32BigEndian(payload, 25, command.frameHeight);
        WriteInt32BigEndian(payload, 29, command.holdMs);
        WriteInt32BigEndian(payload, 33, command.stickXFixed);
        WriteInt32BigEndian(payload, 37, command.stickYFixed);
        return payload;
    }

    bool PayloadEquivalent(
        const std::array<std::uint8_t, kTitanTwoPacketSize>& lhs,
        const std::array<std::uint8_t, kTitanTwoPacketSize>& rhs) const {
        return std::equal(
                   lhs.begin(),
                   lhs.begin() + static_cast<std::ptrdiff_t>(kTitanTwoSequenceOffset),
                   rhs.begin()) &&
            std::equal(
                lhs.begin() + static_cast<std::ptrdiff_t>(kTitanTwoSequenceOffset + kTitanTwoSequenceSize),
                lhs.end(),
                rhs.begin() + static_cast<std::ptrdiff_t>(kTitanTwoSequenceOffset + kTitanTwoSequenceSize));
    }

    HANDLE m_handle = INVALID_HANDLE_VALUE;
    USHORT m_vendorId = 0x2508;
    USHORT m_productId = 0x0032;
    int m_reportSize = 65;
    int m_payloadOffset = 1;
    int m_reportId = 0;
    std::array<std::uint8_t, kTitanTwoPacketSize> m_lastPayload{};
    bool m_hasLastPayload = false;
    std::chrono::steady_clock::time_point m_lastWriteTime{};
    std::uint64_t m_dispatchCount = 0;
    std::uint64_t m_skipCount = 0;
    double m_lastSubmitMs = 0.0;
};

MovementCommand BuildMovementCommand(
    const MovementSettings& settings,
    MovementAimState& aimState,
    int frameWidth,
    int frameHeight,
    const std::vector<DetectionObject>& detections,
    MovementDebugState& debugState) {
    debugState = MovementDebugState{};

    MovementCommand command;
    command.frameWidth = frameWidth;
    command.frameHeight = frameHeight;
    command.holdMs = settings.titanTwoHoldMs;

    if (settings.movementTestStickEnabled) {
        aimState.ResetRuntime();
        const double stickXPercent = std::clamp(settings.movementTestStickXPercent, -100.0, 100.0);
        const double stickYPercent = std::clamp(settings.movementTestStickYPercent, -100.0, 100.0);
        command.hasTarget = true;
        command.active = std::abs(stickXPercent) > 0.0 || std::abs(stickYPercent) > 0.0;
        command.targetX = frameWidth > 0 ? frameWidth / 2 : 0;
        command.targetY = frameHeight > 0 ? frameHeight / 2 : 0;
        command.stickXFixed = EncodeTitanTwoStickPercent(stickXPercent);
        command.stickYFixed = EncodeTitanTwoStickPercent(stickYPercent);

        debugState.hasTarget = command.hasTarget;
        debugState.active = command.active;
        debugState.targetX = command.targetX;
        debugState.targetY = command.targetY;
        debugState.targetLabel = -1;
        debugState.deltaX = 0;
        debugState.deltaY = 0;
        debugState.stickXPercent = stickXPercent;
        debugState.stickYPercent = stickYPercent;
        debugState.score = 1.0f;
        return command;
    }

    if (frameWidth <= 0 || frameHeight <= 0) {
        aimState.lockedTrackId = -1;
        aimState.filteredTrackId = -1;
        aimState.targetFilterX.Reset();
        aimState.targetFilterY.Reset();
        aimState.leadFilterX.Reset();
        aimState.leadFilterY.Reset();
        aimState.hasLastError = false;
        aimState.lastStickXPercent = 0.0;
        aimState.lastStickYPercent = 0.0;
        return command;
    }

    const double dtSeconds = AdvanceAimFrameTime(aimState);
    const double centerX = static_cast<double>(frameWidth) * 0.5 + settings.centerOffsetX;
    const double centerY = static_cast<double>(frameHeight) * 0.5 + settings.centerOffsetY;
    const double aimYFactor = std::clamp(0.5 + settings.verticalBias, 0.0, 1.0);
    const bool isTitanTwoGamepadBackend =
        settings.backend == MovementBackendKind::TitanTwoGcvGamepad;

    const std::vector<DetectionObject>* effectiveDetections = &detections;
    std::vector<DetectionObject> fovFiltered;
    if (settings.fovRadius > 0.0 && !detections.empty()) {
        const double fovRadiusSq = settings.fovRadius * settings.fovRadius;
        fovFiltered.reserve(detections.size());
        for (const auto& det : detections) {
            const double detCenterX = static_cast<double>(det.bbox.x) + static_cast<double>(det.bbox.width) * 0.5;
            const double detCenterY = static_cast<double>(det.bbox.y) + static_cast<double>(det.bbox.height) * aimYFactor;
            const double dx = detCenterX - centerX;
            const double dy = detCenterY - centerY;
            if (dx * dx + dy * dy <= fovRadiusSq) {
                fovFiltered.push_back(det);
            }
        }
        effectiveDetections = &fovFiltered;
    }

    SelectedAimTarget selectedTarget;
    if (settings.aimTrackingEnabled) {
        UpdateAimTracks(aimState, settings, *effectiveDetections, aimYFactor, dtSeconds);
        selectedTarget = SelectTrackedAimTarget(
            aimState,
            settings,
            frameWidth,
            frameHeight,
            centerX,
            centerY);
    } else {
        if (effectiveDetections->empty()) {
            aimState.hasLastError = false;
            aimState.lastStickXPercent = 0.0;
            aimState.lastStickYPercent = 0.0;
            return command;
        }
        selectedTarget = SelectDirectAimTarget(
            settings,
            frameWidth,
            frameHeight,
            *effectiveDetections,
            centerX,
            centerY,
            aimYFactor);
    }

    if (!selectedTarget.found) {
        aimState.lockedTrackId = -1;
        aimState.filteredTrackId = -1;
        aimState.targetFilterX.Reset();
        aimState.targetFilterY.Reset();
        aimState.leadFilterX.Reset();
        aimState.leadFilterY.Reset();
        aimState.hasLastError = false;
        aimState.lastStickXPercent = 0.0;
        aimState.lastStickYPercent = 0.0;
        return command;
    }

    if (aimState.filteredTrackId != selectedTarget.trackId) {
        aimState.filteredTrackId = selectedTarget.trackId;
        aimState.targetFilterX.Reset();
        aimState.targetFilterY.Reset();
        aimState.leadFilterX.Reset();
        aimState.leadFilterY.Reset();
        aimState.hasLastError = false;
    }

    const double filteredTargetX = settings.aimOneEuroEnabled
        ? aimState.targetFilterX.Filter(
              selectedTarget.x,
              dtSeconds,
              settings.aimOneEuroMinCutoff,
              settings.aimOneEuroBeta,
              settings.aimOneEuroDerivativeCutoff)
        : selectedTarget.x;
    const double filteredTargetY = settings.aimOneEuroEnabled
        ? aimState.targetFilterY.Filter(
              selectedTarget.y,
              dtSeconds,
              settings.aimOneEuroMinCutoff,
              settings.aimOneEuroBeta,
              settings.aimOneEuroDerivativeCutoff)
        : selectedTarget.y;

    const double leadSeconds = std::clamp(settings.aimPredictionMs, 0.0, 250.0) / 1000.0;
    const double maxLeadX = (std::max)(0.0, selectedTarget.width * settings.aimPredictionMaxBoxFraction);
    const double maxLeadY = (std::max)(0.0, selectedTarget.height * settings.aimPredictionMaxBoxFraction);
    const double rawLeadX = std::clamp(selectedTarget.velocityX * leadSeconds, -maxLeadX, maxLeadX);
    const double rawLeadY = std::clamp(selectedTarget.velocityY * leadSeconds, -maxLeadY, maxLeadY);
    const double leadX = settings.aimOneEuroEnabled
        ? aimState.leadFilterX.Filter(
              rawLeadX,
              dtSeconds,
              settings.aimOneEuroMinCutoff,
              settings.aimOneEuroBeta,
              settings.aimOneEuroDerivativeCutoff)
        : rawLeadX;
    const double leadY = settings.aimOneEuroEnabled
        ? aimState.leadFilterY.Filter(
              rawLeadY,
              dtSeconds,
              settings.aimOneEuroMinCutoff,
              settings.aimOneEuroBeta,
              settings.aimOneEuroDerivativeCutoff)
        : rawLeadY;

    double bestTargetX = std::clamp(filteredTargetX + leadX, 0.0, static_cast<double>(frameWidth - 1));
    double bestTargetY = std::clamp(filteredTargetY + leadY, 0.0, static_cast<double>(frameHeight - 1));

    const double rawDeltaX = bestTargetX - centerX;
    const double rawDeltaY = bestTargetY - centerY;
    const bool outsideDeadzone =
        std::abs(rawDeltaX) >= settings.deadzonePixels ||
        std::abs(rawDeltaY) >= settings.deadzonePixels;

    command.hasTarget = true;
    debugState.originX = static_cast<int>(std::lround(centerX));
    debugState.originY = static_cast<int>(std::lround(centerY));
    command.targetX = static_cast<int>(std::lround(bestTargetX));
    command.targetY = static_cast<int>(std::lround(bestTargetY));
    debugState.targetLabel = selectedTarget.label;
    debugState.targetTrackId = selectedTarget.trackId;
    debugState.rawDeltaX = rawDeltaX;
    debugState.rawDeltaY = rawDeltaY;
    debugState.filteredTargetX = filteredTargetX;
    debugState.filteredTargetY = filteredTargetY;
    debugState.leadX = leadX;
    debugState.leadY = leadY;
    debugState.targetVelocityX = selectedTarget.velocityX;
    debugState.targetVelocityY = selectedTarget.velocityY;
    if (outsideDeadzone) {
        const int scaledDeltaX = static_cast<int>(std::lround(rawDeltaX * settings.gain));
        const int scaledDeltaY = static_cast<int>(std::lround(rawDeltaY * settings.gain));
        command.deltaX = std::clamp(scaledDeltaX, -settings.maxStep, settings.maxStep);
        command.deltaY = std::clamp(scaledDeltaY, -settings.maxStep, settings.maxStep);
    }
    command.active = outsideDeadzone;

    const double normalizedX = std::clamp(
        rawDeltaX / (centerX > 0.0f ? static_cast<double>(centerX) : 1.0),
        -1.0,
        1.0);
    const double normalizedY = std::clamp(
        rawDeltaY / (centerY > 0.0f ? static_cast<double>(centerY) : 1.0),
        -1.0,
        1.0);
    const double stickMaxPercent = std::clamp(settings.titanTwoStickMaxPercent, 0.0, 100.0);
    const double stickCurve = std::clamp(settings.titanTwoStickCurve, 0.25, 4.0);
    const double responseBoost = std::clamp(settings.titanTwoStickResponseBoost, 0.5, 4.0);
    double boostedNormalizedX = normalizedX * responseBoost;
    double boostedNormalizedY = normalizedY * responseBoost;
    if (isTitanTwoGamepadBackend && settings.titanTwoStickPdEnabled) {
        const double errorVelocityX = aimState.hasLastError ? (rawDeltaX - aimState.lastErrorX) / dtSeconds : 0.0;
        const double errorVelocityY = aimState.hasLastError ? (rawDeltaY - aimState.lastErrorY) / dtSeconds : 0.0;
        const double normalizedErrorVelocityX = errorVelocityX / (centerX > 0.0 ? centerX : 1.0);
        const double normalizedErrorVelocityY = errorVelocityY / (centerY > 0.0 ? centerY : 1.0);
        const double normalizedTargetVelocityX = selectedTarget.velocityX / (centerX > 0.0 ? centerX : 1.0);
        const double normalizedTargetVelocityY = selectedTarget.velocityY / (centerY > 0.0 ? centerY : 1.0);
        boostedNormalizedX += normalizedErrorVelocityX * settings.titanTwoStickDerivativeGain;
        boostedNormalizedY += normalizedErrorVelocityY * settings.titanTwoStickDerivativeGain;
        boostedNormalizedX += normalizedTargetVelocityX * settings.titanTwoStickFeedForward;
        boostedNormalizedY += normalizedTargetVelocityY * settings.titanTwoStickFeedForward;
    }
    boostedNormalizedX = std::clamp(boostedNormalizedX, -1.0, 1.0);
    boostedNormalizedY = std::clamp(boostedNormalizedY, -1.0, 1.0);
    double curvedX = outsideDeadzone ? ApplySignedCurve(boostedNormalizedX, stickCurve) : 0.0;
    double curvedY = outsideDeadzone ? ApplySignedCurve(boostedNormalizedY, stickCurve) : 0.0;
    double stickXPercent = curvedX * stickMaxPercent;
    double stickYPercent = curvedY * stickMaxPercent;
    if (outsideDeadzone && isTitanTwoGamepadBackend) {
        const double minimumStickPercent = std::clamp(settings.titanTwoStickMinPercent, 0.0, stickMaxPercent);
        const double minimumStickMagnitude = stickMaxPercent > 0.0 ? (minimumStickPercent / stickMaxPercent) : 0.0;
        if (std::abs(boostedNormalizedX) > 0.0) {
            curvedX = ApplyMinimumSignedMagnitude(curvedX, minimumStickMagnitude);
            stickXPercent = curvedX * stickMaxPercent;
        }
        if (std::abs(boostedNormalizedY) > 0.0) {
            curvedY = ApplyMinimumSignedMagnitude(curvedY, minimumStickMagnitude);
            stickYPercent = curvedY * stickMaxPercent;
        }
    }
    if (isTitanTwoGamepadBackend) {
        const double maxStickChange =
            std::clamp(settings.titanTwoStickSlewPercentPerSecond, 0.0, 5000.0) * dtSeconds;
        stickXPercent = ApplySlewLimit(aimState.lastStickXPercent, stickXPercent, maxStickChange);
        stickYPercent = ApplySlewLimit(aimState.lastStickYPercent, stickYPercent, maxStickChange);
    }
    command.stickXFixed = EncodeTitanTwoStickPercent(stickXPercent);
    command.stickYFixed = EncodeTitanTwoStickPercent(stickYPercent);
    command.active = outsideDeadzone || std::abs(stickXPercent) >= 0.1 || std::abs(stickYPercent) >= 0.1;

    aimState.hasLastError = true;
    aimState.lastErrorX = rawDeltaX;
    aimState.lastErrorY = rawDeltaY;
    aimState.lastStickXPercent = stickXPercent;
    aimState.lastStickYPercent = stickYPercent;
    aimState.lastFilteredTargetX = filteredTargetX;
    aimState.lastFilteredTargetY = filteredTargetY;

    debugState.hasTarget = command.hasTarget;
    debugState.active = command.active;
    debugState.targetX = command.targetX;
    debugState.targetY = command.targetY;
    debugState.deltaX = command.deltaX;
    debugState.deltaY = command.deltaY;
    debugState.stickXPercent = stickXPercent;
    debugState.stickYPercent = stickYPercent;
    debugState.score = static_cast<float>(selectedTarget.score);
    return command;
}

std::unique_ptr<IMovementOutput> CreateMovementOutput(MovementBackendKind backend) {
    switch (backend) {
    case MovementBackendKind::TitanTwoGcv:
    case MovementBackendKind::TitanTwoGcvGamepad:
        return std::make_unique<TitanTwoGcvMovementOutput>();
    case MovementBackendKind::TitanTwoHidDirect:
        return std::make_unique<TitanTwoHidDirectMovementOutput>();
    case MovementBackendKind::None:
    default:
        return std::make_unique<NullMovementOutput>();
    }
}

} // namespace

struct MovementController::Impl {
    MovementSettings settings;
    MovementDebugState debugState;
    std::unique_ptr<IMovementOutput> output;
    MovementAimState aimState;
    std::uint64_t sequence = 0;

    void Reset() {
        if (output) {
            output->Release();
        }
        output.reset();
        settings = MovementSettings{};
        debugState = MovementDebugState{};
        aimState.ResetRuntime();
        aimState.nextTrackId = 1;
        sequence = 0;
    }
};

MovementController::MovementController()
    : m_impl(std::make_unique<Impl>()) {
}

MovementController::~MovementController() {
    Release();
}

bool MovementController::Initialize(const MovementSettings& settings) {
    Release();

    auto output = CreateMovementOutput(settings.backend);
    std::string error;
    if (!output->Initialize(settings, error)) {
        m_lastError = error.empty() ? "Movement output initialization failed." : error;
        return false;
    }

    m_impl->settings = settings;
    m_impl->output = std::move(output);
    m_impl->debugState = MovementDebugState{};
    m_impl->aimState.ResetRuntime();
    m_impl->aimState.nextTrackId = 1;
    m_impl->sequence = 0;
    m_lastError.clear();
    return true;
}

bool MovementController::SubmitFromDetections(
    int frameWidth,
    int frameHeight,
    const std::vector<DetectionObject>& detections) {
    if (!m_impl->output) {
        m_lastError = "Movement output is not initialized.";
        return false;
    }

    MovementCommand command = BuildMovementCommand(
        m_impl->settings,
        m_impl->aimState,
        frameWidth,
        frameHeight,
        detections,
        m_impl->debugState);
    command.sequence = static_cast<std::uint32_t>(++m_impl->sequence);
    m_impl->debugState.sequence = m_impl->sequence;

    std::string error;
    if (!m_impl->output->Submit(command, error)) {
        m_lastError = error.empty() ? "Movement output submission failed." : error;
        return false;
    }

    const MovementTransportTelemetry telemetry = m_impl->output->GetTelemetry();
    m_impl->debugState.transportReady = telemetry.ready;
    m_impl->debugState.transportDispatchCount = telemetry.dispatchCount;
    m_impl->debugState.transportSkipCount = telemetry.skipCount;
    m_impl->debugState.transportLastSubmitMs = telemetry.lastSubmitMs;
    m_impl->debugState.transportUserIndex = telemetry.userIndex;

    m_lastError.clear();
    return true;
}

void MovementController::UpdateRuntimeSettings(const MovementSettings& settings) {
    if (!m_impl) {
        return;
    }

    MovementSettings updatedSettings = m_impl->settings;
    updatedSettings.gain = settings.gain;
    updatedSettings.maxStep = settings.maxStep;
    updatedSettings.deadzonePixels = settings.deadzonePixels;
    updatedSettings.verticalBias = settings.verticalBias;
    updatedSettings.centerOffsetX = settings.centerOffsetX;
    updatedSettings.centerOffsetY = settings.centerOffsetY;
    updatedSettings.titanTwoHoldMs = settings.titanTwoHoldMs;
    updatedSettings.titanTwoStickMaxPercent = settings.titanTwoStickMaxPercent;
    updatedSettings.titanTwoStickCurve = settings.titanTwoStickCurve;
    updatedSettings.titanTwoStickResponseBoost = settings.titanTwoStickResponseBoost;
    updatedSettings.titanTwoStickMinPercent = settings.titanTwoStickMinPercent;
    updatedSettings.aimTrackingEnabled = settings.aimTrackingEnabled;
    updatedSettings.aimTrackConfirmFrames = settings.aimTrackConfirmFrames;
    updatedSettings.aimTrackLostFrames = settings.aimTrackLostFrames;
    updatedSettings.aimTrackMatchMaxCost = settings.aimTrackMatchMaxCost;
    updatedSettings.aimTargetLockBonus = settings.aimTargetLockBonus;
    updatedSettings.aimTargetSwitchMargin = settings.aimTargetSwitchMargin;
    updatedSettings.aimOneEuroEnabled = settings.aimOneEuroEnabled;
    updatedSettings.aimOneEuroMinCutoff = settings.aimOneEuroMinCutoff;
    updatedSettings.aimOneEuroBeta = settings.aimOneEuroBeta;
    updatedSettings.aimOneEuroDerivativeCutoff = settings.aimOneEuroDerivativeCutoff;
    updatedSettings.aimPredictionMs = settings.aimPredictionMs;
    updatedSettings.aimPredictionMaxBoxFraction = settings.aimPredictionMaxBoxFraction;
    updatedSettings.titanTwoStickPdEnabled = settings.titanTwoStickPdEnabled;
    updatedSettings.titanTwoStickDerivativeGain = settings.titanTwoStickDerivativeGain;
    updatedSettings.titanTwoStickFeedForward = settings.titanTwoStickFeedForward;
    updatedSettings.titanTwoStickSlewPercentPerSecond = settings.titanTwoStickSlewPercentPerSecond;
    updatedSettings.fovRadius = settings.fovRadius;
    updatedSettings.movementTestStickEnabled = settings.movementTestStickEnabled;
    updatedSettings.movementTestStickXPercent = settings.movementTestStickXPercent;
    updatedSettings.movementTestStickYPercent = settings.movementTestStickYPercent;
    m_impl->settings = updatedSettings;
}

void MovementController::Release() {
    if (m_impl) {
        m_impl->Reset();
    }
    m_lastError.clear();
}

const MovementSettings& MovementController::GetSettings() const {
    return m_impl->settings;
}

const MovementDebugState& MovementController::GetDebugState() const {
    return m_impl->debugState;
}

const char* MovementBackendKindToString(MovementBackendKind backend) {
    switch (backend) {
    case MovementBackendKind::TitanTwoGcv:
        return "TitanTwoGcv";
    case MovementBackendKind::TitanTwoGcvGamepad:
        return "TitanTwoGcvGamepad";
    case MovementBackendKind::TitanTwoHidDirect:
        return "TitanTwoHidDirect";
    case MovementBackendKind::None:
    default:
        return "None";
    }
}

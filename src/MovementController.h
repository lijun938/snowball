#pragma once

#include "DetectionTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class MovementBackendKind {
    None,
    TitanTwoGcv,
    TitanTwoGcvGamepad,
    TitanTwoHidDirect,
};

struct MovementSettings {
    MovementBackendKind backend = MovementBackendKind::None;
    double gain = 0.35;
    int maxStep = 80;
    double deadzonePixels = 6.0;
    double verticalBias = -0.20;
    double centerOffsetX = 0.0;
    double centerOffsetY = 0.0;
    std::string titanTwoGcvPath = "titan_two_gcv.bin";
    int titanTwoHoldMs = 16;
    double titanTwoStickMaxPercent = 100.0;
    double titanTwoStickCurve = 1.15;
    double titanTwoStickResponseBoost = 2.0;
    double titanTwoStickMinPercent = 6.5;
    bool aimTrackingEnabled = true;
    int aimTrackConfirmFrames = 1;
    int aimTrackLostFrames = 10;
    double aimTrackMatchMaxCost = 2.0;
    double aimTargetLockBonus = 1.5;
    double aimTargetSwitchMargin = 0.6;
    bool aimOneEuroEnabled = true;
    double aimOneEuroMinCutoff = 1.2;
    double aimOneEuroBeta = 0.04;
    double aimOneEuroDerivativeCutoff = 1.0;
    double aimPredictionMs = 45.0;
    double aimPredictionMaxBoxFraction = 0.35;
    bool titanTwoStickPdEnabled = true;
    double titanTwoStickDerivativeGain = 0.035;
    double titanTwoStickFeedForward = 0.020;
    double titanTwoStickSlewPercentPerSecond = 650.0;
    double fovRadius = 0.0;
    bool movementTestStickEnabled = false;
    double movementTestStickXPercent = 0.0;
    double movementTestStickYPercent = 0.0;
    int titanTwoHidVendorId = 0x2508;
    int titanTwoHidProductId = 0x0032;
    int titanTwoHidReportSize = 0;
    int titanTwoHidPayloadOffset = 1;
    int titanTwoHidReportId = 0;
};

struct MovementDebugState {
    bool hasTarget = false;
    bool active = false;
    int originX = 0;
    int originY = 0;
    int targetX = 0;
    int targetY = 0;
    int targetLabel = -1;
    int targetTrackId = -1;
    int deltaX = 0;
    int deltaY = 0;
    double rawDeltaX = 0.0;
    double rawDeltaY = 0.0;
    double filteredTargetX = 0.0;
    double filteredTargetY = 0.0;
    double leadX = 0.0;
    double leadY = 0.0;
    double targetVelocityX = 0.0;
    double targetVelocityY = 0.0;
    double stickXPercent = 0.0;
    double stickYPercent = 0.0;
    float score = 0.0f;
    std::uint64_t sequence = 0;
    bool transportReady = false;
    std::uint64_t transportDispatchCount = 0;
    std::uint64_t transportSkipCount = 0;
    double transportLastSubmitMs = 0.0;
    int transportUserIndex = -1;
};

class MovementController {
public:
    MovementController();
    ~MovementController();

    bool Initialize(const MovementSettings& settings);
    bool SubmitFromDetections(int frameWidth, int frameHeight, const std::vector<DetectionObject>& detections);
    void UpdateRuntimeSettings(const MovementSettings& settings);
    void Release();

    const std::string& GetLastError() const { return m_lastError; }
    const MovementSettings& GetSettings() const;
    const MovementDebugState& GetDebugState() const;

private:
    struct Impl;

    std::unique_ptr<Impl> m_impl;
    std::string m_lastError;
};

const char* MovementBackendKindToString(MovementBackendKind backend);

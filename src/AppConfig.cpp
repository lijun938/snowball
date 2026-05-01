#include "AppConfig.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool SplitKeyValue(const std::string& arg, std::string& key, std::string& value) {
    const std::size_t pos = arg.find('=');
    if (pos == std::string::npos) {
        return false;
    }

    key = arg.substr(0, pos);
    value = arg.substr(pos + 1);
    return true;
}

bool ParseInt(const std::string& value, int& result) {
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || (end != nullptr && *end != '\0')) {
        return false;
    }

    result = static_cast<int>(parsed);
    return true;
}

bool ParseDouble(const std::string& value, double& result) {
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || (end != nullptr && *end != '\0')) {
        return false;
    }

    result = parsed;
    return true;
}

bool ParseSignedPercent(const std::string& value, double& result) {
    if (!ParseDouble(value, result) || !std::isfinite(result) || result < -100.0 || result > 100.0) {
        return false;
    }
    return true;
}

bool ParseTensorRtRoi(const std::string& value, TensorRtRoiSettings& roi) {
    std::stringstream ss(value);
    std::string part;
    int parsed[4] = {};
    int index = 0;
    while (std::getline(ss, part, ',')) {
        if (index >= 4 || !ParseInt(part, parsed[index])) {
            return false;
        }
        ++index;
    }

    if (index != 4) {
        return false;
    }
    if (parsed[0] < 0 || parsed[1] < 0 || parsed[2] <= 0 || parsed[3] <= 0) {
        return false;
    }

    roi.enabled = true;
    roi.x = parsed[0];
    roi.y = parsed[1];
    roi.width = parsed[2];
    roi.height = parsed[3];
    return true;
}

bool ParseBackend(const std::string& value, InferenceBackendKind& backend) {
    const std::string normalized = ToLower(value);
    if (normalized == "trt" ||
        normalized == "tensorrt" ||
        normalized == "trt-yolo" ||
        normalized == "tensorrt-yolo") {
        backend = InferenceBackendKind::TensorRTYolo;
        return true;
    }
    if (normalized == "trt-legacy" ||
        normalized == "tensorrt-legacy" ||
        normalized == "legacy-trt") {
        backend = InferenceBackendKind::TensorRTLegacy;
        return true;
    }
    if (normalized == "trt-preferred" || normalized == "tensorrt-preferred" || normalized == "auto") {
        backend = InferenceBackendKind::TensorRTPreferred;
        return true;
    }
    return false;
}

bool ParseTensorRTYoloInputColor(const std::string& value, TensorRTYoloInputColor& inputColor) {
    const std::string normalized = ToLower(value);
    if (normalized == "bgr" || normalized == "bgr24") {
        inputColor = TensorRTYoloInputColor::BGR;
        return true;
    }
    if (normalized == "rgb" || normalized == "rgb24") {
        inputColor = TensorRTYoloInputColor::RGB;
        return true;
    }
    return false;
}

bool ParseMovementBackend(const std::string& value, MovementBackendKind& backend) {
    const std::string normalized = ToLower(value);
    if (normalized == "none" || normalized == "off" || normalized == "disabled") {
        backend = MovementBackendKind::None;
        return true;
    }
    if (normalized == "titan-two" || normalized == "titantwo" || normalized == "titan-two-gcv" || normalized == "t2-gcv") {
        backend = MovementBackendKind::TitanTwoGcv;
        return true;
    }
    if (normalized == "titan-two-gamepad" || normalized == "t2-pad" || normalized == "t2-gamepad") {
        backend = MovementBackendKind::TitanTwoGcvGamepad;
        return true;
    }
    if (normalized == "titan-two-hid" || normalized == "t2-hid" || normalized == "hid-direct" || normalized == "tt2-hid") {
        backend = MovementBackendKind::TitanTwoHidDirect;
        return true;
    }
    return false;
}

bool ParseQcapInput(const std::string& value, ULONG& inputType) {
    const std::string normalized = ToLower(value);
    if (normalized == "hdmi") {
        inputType = QCAP_INPUT_TYPE_HDMI;
        return true;
    }
    if (normalized == "dvi") {
        inputType = QCAP_INPUT_TYPE_DVI_D;
        return true;
    }
    if (normalized == "vga" || normalized == "rgb") {
        inputType = QCAP_INPUT_TYPE_VGA;
        return true;
    }
    if (normalized == "sdi") {
        inputType = QCAP_INPUT_TYPE_SDI;
        return true;
    }
    if (normalized == "displayport" || normalized == "dp") {
        inputType = QCAP_INPUT_TYPE_DISPLAY_PORT;
        return true;
    }
    if (normalized == "auto") {
        inputType = QCAP_INPUT_TYPE_AUTO;
        return true;
    }
    return false;
}

bool ParseBool(const std::string& value, bool& result) {
    const std::string normalized = ToLower(value);
    if (normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "on") {
        result = true;
        return true;
    }
    if (normalized == "false" || normalized == "0" || normalized == "no" || normalized == "off") {
        result = false;
        return true;
    }
    return false;
}

std::string Trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::filesystem::path GetExecutableDirectory() {
    wchar_t pathBuf[MAX_PATH] = {};
    const DWORD len = GetModuleFileNameW(nullptr, pathBuf, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return std::filesystem::current_path();
    return std::filesystem::path(pathBuf).parent_path();
}

bool ParseQcapColorSpace(const std::string& value, ULONG& colorSpaceType) {
    const std::string normalized = ToLower(value);
    if (normalized == "bgr24" || normalized == "bgr") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_BGR24;
        return true;
    }
    if (normalized == "rgb24" || normalized == "rgb") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_RGB24;
        return true;
    }
    if (normalized == "yuy2") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_YUY2;
        return true;
    }
    if (normalized == "uyvy") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_UYVY;
        return true;
    }
    if (normalized == "nv12") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_NV12;
        return true;
    }
    if (normalized == "i420") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_I420;
        return true;
    }
    if (normalized == "yv12") {
        colorSpaceType = QCAP_COLORSPACE_TYPE_YV12;
        return true;
    }
    return false;
}

} // namespace

bool LoadConfigFromIniFile(const std::string& iniPath, AppConfig& config, std::string& errorMessage) {
    std::ifstream file{iniPath};
    if (!file.is_open()) {
        errorMessage = "Cannot open INI file: " + iniPath;
        return false;
    }

    std::string currentSection;
    std::string line;
    int lineNum = 0;

    while (std::getline(file, line)) {
        ++lineNum;
        line = Trim(line);

        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        if (line.front() == '[' && line.back() == ']') {
            currentSection = ToLower(Trim(line.substr(1, line.size() - 2)));
            continue;
        }

        std::string key, value;
        if (!SplitKeyValue(line, key, value)) {
            continue;
        }
        key = Trim(ToLower(key));
        value = Trim(value);

        if (currentSection == "detector") {
            if (key == "backend") {
                if (value.empty()) continue;
                if (!ParseBackend(value, config.detector.backend)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid backend: " + value;
                    return false;
                }
            } else if (key == "engine") {
                config.detector.enginePath = value;
            } else if (key == "model") {
                config.detector.modelPath = value;
            } else if (key == "device") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.detector.deviceIndex)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid device index: " + value;
                    return false;
                }
            } else if (key == "threads") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.detector.numThreads)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid thread count: " + value;
                    return false;
                }
            } else if (key == "candidate_topk") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.detector.tensorRtCandidateTopK)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid candidate topk: " + value;
                    return false;
                }
            } else if (key == "input_color") {
                if (value.empty()) continue;
                if (!ParseTensorRTYoloInputColor(value, config.detector.tensorRtYoloInputColor)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid input color: " + value;
                    return false;
                }
            } else if (key == "roi") {
                if (value.empty()) continue;
                if (!ParseTensorRtRoi(value, config.detector.tensorRtRoi)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid roi: " + value;
                    return false;
                }
            }
        } else if (currentSection == "capture") {
            if (key == "width") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.capture.width)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid capture width: " + value;
                    return false;
                }
            } else if (key == "height") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.capture.height)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid capture height: " + value;
                    return false;
                }
            } else if (key == "device") {
                if (!value.empty()) config.capture.deviceName = value;
            } else if (key == "index") {
                if (value.empty()) continue;
                int parsed = 0;
                if (!ParseInt(value, parsed) || parsed < 0) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid capture index: " + value;
                    return false;
                }
                config.capture.deviceIndex = static_cast<unsigned int>(parsed);
            } else if (key == "input") {
                if (value.empty()) continue;
                if (!ParseQcapInput(value, config.capture.videoInput)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid capture input: " + value;
                    return false;
                }
            } else if (key == "fps") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.capture.frameRate)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid capture fps: " + value;
                    return false;
                }
            } else if (key == "colorspace") {
                if (value.empty()) continue;
                if (!ParseQcapColorSpace(value, config.capture.outputColorSpace)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid capture colorspace: " + value;
                    return false;
                }
            } else if (key == "gpudirect") {
                if (value.empty()) continue;
                if (!ParseBool(value, config.capture.requestGpuDirect)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid gpudirect value: " + value;
                    return false;
                }
            }
        } else if (currentSection == "movement") {
            if (key == "backend") {
                if (value.empty()) continue;
                if (!ParseMovementBackend(value, config.movement.backend)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid movement backend: " + value;
                    return false;
                }
            } else if (key == "vertical_bias") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.verticalBias)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid vertical_bias: " + value;
                    return false;
                }
            } else if (key == "deadzone") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.deadzonePixels)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid deadzone: " + value;
                    return false;
                }
            } else if (key == "center_offset_x") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.centerOffsetX)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid center_offset_x: " + value;
                    return false;
                }
            } else if (key == "center_offset_y") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.centerOffsetY)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid center_offset_y: " + value;
                    return false;
                }
            } else if (key == "titan_two_gcv_path") {
                config.movement.titanTwoGcvPath = value;
            } else if (key == "titan_two_hold_ms") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.titanTwoHoldMs)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid titan_two_hold_ms: " + value;
                    return false;
                }
            } else if (key == "stick_max_percent") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickMaxPercent)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_max_percent: " + value;
                    return false;
                }
            } else if (key == "stick_curve") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickCurve)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_curve: " + value;
                    return false;
                }
            } else if (key == "stick_response_boost") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickResponseBoost)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_response_boost: " + value;
                    return false;
                }
            } else if (key == "stick_min_percent") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickMinPercent)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_min_percent: " + value;
                    return false;
                }
            } else if (key == "aim_tracking") {
                if (value.empty()) continue;
                if (!ParseBool(value, config.movement.aimTrackingEnabled)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_tracking: " + value;
                    return false;
                }
            } else if (key == "aim_track_confirm_frames") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.aimTrackConfirmFrames)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_track_confirm_frames: " + value;
                    return false;
                }
            } else if (key == "aim_track_lost_frames") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.aimTrackLostFrames)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_track_lost_frames: " + value;
                    return false;
                }
            } else if (key == "aim_track_match_max_cost") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimTrackMatchMaxCost)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_track_match_max_cost: " + value;
                    return false;
                }
            } else if (key == "aim_lock_bonus") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimTargetLockBonus)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_lock_bonus: " + value;
                    return false;
                }
            } else if (key == "aim_switch_margin") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimTargetSwitchMargin)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_switch_margin: " + value;
                    return false;
                }
            } else if (key == "aim_one_euro") {
                if (value.empty()) continue;
                if (!ParseBool(value, config.movement.aimOneEuroEnabled)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_one_euro: " + value;
                    return false;
                }
            } else if (key == "aim_one_euro_min_cutoff") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimOneEuroMinCutoff)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_one_euro_min_cutoff: " + value;
                    return false;
                }
            } else if (key == "aim_one_euro_beta") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimOneEuroBeta)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_one_euro_beta: " + value;
                    return false;
                }
            } else if (key == "aim_one_euro_d_cutoff") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimOneEuroDerivativeCutoff)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_one_euro_d_cutoff: " + value;
                    return false;
                }
            } else if (key == "aim_prediction_ms") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimPredictionMs)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_prediction_ms: " + value;
                    return false;
                }
            } else if (key == "aim_prediction_max_box") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.aimPredictionMaxBoxFraction)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid aim_prediction_max_box: " + value;
                    return false;
                }
            } else if (key == "stick_pd") {
                if (value.empty()) continue;
                if (!ParseBool(value, config.movement.titanTwoStickPdEnabled)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_pd: " + value;
                    return false;
                }
            } else if (key == "stick_d_gain") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickDerivativeGain)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_d_gain: " + value;
                    return false;
                }
            } else if (key == "stick_feed_forward") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickFeedForward)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_feed_forward: " + value;
                    return false;
                }
            } else if (key == "stick_slew") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.titanTwoStickSlewPercentPerSecond)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid stick_slew: " + value;
                    return false;
                }
            } else if (key == "fov_radius") {
                if (value.empty()) continue;
                if (!ParseDouble(value, config.movement.fovRadius)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid fov_radius: " + value;
                    return false;
                }
            } else if (key == "tt2_hid_vid") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.titanTwoHidVendorId)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid tt2_hid_vid: " + value;
                    return false;
                }
            } else if (key == "tt2_hid_pid") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.titanTwoHidProductId)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid tt2_hid_pid: " + value;
                    return false;
                }
            } else if (key == "tt2_hid_report_size") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.titanTwoHidReportSize)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid tt2_hid_report_size: " + value;
                    return false;
                }
            } else if (key == "tt2_hid_payload_offset") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.titanTwoHidPayloadOffset)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid tt2_hid_payload_offset: " + value;
                    return false;
                }
            } else if (key == "tt2_hid_report_id") {
                if (value.empty()) continue;
                if (!ParseInt(value, config.movement.titanTwoHidReportId)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid tt2_hid_report_id: " + value;
                    return false;
                }
            }
        } else if (currentSection == "general") {
            if (key == "headless") {
                if (value.empty()) continue;
                if (!ParseBool(value, config.headless)) {
                    errorMessage = "config.ini:" + std::to_string(lineNum) + " invalid headless value: " + value;
                    return false;
                }
            }
        }
    }

    return true;
}

bool ParseAppConfig(int argc, char** argv, AppConfig& config, std::string& errorMessage) {
    config = AppConfig{};
    errorMessage.clear();

    const auto iniPath = GetExecutableDirectory() / "config.ini";
    if (std::filesystem::exists(iniPath)) {
        if (!LoadConfigFromIniFile(iniPath.string(), config, errorMessage)) {
            return false;
        }
        std::cout << "[Config] Loaded config.ini from " << iniPath.string() << std::endl;
    } else {
        std::cout << "[Config] No config.ini found, using defaults + CLI" << std::endl;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            config.showHelp = true;
            return true;
        }
        if (arg == "--headless") {
            config.headless = true;
            continue;
        }
        if (arg == "--inference-screenshot-detections-only") {
            config.inferenceScreenshotDetectionsOnly = true;
            continue;
        }
        if (arg == "--no-gpudirect") {
            config.capture.requestGpuDirect = false;
            continue;
        }
        if (arg == "--allow-trt-build") {
            config.detector.allowTensorRtEngineBuild = true;
            continue;
        }
        if (arg == "--no-aim-tracking") {
            config.movement.aimTrackingEnabled = false;
            continue;
        }
        if (arg == "--no-aim-one-euro") {
            config.movement.aimOneEuroEnabled = false;
            continue;
        }
        if (arg == "--no-stick-pd") {
            config.movement.titanTwoStickPdEnabled = false;
            continue;
        }

        std::string key;
        std::string value;
        if (!SplitKeyValue(arg, key, value)) {
            errorMessage = "Unsupported argument: " + arg;
            return false;
        }

        if (key == "--backend") {
            if (!ParseBackend(value, config.detector.backend)) {
                errorMessage = "Unsupported backend: " + value;
                return false;
            }
            continue;
        }
        if (key == "--model") {
            config.detector.modelPath = value;
            continue;
        }
        if (key == "--engine") {
            config.detector.enginePath = value;
            continue;
        }
        if (key == "--device") {
            if (!ParseInt(value, config.detector.deviceIndex)) {
                errorMessage = "Invalid detector device index: " + value;
                return false;
            }
            continue;
        }
        if (key == "--movement") {
            if (!ParseMovementBackend(value, config.movement.backend)) {
                errorMessage = "Unsupported movement backend: " + value;
                return false;
            }
            continue;
        }
        if (key == "--move-gain") {
            if (!ParseDouble(value, config.movement.gain) || !std::isfinite(config.movement.gain) || config.movement.gain < 0.0) {
                errorMessage = "Invalid move gain: " + value;
                return false;
            }
            continue;
        }
        if (key == "--move-max-step") {
            if (!ParseInt(value, config.movement.maxStep) || config.movement.maxStep < 0) {
                errorMessage = "Invalid move max step: " + value;
                return false;
            }
            continue;
        }
        if (key == "--move-deadzone") {
            if (!ParseDouble(value, config.movement.deadzonePixels) || !std::isfinite(config.movement.deadzonePixels) || config.movement.deadzonePixels < 0.0) {
                errorMessage = "Invalid move deadzone: " + value;
                return false;
            }
            continue;
        }
        if (key == "--move-vertical-bias") {
            if (!ParseDouble(value, config.movement.verticalBias) || !std::isfinite(config.movement.verticalBias)) {
                errorMessage = "Invalid move vertical bias: " + value;
                return false;
            }
            continue;
        }
        if (key == "--move-center-offset-x") {
            if (!ParseDouble(value, config.movement.centerOffsetX) || !std::isfinite(config.movement.centerOffsetX)) {
                errorMessage = "Invalid move center offset X: " + value;
                return false;
            }
            continue;
        }
        if (key == "--move-center-offset-y") {
            if (!ParseDouble(value, config.movement.centerOffsetY) || !std::isfinite(config.movement.centerOffsetY)) {
                errorMessage = "Invalid move center offset Y: " + value;
                return false;
            }
            continue;
        }
        if (key == "--titan-two-gcv-path") {
            config.movement.titanTwoGcvPath = value;
            continue;
        }
        if (key == "--titan-two-hold-ms") {
            if (!ParseInt(value, config.movement.titanTwoHoldMs) || config.movement.titanTwoHoldMs < 0) {
                errorMessage = "Invalid Titan Two hold ms: " + value;
                return false;
            }
            continue;
        }
        if (key == "--titan-two-stick-max-percent") {
            if (!ParseDouble(value, config.movement.titanTwoStickMaxPercent) ||
                !std::isfinite(config.movement.titanTwoStickMaxPercent) ||
                config.movement.titanTwoStickMaxPercent < 0.0 ||
                config.movement.titanTwoStickMaxPercent > 100.0) {
                errorMessage = "Invalid Titan Two stick max percent: " + value;
                return false;
            }
            continue;
        }
        if (key == "--titan-two-stick-curve") {
            if (!ParseDouble(value, config.movement.titanTwoStickCurve) ||
                !std::isfinite(config.movement.titanTwoStickCurve) ||
                config.movement.titanTwoStickCurve <= 0.0 ||
                config.movement.titanTwoStickCurve > 4.0) {
                errorMessage = "Invalid Titan Two stick curve: " + value;
                return false;
            }
            continue;
        }
        if (key == "--titan-two-stick-response-boost") {
            if (!ParseDouble(value, config.movement.titanTwoStickResponseBoost) ||
                !std::isfinite(config.movement.titanTwoStickResponseBoost) ||
                config.movement.titanTwoStickResponseBoost <= 0.0 ||
                config.movement.titanTwoStickResponseBoost > 4.0) {
                errorMessage = "Invalid Titan Two stick response boost: " + value;
                return false;
            }
            continue;
        }
        if (key == "--titan-two-stick-min-percent") {
            if (!ParseDouble(value, config.movement.titanTwoStickMinPercent) ||
                !std::isfinite(config.movement.titanTwoStickMinPercent) ||
                config.movement.titanTwoStickMinPercent < 0.0 ||
                config.movement.titanTwoStickMinPercent > 100.0) {
                errorMessage = "Invalid Titan Two stick min percent: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-track-confirm-frames") {
            if (!ParseInt(value, config.movement.aimTrackConfirmFrames) || config.movement.aimTrackConfirmFrames < 1) {
                errorMessage = "Invalid aim track confirm frames: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-track-lost-frames") {
            if (!ParseInt(value, config.movement.aimTrackLostFrames) || config.movement.aimTrackLostFrames < 0) {
                errorMessage = "Invalid aim track lost frames: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-track-match-max-cost") {
            if (!ParseDouble(value, config.movement.aimTrackMatchMaxCost) ||
                !std::isfinite(config.movement.aimTrackMatchMaxCost) ||
                config.movement.aimTrackMatchMaxCost <= 0.0) {
                errorMessage = "Invalid aim track match max cost: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-lock-bonus") {
            if (!ParseDouble(value, config.movement.aimTargetLockBonus) ||
                !std::isfinite(config.movement.aimTargetLockBonus) ||
                config.movement.aimTargetLockBonus < 0.0) {
                errorMessage = "Invalid aim lock bonus: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-switch-margin") {
            if (!ParseDouble(value, config.movement.aimTargetSwitchMargin) ||
                !std::isfinite(config.movement.aimTargetSwitchMargin) ||
                config.movement.aimTargetSwitchMargin < 0.0) {
                errorMessage = "Invalid aim switch margin: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-one-euro-min-cutoff") {
            if (!ParseDouble(value, config.movement.aimOneEuroMinCutoff) ||
                !std::isfinite(config.movement.aimOneEuroMinCutoff) ||
                config.movement.aimOneEuroMinCutoff <= 0.0) {
                errorMessage = "Invalid aim One Euro min cutoff: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-one-euro-beta") {
            if (!ParseDouble(value, config.movement.aimOneEuroBeta) ||
                !std::isfinite(config.movement.aimOneEuroBeta) ||
                config.movement.aimOneEuroBeta < 0.0) {
                errorMessage = "Invalid aim One Euro beta: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-one-euro-d-cutoff") {
            if (!ParseDouble(value, config.movement.aimOneEuroDerivativeCutoff) ||
                !std::isfinite(config.movement.aimOneEuroDerivativeCutoff) ||
                config.movement.aimOneEuroDerivativeCutoff <= 0.0) {
                errorMessage = "Invalid aim One Euro derivative cutoff: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-prediction-ms") {
            if (!ParseDouble(value, config.movement.aimPredictionMs) ||
                !std::isfinite(config.movement.aimPredictionMs) ||
                config.movement.aimPredictionMs < 0.0 ||
                config.movement.aimPredictionMs > 250.0) {
                errorMessage = "Invalid aim prediction ms: " + value;
                return false;
            }
            continue;
        }
        if (key == "--aim-prediction-max-box-fraction") {
            if (!ParseDouble(value, config.movement.aimPredictionMaxBoxFraction) ||
                !std::isfinite(config.movement.aimPredictionMaxBoxFraction) ||
                config.movement.aimPredictionMaxBoxFraction < 0.0 ||
                config.movement.aimPredictionMaxBoxFraction > 1.0) {
                errorMessage = "Invalid aim prediction max box fraction: " + value;
                return false;
            }
            continue;
        }
        if (key == "--stick-d-gain") {
            if (!ParseDouble(value, config.movement.titanTwoStickDerivativeGain) ||
                !std::isfinite(config.movement.titanTwoStickDerivativeGain)) {
                errorMessage = "Invalid stick D gain: " + value;
                return false;
            }
            continue;
        }
        if (key == "--stick-feed-forward") {
            if (!ParseDouble(value, config.movement.titanTwoStickFeedForward) ||
                !std::isfinite(config.movement.titanTwoStickFeedForward)) {
                errorMessage = "Invalid stick feed-forward: " + value;
                return false;
            }
            continue;
        }
        if (key == "--stick-slew-percent-per-second") {
            if (!ParseDouble(value, config.movement.titanTwoStickSlewPercentPerSecond) ||
                !std::isfinite(config.movement.titanTwoStickSlewPercentPerSecond) ||
                config.movement.titanTwoStickSlewPercentPerSecond < 0.0) {
                errorMessage = "Invalid stick slew percent per second: " + value;
                return false;
            }
            continue;
        }
        if (key == "--fov-radius") {
            if (!ParseDouble(value, config.movement.fovRadius) ||
                !std::isfinite(config.movement.fovRadius) ||
                config.movement.fovRadius < 0.0) {
                errorMessage = "Invalid FOV radius: " + value;
                return false;
            }
            continue;
        }
        if (key == "--movement-test-stick-x") {
            if (!ParseSignedPercent(value, config.movement.movementTestStickXPercent)) {
                errorMessage = "Invalid movement test stick X percent: " + value;
                return false;
            }
            config.movement.movementTestStickEnabled = true;
            continue;
        }
        if (key == "--movement-test-stick-y") {
            if (!ParseSignedPercent(value, config.movement.movementTestStickYPercent)) {
                errorMessage = "Invalid movement test stick Y percent: " + value;
                return false;
            }
            config.movement.movementTestStickEnabled = true;
            continue;
        }
        if (key == "--threads") {
            if (!ParseInt(value, config.detector.numThreads)) {
                errorMessage = "Invalid thread count: " + value;
                return false;
            }
            continue;
        }
        if (key == "--trt-candidate-topk") {
            if (!ParseInt(value, config.detector.tensorRtCandidateTopK) || config.detector.tensorRtCandidateTopK < 0) {
                errorMessage = "Invalid TensorRT candidate top-k: " + value;
                return false;
            }
            continue;
        }
        if (key == "--trt-roi") {
            if (!ParseTensorRtRoi(value, config.detector.tensorRtRoi)) {
                errorMessage = "Invalid TensorRT ROI. Expected x,y,width,height with non-negative x/y and positive width/height: " + value;
                return false;
            }
            continue;
        }
        if (key == "--trtyolo-input-color" || key == "--trtyolo-input-colorspace") {
            if (!ParseTensorRTYoloInputColor(value, config.detector.tensorRtYoloInputColor)) {
                errorMessage = "Invalid TensorRT-YOLO input color. Expected bgr or rgb: " + value;
                return false;
            }
            continue;
        }
        if (key == "--qcap-device") {
            config.capture.deviceName = value;
            continue;
        }
        if (key == "--qcap-index") {
            int parsed = 0;
            if (!ParseInt(value, parsed) || parsed < 0) {
                errorMessage = "Invalid QCAP device index: " + value;
                return false;
            }
            config.capture.deviceIndex = static_cast<unsigned int>(parsed);
            continue;
        }
        if (key == "--qcap-input") {
            if (!ParseQcapInput(value, config.capture.videoInput)) {
                errorMessage = "Unsupported QCAP input: " + value;
                return false;
            }
            continue;
        }
        if (key == "--capture-width") {
            if (!ParseInt(value, config.capture.width)) {
                errorMessage = "Invalid capture width: " + value;
                return false;
            }
            continue;
        }
        if (key == "--capture-height") {
            if (!ParseInt(value, config.capture.height)) {
                errorMessage = "Invalid capture height: " + value;
                return false;
            }
            continue;
        }
        if (key == "--capture-fps") {
            if (!ParseDouble(value, config.capture.frameRate)) {
                errorMessage = "Invalid capture fps: " + value;
                return false;
            }
            continue;
        }
        if (key == "--capture-colorspace") {
            if (!ParseQcapColorSpace(value, config.capture.outputColorSpace)) {
                errorMessage = "Unsupported capture colorspace: " + value;
                return false;
            }
            continue;
        }
        if (key == "--max-frames") {
            if (!ParseInt(value, config.maxFrames)) {
                errorMessage = "Invalid max frame count: " + value;
                return false;
            }
            continue;
        }
        if (key == "--max-seconds") {
            if (!ParseDouble(value, config.maxSeconds)) {
                errorMessage = "Invalid max seconds: " + value;
                return false;
            }
            continue;
        }
        if (key == "--warmup-seconds") {
            if (!ParseDouble(value, config.warmupSeconds) ||
                !std::isfinite(config.warmupSeconds) ||
                config.warmupSeconds < 0.0) {
                errorMessage = "Invalid warmup seconds: " + value;
                return false;
            }
            continue;
        }
        if (key == "--inference-screenshot") {
            config.inferenceScreenshotPath = value;
            continue;
        }
        if (key == "--inference-screenshot-every") {
            if (!ParseInt(value, config.inferenceScreenshotEveryNFrames) ||
                config.inferenceScreenshotEveryNFrames < 0) {
                errorMessage = "Invalid inference screenshot cadence: " + value;
                return false;
            }
            continue;
        }
        if (key == "--debug-dump-qcap-frame") {
            config.debugDumpQcapFramePath = value;
            continue;
        }
        if (key == "--debug-force-gpudirect-rebind-frame") {
            if (!ParseInt(value, config.debugForceGpuDirectRebindFrame) ||
                config.debugForceGpuDirectRebindFrame < 0) {
                errorMessage = "Invalid debug GPUDirect rebind frame: " + value;
                return false;
            }
            continue;
        }

        errorMessage = "Unsupported argument: " + arg;
        return false;
    }

    return true;
}

void PrintAppUsage() {
    std::cout
        << "Usage: snowball.exe [options]\n"
        << "  --backend=trt|trt-legacy|trt-preferred\n"
        << "  --model=<onnx_path>  legacy TensorRT build input only; TensorRT-YOLO engines are built offline\n"
        << "  --engine=<plan_or_engine_path>\n"
        << "  --allow-trt-build  legacy TensorRT only; TensorRT-YOLO uses trtyolo-export + trtexec offline\n"
        << "  --device=<gpu_index>\n"
        << "  --threads=<num_threads>\n"
        << "  --trt-candidate-topk=<count>\n"
        << "  --trt-roi=<x,y,width,height>  crop TensorRT input before letterbox\n"
        << "  --trtyolo-input-color=bgr|rgb  bgr swaps RB before TensorRT-YOLO, rgb leaves channels as-is\n"
        << "  --movement=none|titan-two-gcv|titan-two-gamepad\n"
        << "  --move-gain=<scale>\n"
        << "  --move-max-step=<pixels>\n"
        << "  --move-deadzone=<pixels>\n"
        << "  --move-vertical-bias=<ratio>\n"
        << "  --move-center-offset-x=<pixels>\n"
        << "  --move-center-offset-y=<pixels>\n"
        << "  --titan-two-gcv-path=<file_path>\n"
        << "  --titan-two-hold-ms=<milliseconds>\n"
        << "  --titan-two-stick-max-percent=<0..100>\n"
        << "  --titan-two-stick-curve=<0..4>\n"
        << "  --titan-two-stick-response-boost=<0..4>\n"
        << "  --titan-two-stick-min-percent=<0..100>\n"
        << "  --no-aim-tracking\n"
        << "  --aim-track-confirm-frames=<frames>\n"
        << "  --aim-track-lost-frames=<frames>\n"
        << "  --aim-track-match-max-cost=<cost>\n"
        << "  --aim-lock-bonus=<score>\n"
        << "  --aim-switch-margin=<score>\n"
        << "  --no-aim-one-euro\n"
        << "  --aim-one-euro-min-cutoff=<hz>\n"
        << "  --aim-one-euro-beta=<scale>\n"
        << "  --aim-one-euro-d-cutoff=<hz>\n"
        << "  --aim-prediction-ms=<milliseconds>\n"
        << "  --aim-prediction-max-box-fraction=<0..1>\n"
        << "  --no-stick-pd\n"
        << "  --stick-d-gain=<scale>\n"
        << "  --stick-feed-forward=<scale>\n"
        << "  --stick-slew-percent-per-second=<percent_per_second>\n"
        << "  --fov-radius=<pixels>  0=disabled, >0 limits aim to a circle of this radius from center\n"
        << "  --movement-test-stick-x=<-100..100>\n"
        << "  --movement-test-stick-y=<-100..100>\n"
        << "  --qcap-device=<device_name>\n"
        << "  --qcap-index=<index>\n"
        << "  --qcap-input=hdmi|dvi|vga|sdi|displayport|auto\n"
        << "  --capture-width=<width>\n"
        << "  --capture-height=<height>\n"
        << "  --capture-fps=<fps>\n"
        << "  --capture-colorspace=bgr24|rgb24|yuy2|uyvy|nv12|i420|yv12\n"
        << "  --headless\n"
        << "  --max-frames=<count>\n"
        << "  --max-seconds=<seconds>\n"
        << "  --warmup-seconds=<seconds>\n"
        << "  --inference-screenshot=<file_path>\n"
        << "  --inference-screenshot-detections-only\n"
        << "  --inference-screenshot-every=<frames>\n"
        << "  --debug-dump-qcap-frame=<file_or_directory>  writes one BGR/RGB comparison frame\n"
        << "  --debug-force-gpudirect-rebind-frame=<frame>\n"
        << "  --no-gpudirect\n";
}

const char* InferenceBackendKindToString(InferenceBackendKind backend) {
    switch (backend) {
    case InferenceBackendKind::TensorRTYolo:
        return "TensorRTYolo";
    case InferenceBackendKind::TensorRTLegacy:
        return "TensorRTLegacy";
    case InferenceBackendKind::TensorRTPreferred:
    default:
        return "TensorRTPreferred";
    }
}

const char* QcapInputToString(ULONG inputType) {
    switch (inputType) {
    case QCAP_INPUT_TYPE_HDMI:
        return "HDMI";
    case QCAP_INPUT_TYPE_DVI_D:
        return "DVI";
    case QCAP_INPUT_TYPE_VGA:
        return "VGA";
    case QCAP_INPUT_TYPE_SDI:
        return "SDI";
    case QCAP_INPUT_TYPE_DISPLAY_PORT:
        return "DisplayPort";
    case QCAP_INPUT_TYPE_AUTO:
        return "Auto";
    default:
        return "Unknown";
    }
}

const char* QcapColorSpaceToString(ULONG colorSpaceType) {
    switch (colorSpaceType) {
    case QCAP_COLORSPACE_TYPE_BGR24:
        return "BGR24";
    case QCAP_COLORSPACE_TYPE_RGB24:
        return "RGB24";
    case QCAP_COLORSPACE_TYPE_YUY2:
        return "YUY2";
    case QCAP_COLORSPACE_TYPE_UYVY:
        return "UYVY";
    case QCAP_COLORSPACE_TYPE_NV12:
        return "NV12";
    case QCAP_COLORSPACE_TYPE_I420:
        return "I420";
    case QCAP_COLORSPACE_TYPE_YV12:
        return "YV12";
    default:
        return "Unknown";
    }
}

namespace {

const char* MovementBackendToIniString(MovementBackendKind backend) {
    switch (backend) {
    case MovementBackendKind::TitanTwoGcv: return "titan-two-gcv";
    case MovementBackendKind::TitanTwoGcvGamepad: return "titan-two-gamepad";
    case MovementBackendKind::TitanTwoHidDirect: return "tt2-hid";
    case MovementBackendKind::None:
    default: return "none";
    }
}

const char* InferenceBackendToIniString(InferenceBackendKind backend) {
    switch (backend) {
    case InferenceBackendKind::TensorRTYolo: return "trt";
    case InferenceBackendKind::TensorRTLegacy: return "trt-legacy";
    case InferenceBackendKind::TensorRTPreferred:
    default: return "auto";
    }
}

std::string FormatDouble(double value, int precision) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << value;
    return oss.str();
}

std::string BoolStr(bool value) {
    return value ? "true" : "false";
}

} // namespace

bool SaveConfigToIniFile(const std::string& iniPath, const AppConfig& config, std::string& errorMessage) {
    std::ofstream file(iniPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        errorMessage = "Cannot open INI file for writing: " + iniPath;
        return false;
    }

    file << "; AmtoOBS configuration file\n";
    file << "; Auto-saved by Snowball\n\n";

    file << "[detector]\n";
    file << "backend=" << InferenceBackendToIniString(config.detector.backend) << "\n";
    file << "engine=" << config.detector.enginePath << "\n";
    file << "model=" << config.detector.modelPath << "\n";
    file << "device=" << config.detector.deviceIndex << "\n";
    file << "\n";

    file << "[capture]\n";
    file << "width=" << config.capture.width << "\n";
    file << "height=" << config.capture.height << "\n";
    file << "device=" << config.capture.deviceName << "\n";
    file << "index=" << config.capture.deviceIndex << "\n";
    file << "input=" << QcapInputToString(config.capture.videoInput) << "\n";
    file << "fps=" << FormatDouble(config.capture.frameRate, 0) << "\n";
    file << "colorspace=" << QcapColorSpaceToString(config.capture.outputColorSpace) << "\n";
    file << "gpudirect=" << BoolStr(config.capture.requestGpuDirect) << "\n";
    file << "\n";

    file << "[movement]\n";
    file << "backend=" << MovementBackendToIniString(config.movement.backend) << "\n";
    file << "vertical_bias=" << FormatDouble(config.movement.verticalBias, 3) << "\n";
    file << "deadzone=" << FormatDouble(config.movement.deadzonePixels, 2) << "\n";
    file << "center_offset_x=" << FormatDouble(config.movement.centerOffsetX, 2) << "\n";
    file << "center_offset_y=" << FormatDouble(config.movement.centerOffsetY, 2) << "\n";
    file << "titan_two_gcv_path=" << config.movement.titanTwoGcvPath << "\n";
    file << "titan_two_hold_ms=" << config.movement.titanTwoHoldMs << "\n";
    file << "stick_max_percent=" << FormatDouble(config.movement.titanTwoStickMaxPercent, 2) << "\n";
    file << "stick_curve=" << FormatDouble(config.movement.titanTwoStickCurve, 3) << "\n";
    file << "stick_response_boost=" << FormatDouble(config.movement.titanTwoStickResponseBoost, 3) << "\n";
    file << "stick_min_percent=" << FormatDouble(config.movement.titanTwoStickMinPercent, 2) << "\n";
    file << "aim_tracking=" << BoolStr(config.movement.aimTrackingEnabled) << "\n";
    file << "aim_track_confirm_frames=" << config.movement.aimTrackConfirmFrames << "\n";
    file << "aim_track_lost_frames=" << config.movement.aimTrackLostFrames << "\n";
    file << "aim_track_match_max_cost=" << FormatDouble(config.movement.aimTrackMatchMaxCost, 3) << "\n";
    file << "aim_lock_bonus=" << FormatDouble(config.movement.aimTargetLockBonus, 3) << "\n";
    file << "aim_switch_margin=" << FormatDouble(config.movement.aimTargetSwitchMargin, 3) << "\n";
    file << "aim_one_euro=" << BoolStr(config.movement.aimOneEuroEnabled) << "\n";
    file << "aim_one_euro_min_cutoff=" << FormatDouble(config.movement.aimOneEuroMinCutoff, 3) << "\n";
    file << "aim_one_euro_beta=" << FormatDouble(config.movement.aimOneEuroBeta, 4) << "\n";
    file << "aim_one_euro_d_cutoff=" << FormatDouble(config.movement.aimOneEuroDerivativeCutoff, 3) << "\n";
    file << "aim_prediction_ms=" << FormatDouble(config.movement.aimPredictionMs, 1) << "\n";
    file << "aim_prediction_max_box=" << FormatDouble(config.movement.aimPredictionMaxBoxFraction, 3) << "\n";
    file << "stick_pd=" << BoolStr(config.movement.titanTwoStickPdEnabled) << "\n";
    file << "stick_d_gain=" << FormatDouble(config.movement.titanTwoStickDerivativeGain, 4) << "\n";
    file << "stick_feed_forward=" << FormatDouble(config.movement.titanTwoStickFeedForward, 4) << "\n";
    file << "stick_slew=" << FormatDouble(config.movement.titanTwoStickSlewPercentPerSecond, 1) << "\n";
    file << "fov_radius=" << FormatDouble(config.movement.fovRadius, 1) << "\n";
    file << "tt2_hid_vid=" << config.movement.titanTwoHidVendorId << "\n";
    file << "tt2_hid_pid=" << config.movement.titanTwoHidProductId << "\n";
    file << "tt2_hid_report_size=" << config.movement.titanTwoHidReportSize << "\n";
    file << "tt2_hid_payload_offset=" << config.movement.titanTwoHidPayloadOffset << "\n";
    file << "tt2_hid_report_id=" << config.movement.titanTwoHidReportId << "\n";
    file << "\n";

    file << "[general]\n";
    file << "headless=" << BoolStr(config.headless) << "\n";

    file.flush();
    if (!file.good()) {
        errorMessage = "Failed to write INI file: " + iniPath;
        return false;
    }

    return true;
}

bool ExportConfigToJson(const std::string& jsonPath, const AppConfig& config, std::string& errorMessage) {
    std::ofstream file(jsonPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        errorMessage = "Cannot open JSON file for writing: " + jsonPath;
        return false;
    }

    file << "{\n";
    file << "  \"detector\": {\n";
    file << "    \"backend\": \"" << InferenceBackendToIniString(config.detector.backend) << "\",\n";
    file << "    \"engine\": \"" << config.detector.enginePath << "\",\n";
    file << "    \"model\": \"" << config.detector.modelPath << "\",\n";
    file << "    \"device\": " << config.detector.deviceIndex << "\n";
    file << "  },\n";

    file << "  \"capture\": {\n";
    file << "    \"width\": " << config.capture.width << ",\n";
    file << "    \"height\": " << config.capture.height << ",\n";
    file << "    \"device\": \"" << config.capture.deviceName << "\",\n";
    file << "    \"gpudirect\": " << BoolStr(config.capture.requestGpuDirect) << "\n";
    file << "  },\n";

    file << "  \"movement\": {\n";
    file << "    \"backend\": \"" << MovementBackendToIniString(config.movement.backend) << "\",\n";
    file << "    \"vertical_bias\": " << FormatDouble(config.movement.verticalBias, 3) << ",\n";
    file << "    \"deadzone\": " << FormatDouble(config.movement.deadzonePixels, 2) << ",\n";
    file << "    \"center_offset_x\": " << FormatDouble(config.movement.centerOffsetX, 2) << ",\n";
    file << "    \"center_offset_y\": " << FormatDouble(config.movement.centerOffsetY, 2) << ",\n";
    file << "    \"titan_two_hold_ms\": " << config.movement.titanTwoHoldMs << ",\n";
    file << "    \"stick_max_percent\": " << FormatDouble(config.movement.titanTwoStickMaxPercent, 2) << ",\n";
    file << "    \"stick_curve\": " << FormatDouble(config.movement.titanTwoStickCurve, 3) << ",\n";
    file << "    \"stick_response_boost\": " << FormatDouble(config.movement.titanTwoStickResponseBoost, 3) << ",\n";
    file << "    \"stick_min_percent\": " << FormatDouble(config.movement.titanTwoStickMinPercent, 2) << ",\n";
    file << "    \"aim_tracking\": " << BoolStr(config.movement.aimTrackingEnabled) << ",\n";
    file << "    \"aim_track_confirm_frames\": " << config.movement.aimTrackConfirmFrames << ",\n";
    file << "    \"aim_track_lost_frames\": " << config.movement.aimTrackLostFrames << ",\n";
    file << "    \"aim_track_match_max_cost\": " << FormatDouble(config.movement.aimTrackMatchMaxCost, 3) << ",\n";
    file << "    \"aim_lock_bonus\": " << FormatDouble(config.movement.aimTargetLockBonus, 3) << ",\n";
    file << "    \"aim_switch_margin\": " << FormatDouble(config.movement.aimTargetSwitchMargin, 3) << ",\n";
    file << "    \"aim_one_euro\": " << BoolStr(config.movement.aimOneEuroEnabled) << ",\n";
    file << "    \"aim_one_euro_min_cutoff\": " << FormatDouble(config.movement.aimOneEuroMinCutoff, 3) << ",\n";
    file << "    \"aim_one_euro_beta\": " << FormatDouble(config.movement.aimOneEuroBeta, 4) << ",\n";
    file << "    \"aim_one_euro_d_cutoff\": " << FormatDouble(config.movement.aimOneEuroDerivativeCutoff, 3) << ",\n";
    file << "    \"aim_prediction_ms\": " << FormatDouble(config.movement.aimPredictionMs, 1) << ",\n";
    file << "    \"aim_prediction_max_box\": " << FormatDouble(config.movement.aimPredictionMaxBoxFraction, 3) << ",\n";
    file << "    \"stick_pd\": " << BoolStr(config.movement.titanTwoStickPdEnabled) << ",\n";
    file << "    \"stick_d_gain\": " << FormatDouble(config.movement.titanTwoStickDerivativeGain, 4) << ",\n";
    file << "    \"stick_feed_forward\": " << FormatDouble(config.movement.titanTwoStickFeedForward, 4) << ",\n";
    file << "    \"stick_slew\": " << FormatDouble(config.movement.titanTwoStickSlewPercentPerSecond, 1) << ",\n";
    file << "    \"fov_radius\": " << FormatDouble(config.movement.fovRadius, 1) << "\n";
    file << "  }\n";
    file << "}\n";

    file.flush();
    if (!file.good()) {
        errorMessage = "Failed to write JSON file: " + jsonPath;
        return false;
    }

    return true;
}

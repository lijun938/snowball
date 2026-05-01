#pragma once

#include "MovementController.h"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

enum class PanelEngineAction {
    None,
    StartInference,
    StopInference,
    SwitchEngine,
    SaveConfig,
    ExportJson,
};

struct LiveControlPanelState {
    MovementSettings movement;
    MovementSettings defaults;
    float detectionConfidenceThreshold = 0.40f;
    float detectionNmsThreshold = 0.45f;
    float defaultDetectionConfidenceThreshold = 0.40f;
    float defaultDetectionNmsThreshold = 0.45f;
    int labelFilter = -1;
    bool capturePreviewEnabled = true;

    double fps = 0.0;
    int detectionCount = 0;
    std::string backendName;
    int currentTargetLabel = -1;
    bool hasTarget = false;
    bool outputActive = false;
    bool transportReady = false;
    std::uint64_t transportDispatchCount = 0;
    std::uint64_t transportSkipCount = 0;
    double transportLastSubmitMs = 0.0;
    int transportUserIndex = -1;
    int originX = 0;
    int originY = 0;
    int targetX = 0;
    int targetY = 0;
    int deltaX = 0;
    int deltaY = 0;
    double rawDeltaX = 0.0;
    double rawDeltaY = 0.0;
    double stickXPercent = 0.0;
    double stickYPercent = 0.0;
    float score = 0.0f;
    std::vector<int> observedLabels;

    bool inferenceRunning = true;
    std::string selectedEnginePath;
    std::vector<std::string> availableEngines;
    PanelEngineAction engineAction = PanelEngineAction::None;
    std::string configIniPath;
    std::string exportJsonPath;
};

class LiveControlPanel {
public:
    LiveControlPanel();
    ~LiveControlPanel();

    bool Initialize(const LiveControlPanelState& initialState);
    void Shutdown();
    bool PumpMessages();
    void SyncObservedState(const LiveControlPanelState& state);
    void UpdatePreview(const cv::Mat& frame);
    bool PullState(LiveControlPanelState& state);

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

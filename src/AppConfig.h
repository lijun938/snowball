#pragma once

#include "DetectorBackend.h"
#include "MovementController.h"
#include "QCAPCapture.h"

#include <string>

struct AppConfig {
    DetectorSettings detector;
    QCAPCaptureConfig capture;
    MovementSettings movement;
    bool headless = false;
    int maxFrames = 0;
    double maxSeconds = 0.0;
    double warmupSeconds = 0.0;
    std::string inferenceScreenshotPath;
    bool inferenceScreenshotDetectionsOnly = false;
    int inferenceScreenshotEveryNFrames = 0;
    std::string debugDumpQcapFramePath;
    int debugForceGpuDirectRebindFrame = 0;
    bool showHelp = false;
};

bool ParseAppConfig(int argc, char** argv, AppConfig& config, std::string& errorMessage);
bool LoadConfigFromIniFile(const std::string& iniPath, AppConfig& config, std::string& errorMessage);
bool SaveConfigToIniFile(const std::string& iniPath, const AppConfig& config, std::string& errorMessage);
bool ExportConfigToJson(const std::string& jsonPath, const AppConfig& config, std::string& errorMessage);
void PrintAppUsage();
const char* InferenceBackendKindToString(InferenceBackendKind backend);
const char* QcapInputToString(ULONG inputType);
const char* QcapColorSpaceToString(ULONG colorSpaceType);

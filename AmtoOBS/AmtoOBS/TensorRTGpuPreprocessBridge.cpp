#include "TensorRTGpuPreprocess.h"

#include "DetectorBackend.h"
#include "QCAPCapture.h"

#include <algorithm>

bool TensorRTGpuPreprocess::SupportsFrameView(const QCAPFrameView& frameView) {
    return frameView.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24 &&
        frameView.width > 0 &&
        frameView.height > 0 &&
        frameView.sourceDevicePointer != nullptr;
}

TensorRTGpuPreprocessRequest TensorRTGpuPreprocess::BuildRequest(
    const QCAPFrameView& frameView,
    const TensorRtRoiSettings* roiSettings) {
    TensorRTGpuPreprocessRequest request;
    request.sourceDevicePointer = frameView.sourceDevicePointer;
    request.sourceHostPointer = frameView.sourceBuffer;
    request.sourceWidth = frameView.width;
    request.sourceHeight = frameView.height;
    request.sourcePitchBytes = static_cast<std::size_t>(frameView.width) * 3U;
    request.sourceBufferSize = frameView.bufferSize;
    request.sourceImageUpsideDown = frameView.imageUpsideDown;
    if (roiSettings != nullptr && roiSettings->enabled && frameView.width > 0 && frameView.height > 0) {
        const int roiX = std::clamp(roiSettings->x, 0, frameView.width - 1);
        const int roiY = std::clamp(roiSettings->y, 0, frameView.height - 1);
        const int maxWidth = frameView.width - roiX;
        const int maxHeight = frameView.height - roiY;
        request.useRoi = true;
        request.roiX = roiX;
        request.roiY = roiY;
        request.roiWidth = std::clamp(roiSettings->width, 1, maxWidth);
        request.roiHeight = std::clamp(roiSettings->height, 1, maxHeight);
    }
    return request;
}

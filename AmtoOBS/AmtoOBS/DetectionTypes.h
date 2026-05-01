#pragma once

struct DetectionObject {
    struct {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
    } bbox;
    int label = -1;
    float prob = 0.0f;
};

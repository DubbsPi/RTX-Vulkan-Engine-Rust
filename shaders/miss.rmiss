#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"


layout(location = 0) rayPayloadInEXT RayPayload payload;


void main() {
    payload.modelIndex = -1;
}
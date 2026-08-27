#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"


layout(location = 0) rayPayloadInEXT vec3 hitColor;


void main() {
    float t = gl_WorldRayDirectionEXT.y * 0.5 + 0.5;
    hitColor = vec3(0, 0, t * 0.5);
}
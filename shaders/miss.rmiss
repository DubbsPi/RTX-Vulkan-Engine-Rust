#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"


layout(location = 0) rayPayloadInEXT vec3 hitColor;


layout(binding = 2, set = 0) uniform CameraUBO {
    mat4 viewInverse;
    mat4 projInverse;
    float time;
} cam;


void main() {
    hitColor = getSky(gl_WorldRayDirectionEXT, sunLightDir, cam.viewInverse[3].y);
}
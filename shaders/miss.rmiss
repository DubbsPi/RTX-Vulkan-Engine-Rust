#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "fog.glsl"


layout(location = 0) rayPayloadInEXT RayPayload payload;


layout(binding = 2, set = 0) uniform CameraUBO {
    mat4 viewInverse;
    mat4 projInverse;
    float time;
} cam;


void main() {
    vec3 camPos = cam.viewInverse[3].xyz;

    vec3 sky = getSky(gl_WorldRayDirectionEXT, sunLightDir, camPos.y);
    //mat2x3 fog = marchFog(camPos, gl_WorldRayDirectionEXT, gl_RayTmaxEXT);
    
    payload.radiance = sky;// * fog[1] + fog[0];
    payload.nextDirEnc = vec2(10);
}
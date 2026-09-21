#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"


layout(location = 0) rayPayloadInEXT RayPayload payload;

hitAttributeEXT vec2 attribs;


void main() {
    payload.modelIndex = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    payload.primitiveId = gl_PrimitiveID;
    payload.attribs = attribs;
    payload.tHit = gl_HitTEXT;
    payload.transform = gl_WorldToObjectEXT;
}
#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"


layout(location = 0) rayPayloadInEXT vec3 hitColor;
layout(location = 1) rayPayloadEXT bool shadowed;

hitAttributeEXT vec2 attribs;


layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3, set = 0, scalar) buffer ObjectDescs {ObjectDesc descs[];};
layout(binding = 4, set = 0, scalar) buffer Materials {Material mats[];};
layout(binding = 5, set = 0, scalar) buffer MaterialIDs {uint materialIds[];};

const vec3 LIGHT_DIR = normalize(vec3(0.4, 1.0, -0.2));


void main() {
    ObjectDesc obj = descs[gl_InstanceCustomIndexEXT];
    Indices indices = Indices(obj.indexAddress);
    Vertices vertices = Vertices(obj.vertexAddress);

    uint i0 = indices.i[3 * gl_PrimitiveID + 0];
    uint i1 = indices.i[3 * gl_PrimitiveID + 1];
    uint i2 = indices.i[3 * gl_PrimitiveID + 2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 p0 = vertices.v[i0];
    vec3 p1 = vertices.v[i1];
    vec3 p2 = vertices.v[i2];

    vec3 hitPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    hitPos = (gl_ObjectToWorldEXT * vec4(hitPos, 1.0)).xyz;

    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 rawNormal = cross(edge1, edge2);

    float len = length(rawNormal);
    vec3 normal = (len > 1e-8) ? (rawNormal / len) : vec3(0, 1, 0);

    normal = normalize((gl_ObjectToWorldEXT * vec4(normal, 0)).xyz);
    
    uint materialId = materialIds[gl_PrimitiveID];
    Material mat = mats[materialId];
    
    shadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0,
        0,
        1,
        hitPos + normal * 0.001,
        0.001,
        LIGHT_DIR,
        10000.0,
        1
    );

    float NdotL = max(dot(normal, LIGHT_DIR), 0.0);
    float shadowFactor = shadowed ? 0.25 : 1.0;
    hitColor = mat.albedo * NdotL * shadowFactor;
}
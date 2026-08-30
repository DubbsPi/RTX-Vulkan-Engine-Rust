#version 460
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"


layout(location = 0) rayPayloadInEXT vec3 hitColor;
layout(location = 1) rayPayloadEXT bool shadowed;

hitAttributeEXT vec2 attribs;

layout(binding = 2, set = 0) uniform CameraUBO {
    mat4 viewInverse;
    mat4 projInverse;
    float time;
} cam;


layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3, set = 0, scalar) buffer ObjectDescs {ObjectDesc descs[];};
layout(binding = 4, set = 0, scalar) buffer Materials {Material mats[];};
layout(binding = 5, set = 0, scalar) buffer MaterialIDs {uint materialIds[];};

layout(binding = 6, set = 0) uniform sampler2D textures[];


void main() {
    ObjectDesc obj = descs[gl_InstanceCustomIndexEXT];
    Indices indices = Indices(obj.indexAddress);
    Vertices vertices = Vertices(obj.vertexAddress);

    uint i0 = indices.i[3 * gl_PrimitiveID];
    uint i1 = indices.i[3 * gl_PrimitiveID + 1];
    uint i2 = indices.i[3 * gl_PrimitiveID + 2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 p0 = vertices.vert[i0].v;
    vec3 p1 = vertices.vert[i1].v;
    vec3 p2 = vertices.vert[i2].v;

    vec3 hitPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    hitPos = (gl_ObjectToWorldEXT * vec4(hitPos, 1.0)).xyz;

    vec3 n0 = vertices.vert[i0].n;
    vec3 n1 = vertices.vert[i1].n;
    vec3 n2 = vertices.vert[i2].n;
    vec3 normal = n0 * bary.x + n1 * bary.y + n2 * bary.z;
    
    vec2 uv0 = vertices.vert[i0].uv;
    vec2 uv1 = vertices.vert[i1].uv;
    vec2 uv2 = vertices.vert[i2].uv;
    vec2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;


    // Special normal calculation for smooth shading
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 rawNormal = cross(edge1, edge2);

    vec3 geometric_normal = normalize((gl_ObjectToWorldEXT * vec4(rawNormal, 0)).xyz);

    // Optional special shadow position calculations. Produces some artifacts
    #ifdef ALTERNATE_SHADOW_ORIGIN
    vec3 t0 = hitPos - p0;
    vec3 t1 = hitPos - p1;
    vec3 t2 = hitPos - p2;

    float d0 = dot(t0, n0);
    float d1 = dot(t1, n1);
    float d2 = dot(t2, n2);

    vec3 o0 = d0 * n0;
    vec3 o1 = d1 * n1;
    vec3 o2 = d2 * n2;

    vec3 iO = bary.x * o0 + bary.y * o1 + bary.z * o2;
    vec3 adjustedPos = hitPos - iO;
    #endif

    // Texturing
    uint materialId = materialIds[gl_PrimitiveID];
    Material mat = mats[materialId];
    vec3 albedo = mat.albedo;

    if (mat.albedo_texture_index >= 0)
        albedo = texture(textures[nonuniformEXT(mat.albedo_texture_index)], uv).rgb;
    

    shadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0,
        0,
        1,
        #ifdef ALTERNATE_SHADOW_ORIGIN
        adjustedPos + geometric_normal * 0.001,
        #else
        hitPos + geometric_normal * 0.001,
        #endif
        0.001,
        sunLightDir,
        10000.0,
        1
    );

    const float ambient = 0.025;
    float lighting = max(dot(normal, sunLightDir), 0.0);
    float shadowFactor = shadowed ? 0.0 : 1.0;
    hitColor = albedo * max(lighting * shadowFactor, ambient);
}
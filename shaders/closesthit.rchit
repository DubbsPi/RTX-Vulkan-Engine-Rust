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


float G1(in float NdotX, in float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}


void main() {
    ObjectDesc obj = descs[gl_InstanceCustomIndexEXT];
    Indices indices = Indices(obj.indexAddress);
    Vertices vertices = Vertices(obj.vertexAddress);

    uint i0 = indices.i[3 * gl_PrimitiveID];
    uint i1 = indices.i[3 * gl_PrimitiveID + 1];
    uint i2 = indices.i[3 * gl_PrimitiveID + 2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 p0 = vertices.vert[i0].p;
    vec3 p1 = vertices.vert[i1].p;
    vec3 p2 = vertices.vert[i2].p;

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

    vec3 geometricNormal = normalize((gl_ObjectToWorldEXT * vec4(rawNormal, 0)).xyz);


    // Texturing
    uint materialId = materialIds[obj.materialId + gl_PrimitiveID];
    Material mat = mats[materialId];
    vec3 albedo = mat.albedo;

    if (mat.albedoTextureIndex >= 0) {
        vec4 textureSample = texture(textures[nonuniformEXT(mat.albedoTextureIndex)], uv);
        
        if (textureSample.a < 0.001) return;  // Placeholder for now
        
        albedo = textureSample.rgb;
    }

    
    vec3 V = -gl_WorldRayDirectionEXT;
    vec3 H = normalize(V + sunLightDir);
    float NdotV = max(dot(normal, V), 0.0001);
    float NdotL = max(dot(normal, sunLightDir), 0.0001);
    float NdotH = max(dot(normal, H), 0.0);
    float VdotH = max(dot(normal, H), 0.0);


    float shadowFactor = 1.0;
    if (NdotL > 0.0) {
        // Trace shadow ray
        shadowed = true;
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
            0xFF, 0, 0, 1,
            hitPos + geometricNormal * 0.001,
            0.001, sunLightDir, 10000.0, 1
        );

        shadowFactor = shadowed? 0.05 : 1.0;
    }


    const float invPi = 1.0 / PI;
    const vec3 lightColor = vec3(1);

    vec3 F0 = mix(vec3(0.08 * mat.specular), albedo, mat.metallic);

    vec3 F;
    vec3 specular = cookTorrance(mat.roughness, F0, NdotV, NdotL, NdotH, VdotH, F);
    vec3 clearcoatLobe = evalClearcoat(mat.clearcoat, mat.clearcoatRoughness, NdotV, NdotL, NdotH, VdotH);
    vec3 sheenLobe = evalSheen(mat.sheen, mat.sheenColor, VdotH);

    vec3 kd = (1.0 - F) * (1.0 - mat.metallic);
    vec3 diffuseColor = albedo * (1.0 - mat.metallic) * (1.0 - mat.transmission);
    vec3 diffuse = kd * diffuseColor * invPi;


    vec3 direct = (diffuse + specular + clearcoatLobe + sheenLobe) * NdotL * lightColor * shadowFactor;
    
    vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, normal);
    vec3 reflectionColor = any(greaterThan(F * mat.metallic, vec3(0.05)))? getSky(reflectDir, sunLightDir, cam.viewInverse[3].y) : vec3(0);

    hitColor = direct + mat.emission + reflectionColor * F * mat.metallic;
}
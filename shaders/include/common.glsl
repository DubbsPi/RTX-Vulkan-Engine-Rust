#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require


#define PI 3.14159265359
#define TAU 6.28318530718


const float denoiseStrength = 1.0;
#define DEPTH_WEIGHT_SCALE 3.0 
#define DEPTH_SENSITIVITY 0.5


const vec3 sunLightDir = normalize(vec3(-0.4, 1, 0.2));
const vec3 sunLightColor = vec3(1);
const float sunAngularRadius = 0.01;

#define SKY_VIEW_SAMPLES 16
#define SKY_LIGHT_SAMPLES 8

const float sunIntensity  = 15.0;
const float earthRadius  = 6371000.0;
const float atmosphereRadius = earthRadius + 100000.0;
const vec3  betaRayleigh = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const float betaMie = 21e-6;
const float mieG = 0.76;
const float hr = 8500.0;
const float hm = 1200.0;


const float maxFogDist = 100.0;


const int MAX_BOUNCES = 4;
const int MAX_ACCUMULATION = 2048;


struct Vertex {
    vec3 p;
    float pad0;
    vec3 n;
    float pad1;
    vec2 uv;
    uint16_t ji[4];
    float jw[4];
};


struct RayPayload {
    vec2 attribs;
    uint primitiveId;
    int modelIndex;
    float tHit;
    mat4x3 transform;
};


layout(buffer_reference, scalar) buffer Vertices {
    Vertex vert[];
};
layout(buffer_reference, scalar) buffer Indices {
    uint i[];
};


struct ObjectDesc {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint materialId;
    uint pad0;
};

struct Material {
    vec3 albedo;
    int albedoTextureIndex;
    float metallic;
    float roughness;

    vec3 emission;

    float transmission;
    float ior;

    float specular;
    float clearcoat;
    float clearcoatRoughness;

    float sheen;
    vec3 sheenColor;

    uint alphaMode;
    float alphaCutoff;
};


// Atmospheric scattering
vec2 intersectSphereSky(in vec3 rayOrigin, in vec3 rayDir, in float radius) {
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float d = b*b - c;
    if (d < 0.0) return vec2(-1.0);
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}

float phaseRayleigh(in float mu) {
    return 0.05968310365 * (1.0 + mu * mu);
}

float phaseMie(in float mu) {
    float g  = mieG, g2 = g * g;
    return 0.11936620731 * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(abs(1.0 + g2 - 2.0 * g * mu), 1.5));
}

float opticalDepth(in vec3 pos, in vec3 dir, in float rayLength, in float scaleHeight, in int steps) {
    float stepSize = rayLength / float(steps);
    float depth = 0.0;
    for (int i = 0; i < steps; i++) {
        vec3 p = pos + dir * (float(i) + 0.5) * stepSize;
        float h = length(p) - earthRadius;
        depth += exp(-h / scaleHeight) * stepSize;
    }
    return depth;
}

vec3 scatterAtmosphere(in vec3 viewDir, in vec3 sunDir, in float cameraY) {
    vec3 origin = vec3(0, earthRadius + max(cameraY, 5.0), 0);

    // Get the atmospheric exit
    vec2 atmoHit = intersectSphereSky(origin, viewDir, atmosphereRadius);
    float tMin = max(atmoHit.x, 0.0);
    float tMax = atmoHit.y;

    vec2 groundHit = intersectSphereSky(origin, viewDir, earthRadius);
    if (groundHit.x > 0.0)
        tMax = min(tMax, groundHit.x);

    float stepSize = (tMax - tMin) / float(SKY_VIEW_SAMPLES);

    vec3 sunRayleighAccum  = vec3(0);
	vec3 sunMieAccum = vec3(0);
    float odR = 0.0;
	float odM = 0.0;

    float mu = dot(viewDir, sunDir);

    for (int i = 0; i < SKY_VIEW_SAMPLES; i++) {
        vec3 p = origin + viewDir * (tMin + (float(i) + 0.5) * stepSize);
        float h = length(p) - earthRadius;

        // Local density
        float densR = exp(-h / hr) * stepSize;
        float densM = exp(-h / hm) * stepSize;
        odR += densR;
        odM += densM;

		// Amount of sunlight
        vec2 sunHit = intersectSphereSky(p, sunDir, atmosphereRadius);
        float sunRayLen = sunHit.y;

        float sunOdR = opticalDepth(p, sunDir, sunRayLen, hr, SKY_LIGHT_SAMPLES);
        float sunOdM = opticalDepth(p, sunDir, sunRayLen, hm, SKY_LIGHT_SAMPLES);

        // Total transmittance
        vec3 sunTransmittance = exp(-(betaRayleigh * (odR + sunOdR)) - (betaMie * (odM + sunOdM) * 1.1));

        sunRayleighAccum += densR * sunTransmittance;
        sunMieAccum += densM * sunTransmittance;
    }

    vec3 skyColor = sunIntensity * (phaseRayleigh(mu) * betaRayleigh * sunRayleighAccum + phaseMie(mu) * betaMie * sunMieAccum);
    return skyColor;
}

vec3 getSky(in vec3 rayDir, in vec3 sunDir, in float cameraY) {
    rayDir.y = max(rayDir.y, -0.25);
    vec3 skyColor = scatterAtmosphere(rayDir, sunDir, cameraY);

    const float cosAngularRadius = cos(sunAngularRadius);
    float sunDisc = smoothstep(cosAngularRadius - 0.0001, cosAngularRadius, dot(rayDir, sunDir));
    skyColor += sunIntensity * sunDisc * smoothstep(0.0, 1.0, sunDir.y);

    return skyColor;
}


float bayerDither(in vec2 pixelPos) {
    int x = int(mod(pixelPos.x, 4.0));
    int y = int(mod(pixelPos.y, 4.0));
    
    int index = x + y * 4;
    
    float pattern[16] = float[16](
        0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0,
        12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0,
        3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0,
        15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0
    );
    
    return pattern[index] - 0.5;
}


vec3 cookTorrance(in float roughness, in vec3 F0, in float NdotV, in float NdotL, in float NdotH, in float VdotH, out vec3 F) {
    // Fresnel
    F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    // Distribution
    float a = max(roughness, 0.045);
    a = a * a;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * denom * denom);

    // Geometry
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));

    return (D * G * F) / (4.0 * NdotV * NdotL);
}

vec3 evalClearcoat(in float clearcoat, in float ccRoughness, in float NdotV, in float NdotL, in float NdotH, in float VdotH, out float Fc) {
    float a = ccRoughness * ccRoughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * denom * denom);

    Fc = 0.04 + 0.96 * pow(1.0 - VdotH, 5.0);
    
    float k = 0.25;
    float G = (NdotV / (NdotV * (1.0-k)+k)) * (NdotL / (NdotL*(1.0-k)+k));

    float clearcoatSpec = D * Fc * G / max(4.0 * NdotV * NdotL, 1e-4);
    return vec3(clearcoatSpec * clearcoat);
}

vec3 evalSheen(in float sheen, in vec3 sheenColor, in float roughness, in float NdotH, in float NdotV, in float NdotL) {
    float invAlpha = 1.0 / max(roughness, 0.007);
    float cos2h = NdotH * NdotH;
    float sin2h = max(1.0 - cos2h, 0.0078125);
    float D = (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);

    float V = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV));

    return sheenColor * sheen * D * V;
}

vec3 energyCompensation(in vec3 F0, in float roughness, in float NdotV) {
    float Ess = 1.0 - pow(1.0 - NdotV, 5.0 - 4.0 * roughness);
    return 1.0 + F0 * (1.0 / max(Ess, 0.01) - 1.0);
}


uint hash(in uint x) {
    x ^= x >> 17; x *= 0xbf324c81u;
    x ^= x >> 11; x *= 0x68bc2a9du;
    x ^= x >> 16;
    return x;
}

uint pcg(inout uint state) {
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

float rand(inout uint seed) {
    seed = pcg(seed);
    return float(seed) / float(0xFFFFFFFFu);
}

uint randInt(in uvec2 pixel, in uvec2 resolution, in uint frameIndex) {
    uint seed = pixel.x + pixel.y * resolution.x;
    seed = hash(seed ^ hash(frameIndex));
    return seed;
}

vec3 cosineHemisphere(in vec3 normal, inout uint seed) {
    float r1 = rand(seed);
    float r2 = rand(seed);
    
    float phi = 2.0 * PI * r1;
    float sqrtR2 = sqrt(r2);
    
    vec3 up = abs(normal.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 tangent   = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    
    return normalize(
        tangent   * cos(phi) * sqrtR2 +
        bitangent * sin(phi) * sqrtR2 +
        normal    * sqrt(1.0 - r2)
    );
}

float hash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}


vec3 sampleGGX(inout uint seed, in vec3 normal, in float roughness) {
    float r1 = rand(seed);
    float r2 = rand(seed);

    float a = max(roughness, 0.045);
    a = a * a;

    // GGX importance sampling
    float phi = 2.0 * PI * r1;
    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    vec3 H_tangent = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 up = abs(normal.y) < 0.999 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * H_tangent.x + bitangent * H_tangent.y + normal * H_tangent.z);
}

vec3 fresnelSchlick(in float cosTheta, in vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}


vec3 sampleSunCone(inout uint rngState, in vec3 sunDir, in float sunAngularRadius) {
    float cosThetaMax = cos(sunAngularRadius);
    float xi1 = rand(rngState);
    float xi2 = rand(rngState);

    float cosTheta = 1.0 - xi1 * (1.0 - cosThetaMax);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * PI * xi2;

    float signy = sunDir.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (signy + sunDir.z);
    float b = sunDir.x * sunDir.y * a;
    vec3 tangent = vec3(1.0 + signy * sunDir.x * sunDir.x * a, signy * b, -signy * sunDir.x);
    vec3 bitangent = vec3(b, signy + sunDir.y * sunDir.y * a, -sunDir.y);

    vec3 localDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    return normalize(tangent * localDir.x + bitangent * localDir.y + sunDir * localDir.z);
}


float luminance(in vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float G1(in float NdotX, in float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}


bool finiteFloat(in float v) {
    return !isnan(v) && !isinf(v);
}

bool finiteVec3(in vec3 v) {
    return all(not(isnan(v))) && all(not(isinf(v)));
}

vec3 sanitizeColor(in vec3 c) {
    if (!finiteVec3(c))
        return vec3(0.0);
    return max(c, vec3(0.0));
}


#endif
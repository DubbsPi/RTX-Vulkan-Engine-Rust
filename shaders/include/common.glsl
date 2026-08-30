#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require


const vec3 sunLightDir = normalize(vec3(0.5, 1, -0.2));

#define SKY_VIEW_SAMPLES 12
#define SKY_LIGHT_SAMPLES 6

const float sunIntensity  = 15.0;
const float earthRadius  = 6371000.0;
const float atmosphereRadius = earthRadius + 100000.0;
const vec3  betaRayleigh = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const float betaMie = 21e-6;
const float mieG = 0.76;
const float hr = 8500.0;
const float hm = 1200.0;


struct Vertex {
    vec3 v;
    vec3 n;
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
    uint pad;
};

struct Material {
    vec3 albedo;
    float pad0;
    float metallic;
    float roughness;
    vec2 pad1;
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
    
    float sunDisc = smoothstep(0.9999, 1.0, dot(rayDir, sunDir));
    skyColor += sunIntensity * sunDisc * smoothstep(0.0, 1.0, sunDir.y);

    return skyColor;
}

#endif
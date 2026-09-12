#ifndef FOG_GLSL
#define FOG_GLSL

#extension GL_GOOGLE_include_directive : require
#include "common.glsl"


layout(binding = 2, set = 0) uniform TimeUBO {
    mat4 viewInverse;
    mat4 projInverse;
    float time;
} time;


const float invSamples = 1.0 / float(FOG_SAMPLES);


vec2 randomGradient(vec2 p) {
    float h = hash3(vec3(p, 0)) * 6.2831853;
    return vec2(cos(h), sin(h));
}

float gradientNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = dot(randomGradient(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(randomGradient(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(randomGradient(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(randomGradient(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.5 + 0.5;
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < 4; i++) {
        value += amplitude * gradientNoise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}


vec3 fogAt(vec3 p) {
    float density = -abs(p.y + 5.0) * 0.1 - fbm(p.xz * 0.1);

    return vec3(max(density * 2.5, 0.0));
}


float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}


const vec3 sunColor = vec3(1);

vec3 lightMarch(in vec3 p) {
    vec3 transmittance = vec3(1);
    float stepSize = 1.0;

    for (int i = 0; i < LIGHT_SAMPLES; i++) {
        vec3 samplePos = p + sunLightDir * stepSize * (float(i) + 0.5);
        vec3 density = fogAt(samplePos);
        transmittance *= exp(-density * stepSize);
    }

    return transmittance;
}

mat2x3 marchFog(in vec3 rayOrigin, in vec3 rayDir, in float tMax) {
    vec3 transmittance = vec3(1);
    vec3 accumulatedFog = vec3(0);

    float cosTheta = dot(rayDir, sunLightDir);
    float phase = henyeyGreenstein(cosTheta, 0.3);

    tMax = min(maxFogDist, tMax);
    float prevT = 0.0;

    for (int i = 0; i < FOG_SAMPLES; i++) {
        float linearT = (float(i) + 0.5) * invSamples;

        #ifdef QUADRATIC_FOG
        float t = linearT * linearT;
        #else
        float t = linearT;
        #endif

        float currentDist = t * tMax;

        float stepDist = currentDist - prevT;
        prevT = currentDist;

        vec3 p = rayOrigin + rayDir * currentDist;
        vec3 density = fogAt(p);

        vec3 lightVisibility = lightMarch(p);
        vec3 litColor = sunColor * lightVisibility * phase + vec3(0);

        vec3 stepTransmittance = exp(-density * stepDist);
        vec3 stepScatter = density * stepDist * transmittance * litColor;

        accumulatedFog += stepScatter;
        transmittance *= stepTransmittance;

        if (max(transmittance.r, max(transmittance.g, transmittance.b)) < 0.001) {
            break;
        }
    }

    return mat2x3(accumulatedFog, transmittance);
}


#endif
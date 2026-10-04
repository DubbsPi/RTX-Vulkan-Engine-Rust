#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_shader_invocation_reorder : require


#define PI 3.14159265359
#define TAU 6.28318530718
#define INV_PI 0.3183098861837907
#define INV_4PI 0.07957747154594767


const float denoiseStrength = 1.0;
#define DEPTH_WEIGHT_SCALE 3.0 
#define DEPTH_SENSITIVITY 0.5


layout(constant_id = 0) const uint MAX_ACCUMULATION = 2048;
layout(constant_id = 1) const uint MAX_BOUNCES = 6;
layout(constant_id = 2) const uint MAX_NEE_TESTS = 2;

layout(constant_id = 3) const uint SKY_VIEW_SAMPLES = 16;
layout(constant_id = 4) const uint SKY_LIGHT_SAMPLES = 8;

const uint MAX_GLASS_HOPS = 4;
const float GLASS_NEE_BLEND = 0.5;

const float MAX_STAR_CAMERA_BRIGHTNESS = 1e2;


struct Vertex {
    vec3 p;
    vec3 n;
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
    vec3 absorptionColor;
    float dispersion;

    float specular;
    float clearcoat;
    float clearcoatRoughness;

    float sheen;
    vec3 sheenColor;

    uint alphaMode;
    float alphaCutoff;
};


struct PlanetRenderInfo {
    vec3 planetUp;
    float cameraDist;
    float radius;
    float atmosphereRadius;

    vec3 planetColor;
    int parentStar;

    vec3 betaRayleigh;
    float betaMie;
    float mieG;
    float hr;
    float hm;
};

struct StarRenderInfo {
    vec3 starUp;
    float cameraDist;
    float radius;

    vec3 color;
    float luminosity;
};


// Util functions
bool finiteFloat(in float v) {
    return !isnan(v) && !isinf(v);
}

bool finiteVec3(in vec3 v) {
    return all(not(isnan(v))) && all(not(isinf(v)));
}

vec3 sanitizeColor(in vec3 c) {
    if (!finiteVec3(c))
        return vec3(0);
    return max(c, vec3(0));
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

float hash3(in vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

void buildTangentBasis(in vec3 N, out vec3 T, out vec3 B) {
    vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

vec3 sampleCosineHemisphere(in vec2 xi, in vec3 N) {
    float r = sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;
    vec3 local = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - xi.x)));
    vec3 T, B;
    buildTangentBasis(N, T, B);
    return normalize(local.x * T + local.y * B + local.z * N);
}


float bayerDither(in vec2 pixelPos) {
    int x = int(mod(pixelPos.x, 4.0));
    int y = int(mod(pixelPos.y, 4.0));
    
    int index = x + y * 4;
    
    const float pattern[16] = float[16](
        0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0,
        12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0,
        3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0,
        15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0
    );
    
    return pattern[index] - 0.5;
}

float luminance(in vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}


vec2 intersectSphere(in vec3 ro, in vec3 rd, in vec3 pos, in float r) {
    vec3 p  = ro - pos;
    float b = dot(p, rd);
    float c = dot(p, p) - r * r;

    float discriminant = b * b - c;
    if (discriminant < 0.0) return vec2(-1);
    
    float rtD = sqrt(discriminant);
    
    float tNear = -b - rtD;
    float tFar = -b + rtD;

    if (tFar < 0.0001) return vec2(-1);
    return vec2(tNear, tFar);
}


float balanceHeuristic(in float a, in float b) {
    return a / max(a + b, 1e-30);
}

float diskSolidAngle(in float r, in float d) {
    float s = min(r / d, 0.9999);
    return 2.0 * PI * s * s / (1.0 + sqrt(1.0 - s * s));
}

float D_GGX(in float NdotH, in float a) {
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}


// Atmospheric scattering
struct SphereRayParams {
    float b;
    float c;
};

SphereRayParams sphereRayParams(vec3 dir, float distFromCenter, vec3 up0, float testRadius) {
    SphereRayParams p;
    p.b = distFromCenter * dot(dir, up0);
    float hRel = distFromCenter - testRadius;
    p.c = hRel * (distFromCenter + testRadius);
    return p;
}

vec2 solveSphereHits(in SphereRayParams p) {
    float disc = p.b * p.b - p.c;
    if (disc < 0.0) return vec2(-1.0);
    
    float sqrtD = sqrt(disc);
    float q = (p.b >= 0.0) ? -(p.b + sqrtD) : -(p.b - sqrtD);
    if (abs(q) < 1e-8) return vec2(-p.b, -p.b);

    float t0 = q;
    float t1 = p.c / q;

    return vec2(min(t0, t1), max(t0, t1));
}

float heightAtT(in SphereRayParams p, in float t, in float R) {
    float f = p.c + 2.0 * p.b * t + t * t;
    return f / (R + sqrt(max(R * R + f, 0.0)));
}

vec2 intersectSphereSky(in vec3 rayOrigin, in vec3 rayDir, in float radius) {
    float b = dot(rayOrigin, rayDir);
    float ocLen = length(rayOrigin);
    float c = (ocLen - radius) * (ocLen + radius);
    float d = b * b - c;
    if (d < 0.0) return vec2(-1.0);
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}

float phaseRayleigh(in float mu) {
    return 0.05968310365 * (1.0 + mu * mu);
}

float phaseMie(in float mu, in float mieG) {
    float g2 = mieG * mieG;
    return 0.11936620731 * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(abs(1.0 + g2 - 2.0 * mieG * mu), 1.5));
}

float opticalDepth(in SphereRayParams p, in float t0, in float rayLength, in float scaleHeight, in float R, in uint steps) {
    float stepSize = rayLength / float(steps);
    float depth = 0.0;
    for (int i = 0; i < steps; i++) {
        float t = t0 + (float(i) + 0.5) * stepSize;
        float h = heightAtT(p, t, R);
        depth += exp(-h / scaleHeight) * stepSize;
    }
    return depth;
}

vec3 atmosphereTransmittance(in vec3 rayOrigin, in vec3 rayDir, in PlanetRenderInfo pinfo) {
    if (pinfo.atmosphereRadius <= 0.0) return vec3(1.0);

    float distFromCenter = pinfo.radius + pinfo.cameraDist;
    vec3 planetCenterToOrigin = pinfo.planetUp * distFromCenter + rayOrigin;
    float originDist = length(planetCenterToOrigin);
    vec3 originUp = planetCenterToOrigin / max(originDist, 1e-6);

    SphereRayParams atmoParams = sphereRayParams(rayDir, originDist, originUp, pinfo.radius + pinfo.atmosphereRadius);
    vec2 atmoHits = solveSphereHits(atmoParams);
    if (atmoHits.y <= 0.0) return vec3(1.0);

    float tStart = max(atmoHits.x, 0.0);
    float tEnd = atmoHits.y;
    if (tEnd <= tStart) return vec3(1.0);

    SphereRayParams groundParams = sphereRayParams(rayDir, originDist, originUp, pinfo.radius);
    vec2 groundHits = solveSphereHits(groundParams);
    if (groundHits.x > tStart && groundHits.x < tEnd) return vec3(0.0);

    float pathLength = tEnd - tStart;
    uint sampleCount = max(SKY_LIGHT_SAMPLES, 32u);
    float odR = opticalDepth(groundParams, tStart, pathLength, pinfo.hr, pinfo.radius, sampleCount);
    float odM = opticalDepth(groundParams, tStart, pathLength, pinfo.hm, pinfo.radius, sampleCount);
    return exp(-(pinfo.betaRayleigh * odR) - (pinfo.betaMie * odM * 1.1));
}

vec3 scatterAtmosphere(
    in vec3 viewDir, in vec3 sunDir,
    in vec3 sunColor, in float sunIntensity,
    in SphereRayParams groundParams,
    in float distFromCenter,
    in PlanetRenderInfo pinfo,
    in float jitter,
    in float segStart, in float segEnd,
    out vec3 transmittance
 ) {  // This is only formatted like this to clean up the drowdown in VS
    if (segEnd <= segStart) {
        transmittance = vec3(1);
        return vec3(0);
    }

    float stepSize = (segEnd - segStart) / float(SKY_VIEW_SAMPLES);
    float tStart = segStart + jitter * stepSize;

    vec3 sunRayleighAccum = vec3(0);
    vec3 sunMieAccum = vec3(0);
    float odR = 0.0;
    float odM = 0.0;
    float mu = dot(viewDir, sunDir);

    for (int i = 0; i < SKY_VIEW_SAMPLES; i++) {
        float t = tStart + (float(i) + 0.5) * stepSize;
        float h = heightAtT(groundParams, t, pinfo.radius);

        float densR = exp(-h / pinfo.hr) * stepSize;
        float densM = exp(-h / pinfo.hm) * stepSize;
        odR += densR;
        odM += densM;

        float distFromCenterAtSample = pinfo.radius + h;
        vec3 localPos = pinfo.planetUp * distFromCenter + viewDir * t;
        vec3 localUp = localPos / distFromCenterAtSample;

        SphereRayParams sunGroundParams = sphereRayParams(sunDir, distFromCenterAtSample, localUp, pinfo.radius);
        SphereRayParams sunAtmoParams = sphereRayParams(sunDir, distFromCenterAtSample, localUp, pinfo.radius + pinfo.atmosphereRadius);

        vec2 planetShadow = solveSphereHits(sunGroundParams);
        bool inShadow = planetShadow.x > 0.01;

        if (!inShadow) {
            vec2 sunHit = solveSphereHits(sunAtmoParams);
            float sunRayLen = max(sunHit.y, 0.0);

            float sunOdR = opticalDepth(sunGroundParams, 0.0, sunRayLen, pinfo.hr, pinfo.radius, SKY_LIGHT_SAMPLES);
            float sunOdM = opticalDepth(sunGroundParams, 0.0, sunRayLen, pinfo.hm, pinfo.radius, SKY_LIGHT_SAMPLES);

            vec3 sunTransmittance = exp(-(pinfo.betaRayleigh * (odR + sunOdR)) - (pinfo.betaMie * (odM + sunOdM) * 1.1));

            sunRayleighAccum += densR * sunTransmittance;
            sunMieAccum += densM * sunTransmittance;
        }
    }

    vec3 color = sunIntensity * (phaseRayleigh(mu) * pinfo.betaRayleigh * sunRayleighAccum + phaseMie(mu, pinfo.mieG) * pinfo.betaMie * sunMieAccum) * sunColor;
    color = sanitizeColor(color);

    transmittance = exp(-(pinfo.betaRayleigh * odR) - (pinfo.betaMie * odM * 1.1));
    return color;
}


// Sky/miss stuff
vec3 starField(in vec3 rayDir) {
    vec3 cell = floor(rayDir * 750.0);
    float h = hash3(cell);

    float threshold = 0.998;
    if (h < threshold) return vec3(0.0);

    float brightness = (h - threshold) / (1.0 - threshold);
    brightness = pow(brightness, 4.0);

    float colorSeed = hash3(cell + vec3(17.0, 43.0, 91.0));
    vec3 starColor = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.9, 0.75), colorSeed) * 0.1;

    return starColor * brightness;
}

vec3 getPlanetSky(in vec3 rayDir, in vec3 sunDir, in vec3 sunColor, in float sunIntensity, in PlanetRenderInfo pinfo, in float jitter, in float maxVisibleDistance, in bool includeGround) {
    float distFromCenter = pinfo.radius + pinfo.cameraDist;

    SphereRayParams groundParams = sphereRayParams(rayDir, distFromCenter, pinfo.planetUp, pinfo.radius);
    SphereRayParams atmoParams = sphereRayParams(rayDir, distFromCenter, pinfo.planetUp, pinfo.radius + pinfo.atmosphereRadius);

    vec2 atmoHit = solveSphereHits(atmoParams);
    vec2 groundHit = solveSphereHits(groundParams);

    float segStart = max(atmoHit.x, 0.0);
    float segEnd = min(atmoHit.y, maxVisibleDistance);
    bool hitGround = includeGround && groundHit.x > 0.0;
    if (hitGround) segEnd = min(segEnd, groundHit.x);

    vec3 skyColor = vec3(0);
    vec3 transmittance = vec3(1.0);
    if (segEnd > segStart) {
        skyColor = scatterAtmosphere(rayDir, sunDir, sunColor, sunIntensity, groundParams, distFromCenter, pinfo, jitter, segStart, segEnd, transmittance);
    }
    
    if (hitGround) {
        vec3 hitPos = rayDir * groundHit.x;
        vec3 planetCenter = -pinfo.planetUp * distFromCenter;
        vec3 surfaceNormal = normalize(hitPos - planetCenter); 
        float NdotL = max(dot(surfaceNormal, sunDir), 0.0);

        vec3 groundLight = pinfo.planetColor * sunColor * sunIntensity * NdotL * (1.0 / PI) * 0.1;
        skyColor += groundLight * transmittance;
    }

    skyColor = sanitizeColor(skyColor);
    return skyColor;
}

vec3 starPositionCamRelative(in StarRenderInfo star) {
    return -star.starUp * (star.cameraDist + star.radius);
}

void starLightAtPoint(in StarRenderInfo star, in vec3 pos, out vec3 lightDir, out vec3 irradiance) {
    vec3 toStar = starPositionCamRelative(star) - pos;
    float scale = max(max(abs(toStar.x), max(abs(toStar.y), abs(toStar.z))), 1e-6);
    
    vec3 scaled = toStar / scale;

    float scaledDist = length(scaled);
    float dist = scale * scaledDist;
    float invDist = 1.0 / dist;

    lightDir = toStar * invDist;

    float brightness = star.luminosity * invDist;
    brightness *= invDist;
    irradiance = star.color * brightness * INV_4PI;
}

vec3 sampleStarDisk(in vec3 starDir, float starRadius, float dist, inout uint rngState) {
    float sinThetaMax = clamp(starRadius / dist, 0.0, 1.0);
    float cosThetaMax = sqrt(max(1.0 - sinThetaMax * sinThetaMax, 0.0));

    vec2 xi = vec2(rand(rngState), rand(rngState));
    float cosTheta = mix(1.0, cosThetaMax, xi.x);
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    float phi = 2.0 * PI * xi.y;

    vec3 T, B;
    buildTangentBasis(starDir, T, B);
    return normalize(cosTheta * starDir + sinTheta * cos(phi) * T + sinTheta * sin(phi) * B);
}

float starSelWeight(in StarRenderInfo s, in vec3 p) {
    vec3 dir, E;
    starLightAtPoint(s, p, dir, E);
    return max(E.r, max(E.g, E.b));
}


// PBR functions
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


float fresnelSchlick1(in float cosTheta, in float F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlick(in float cosTheta, in vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float geometrySmithGGX(in float NdotV, in float NdotL, in float roughness) {
    float a = roughness * roughness;
    float k = a * a * 0.5;
    float ggxV = NdotV / (NdotV * (1.0 - k) + k);
    float ggxL = NdotL / (NdotL * (1.0 - k) + k);
    return ggxV * ggxL;
}


vec3 sampleGGXVNDF(in vec3 Ve, in float roughness, in vec2 xi) {
    vec3 Vh = normalize(vec3(roughness * Ve.x, roughness * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;

    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1,0,0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(xi.x);

    float phi = 2.0 * PI * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);

    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    return normalize(vec3(roughness * Nh.x, roughness * Nh.y, max(0.0, Nh.z)));
}

float G1(in float NdotX, in float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}


vec3 evalBSDF(
    in vec3 N, in vec3 V, in vec3 L, in Material mat, in vec3 albedo, in vec3 F0,
    in float baseAvailable, in float specProb, in float coatProb, in float diffProb,
    out float pdf
 ) {
    pdf = 0.0;
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);

    float NdotV = max(dot(N, V), 1e-4);
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    vec3 diffuseColor = albedo * (1.0 - mat.metallic);
    vec3 kd = 1.0 - fresnelSchlick(NdotV, F0);
    vec3 sheenAdd = mat.sheenColor * mat.sheen * pow(1.0 - NdotV, 5.0);
    vec3 f = baseAvailable * (diffuseColor * kd + sheenAdd) * INV_PI;
    pdf += diffProb * NdotL * INV_PI;
    {
        float D = D_GGX(NdotH, max(mat.roughness, 0.0001));
        float G = geometrySmithGGX(NdotV, NdotL, mat.roughness);
        float G1v = geometrySmithGGX(NdotV, 1.0, mat.roughness);
        f += baseAvailable * fresnelSchlick(VdotH, F0) * (D * G / (4.0 * NdotV * NdotL));
        pdf += specProb * G1v * D / (4.0 * NdotV);
    }
    {
        float D = D_GGX(NdotH, max(mat.clearcoatRoughness, 0.0001));
        float G = geometrySmithGGX(NdotV, NdotL, mat.clearcoatRoughness);
        float G1v = geometrySmithGGX(NdotV, 1.0, mat.clearcoatRoughness);
        float Fc = mat.clearcoat * fresnelSchlick1(VdotH, 0.04);
        f += Fc * (D * G / (4.0 * NdotV * NdotL));
        pdf += coatProb * G1v * D / (4.0 * NdotV);
    }
    return f;
}


#endif
#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require


layout(buffer_reference, scalar) buffer Vertices {vec3 v[];};
layout(buffer_reference, scalar) buffer Indices  {uint i[];};


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


#endif
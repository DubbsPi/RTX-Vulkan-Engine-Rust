use std::f32::consts::PI;
use cgmath::InnerSpace;
use noise::{Fbm, NoiseFn, Perlin};

use crate::Vec3;
use crate::Vertex;
use crate::Material;

use crate::scene::dummy_root_skeleton;
use crate::scene::Model;
use crate::game::Planet;


pub fn generate_noise_plane(fbm: &Fbm::<Perlin>, planet: &Planet, resolution: u32, scale: f64) -> Model {
    let res = if resolution % 2 == 1 {
        resolution + 1
    } else {
        resolution
    };
    
    let half_res = res as i32 / 2;
    let mut vertices = Vec::new();

    // Compute vertices
    for x in -half_res..=half_res {
        let a = x as f32 / half_res as f32;
        for z in -half_res..=half_res {
            let b = z as f32 / half_res as f32;

            let r = (a * a + b * b).sqrt().min(1.0);
            let phi = a.atan2(b);
            let theta = r * (PI * 0.5);

            let dir = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );

            let wrapped = dir * planet.radius as f32 * 1.00001;
            let height = fbm.get([wrapped.x as f64, wrapped.y as f64, wrapped.z as f64]) * scale;
            let terrain = dir * (planet.radius as f32 + height as f32) - Vec3::new(0.0, planet.radius as f32, 0.0);

            vertices.push(Vertex::new(terrain, Vec3::new(0.0, 0.0, 0.0), None, None, None));
        }
    }

    // Compute indices
    let mut indices = Vec::new();
    let stride = res + 1;

    for x in 0..res {
        for z in 0..res {
            let bl = x * stride + z;
            let br = x * stride + (z + 1);
            let tl = (x + 1) * stride + z;
            let tr = (x + 1) * stride + (z + 1);

            // Triangle 1
            indices.push(bl as u32);
            indices.push(br as u32);
            indices.push(tl as u32);

            // Triangle 2
            indices.push(br as u32);
            indices.push(tr as u32);
            indices.push(tl as u32);
        }
    }

    // Compute normals
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (v0, v1, v2) = (vertices[i0].pos, vertices[i1].pos, vertices[i2].pos);

        let face_normal = (v1 - v0).cross(v2 - v0);

        vertices[i0].normal += face_normal;
        vertices[i1].normal += face_normal;
        vertices[i2].normal += face_normal;
    }

    for vertex in vertices.iter_mut() {
        vertex.normal = vertex.normal.normalize();
    }


    let material_ids = vec![0; indices.len() / 3]; 

    let materials = vec![Material {
        albedo: Vec3::new(0.5, 0.25, 0.75),
        albedo_texture_index: -1,
        metallic: 0.0,
        roughness: 1.0,
        emission: Vec3::new(0.0, 0.0, 0.0),
        transmission: 0.0,
        ior: 0.0,
        absorption_color: Vec3::new(0.0, 0.0, 0.0),
        dispersion: 0.0,
        specular: 0.0,
        clearcoat: 0.0,
        clearcoat_roughness: 1.0,
        sheen: 0.0,
        sheen_color: Vec3::new(0.0, 0.0, 0.0),
        alpha_mode: 0,
        alpha_cutoff: 0.5,
    }];


    Model {
        vertices, indices,
        material_ids, materials,
        skeleton: Some(dummy_root_skeleton()), animations: Vec::new(),
    }
}

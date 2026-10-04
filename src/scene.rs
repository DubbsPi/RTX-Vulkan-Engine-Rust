use gltf::Document;
use gltf::image::Data;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::cmp::min;

use indexmap::IndexSet;

use cgmath::SquareMatrix;
use cgmath::vec3;
use cgmath::Matrix;
use cgmath::InnerSpace;


use anyhow::{anyhow, Result};
use log::*;

use vulkanalia::vk::DeviceV1_0;
use vulkanalia::vk::HasBuilder;
use vulkanalia::vk::Handle;


use crate::StringOrInt;

use crate::common::Vertex;
use crate::common::Material;
use crate::common::Vec3;
use crate::common::Mat4;
use crate::common::Skeleton;
use crate::common::Bone;
use crate::common::{AnimationChannel, AnimationClip, AnimationPlayer};


#[derive(Clone)]
pub struct UintRange {
    pub min: u32,
    pub max: u32,
}


#[derive(Clone)]
pub struct ModelInfo {
    pub model_vertex_range: UintRange,
    pub model_index_range: UintRange,
    pub skeleton: Option<Skeleton>,
    pub model_class: ModelClass,
    pub geom_ranges: Vec<GeomRange>,
    pub blas_group: Option<u32>,
    pub instance_transform: Mat4,
}

#[derive(Clone, Copy)]
pub struct GeomRange {
    pub tri_start: u32,
    pub tri_count: u32,
    pub opaque: bool,
}


#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ModelClass {
    Rigid,
    Deformable,
}


pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<Material>,

    pub skeleton: Option<Skeleton>,
    pub animations: Vec<AnimationClip>,
}

impl Model {
    fn partition_by_opacity(&mut self) {
        let tri_count = self.indices.len() / 3;
        let is_opaque = |m: &Model, t: usize| {
            m.materials.get(m.material_ids[t] as usize).map_or(true, |mat| mat.alpha_mode == 0)
        };

        let mut order: Vec<usize> = (0..tri_count).collect();
        order.sort_by_key(|&t| !is_opaque(self, t));
        if order.iter().enumerate().all(|(i, &t)| i == t) { return; }

        let mut new_indices = Vec::with_capacity(self.indices.len());
        let mut new_ids = Vec::with_capacity(tri_count);
        for &t in &order {
            new_indices.extend_from_slice(&self.indices[t * 3..t * 3 + 3]);
            new_ids.push(self.material_ids[t]);
        }
        self.indices = new_indices;
        self.material_ids = new_ids;
    }
}

impl From<&Model> for Model {
    fn from(item: &Model) -> Self {
        Model {vertices: item.vertices.clone(), indices: item.indices.clone(), material_ids: item.material_ids.clone(), materials: item.materials.clone(), skeleton: item.skeleton.clone(), animations: item.animations.clone()}
    }
}


pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<Material>,

    material_map: HashMap<Material, u32>,

    pub model_info: Vec<ModelInfo>,
    pub local_transforms: Vec<Mat4>,

    pub animations: Vec<AnimationPlayer>,

    pub model_names: IndexSet<String>,
}

impl Scene {
    pub fn new() -> Self {
        info!("Scene initiated!");
        Self {
            vertices: Vec::new(), indices: Vec::new(),
            material_ids: Vec::new(), materials: Vec::new(),
            material_map: HashMap::new(),
            model_info: Vec::new(),
            local_transforms: Vec::new(),
            animations: Vec::new(),
            model_names: IndexSet::new(),
        }
    }

    pub unsafe fn load_model_into_memory(
        &mut self,
        path: &str,
        instance: &crate::Instance,
        device: &crate::Device,
        data: &mut crate::AppData,
    ) -> Result<Model> { unsafe {
        let source_path = Path::new(path);

        info!("Loading model from cache: {}", source_path.display());

        let asset = build_asset(load_gltf(path)?)?;

        finish_model_load(asset, path, instance, device, data)
    }}

    pub unsafe fn load_model_from_bytes(
        &mut self,
        bytes: &[u8],
        instance: &crate::Instance,
        device: &crate::Device,
        data: &mut crate::AppData,
    ) -> Result<Model> { unsafe {

        let asset = build_asset(load_gltf_slice(bytes)?)?;

        finish_model_load(asset, "embedded model", instance, device, data)
    }}

    pub fn add_model_to_scene(&mut self, model: &Model, model_class: ModelClass, model_name: Option<String>) {
        let vertex_offset = self.vertices.len() as u32;
        let index_offset = self.indices.len() as u32;

        // Default matrix
        let transform_matrix = Mat4::identity();
        self.local_transforms.push(transform_matrix);

        // Add vertices/indices
        let base_vertex = self.vertices.len() as u32;
        self.vertices.extend(&model.vertices);
        self.indices.extend(model.indices.iter().map(|&i| i + base_vertex));

        let mut material_mapping = Vec::with_capacity(model.materials.len());

        // Optimized material reusing
        for material in &model.materials {
            let scene_material_id = if let Some(&id) = self.material_map.get(material) {
                id
            } else {
                let id = self.materials.len() as u32;

                let mat = if model_name == Some("Cubes".to_owned()) {
                    let mut m = *material;
                    m.alpha_mode = 2;
                    m.ior = 1.5;
                    m.transmission = 1.0;
                    m.dispersion = 6.5;
                    m.absorption_color = vec3(0.1, 1.0, 1.0);
                    m
                } else {
                    *material
                };
                
                self.materials.push(mat);
                self.material_map.insert(*material, id);

                id
            };

            material_mapping.push(scene_material_id);
        }

        // Remap triangle material ids
        self.material_ids.extend(
            model.material_ids
                .iter()
                .map(|&model_material_id| {
                    material_mapping[model_material_id as usize]
                })
        );

        let mut geom_ranges: Vec<GeomRange> = Vec::new();

        if model.indices.len() / 3 > model.material_ids.len() {
            match &model_name {
                Some(mn) => error!("Not enough material ids for model {}, some tris will not render!", mn),
                None => error!("Not enough material ids for model {}, some tris will not render!", self.model_info.len())
            }
        }
        for t in 0..min(model.indices.len() / 3, model.material_ids.len()) {
            let opaque = model.materials
                .get(model.material_ids[t] as usize)
                .map_or(true, |m| m.alpha_mode == 0);
            match geom_ranges.last_mut() {
                Some(r) if r.opaque == opaque => r.tri_count += 1,
                _ => geom_ranges.push(GeomRange {tri_start: t as u32, tri_count: 1, opaque}),
            }
        }
        
        // Save model info
        self.model_info.push(ModelInfo {
            model_vertex_range: UintRange {min: vertex_offset, max: vertex_offset + model.vertices.len() as u32},
            model_index_range: UintRange {min: index_offset, max: index_offset + model.indices.len() as u32},
            skeleton: model.skeleton.clone(),
            model_class,
            geom_ranges, blas_group: None,
            instance_transform: Mat4::identity(),
        });

        match model_name {
            Some(mn) => self.model_names.insert(mn),
            None => self.model_names.insert(self.model_info.len().to_string()),
        };
        self.animations.push(AnimationPlayer::new(model.animations.clone()));
    }


    pub fn search_for_model_id(&self, model_name: String) -> Option<usize> {
        self.model_names.get_index_of(&model_name)
    }
    
    pub fn set_blas_group(&mut self, id: StringOrInt, group: Option<u32>) {
        let Some(mi) = self.resolve(id) else {return};

        let world = self.model_info[mi].instance_transform * self.local_transforms[mi];
        self.model_info[mi].blas_group = group;

        let others: Vec<usize> = self.group_members(mi).into_iter().filter(|&m| m != mi).collect();
        match others.first() {
            Some(&anchor) => {
                let t = self.model_info[anchor].instance_transform;
                let inv = t.invert().unwrap_or_else(Mat4::identity);
                self.model_info[mi].instance_transform = t;
                self.local_transforms[mi] = inv * world;
            }
            None => {
                self.model_info[mi].instance_transform = world;
                self.local_transforms[mi] = Mat4::identity();
            }
        }
    }

    pub fn resolve(&self, id: StringOrInt) -> Option<usize> {
        match id {
            StringOrInt::Str(s) => self.search_for_model_id(s),
            StringOrInt::Int(i) => (i < self.model_info.len()).then_some(i),
        }
    }


    pub fn group_members(&self, mi: usize) -> Vec<usize> {
        match self.model_info[mi].blas_group {
            None => vec![mi],
            Some(g) => {
                let class = self.model_info[mi].model_class;
                self.model_info.iter().enumerate()
                    .filter(|(_, m)| m.blas_group == Some(g) && m.model_class == class)
                    .map(|(i, _)| i)
                    .collect()
            }
        }
    }

    pub fn transform_model(&mut self, id: StringOrInt, delta: Mat4) {
        let Some(mi) = self.resolve(id) else { return };
        for m in self.group_members(mi) {
            let t = &mut self.model_info[m].instance_transform;
            *t = delta * *t;
        }
    }

    pub fn translate_model(&mut self, id: StringOrInt, offset: Vec3) {
        self.transform_model(id, Mat4::from_translation(offset));
    }

    pub fn scale_model(&mut self, id: StringOrInt, scale: Vec3) {
        self.transform_model(id, Mat4::from_nonuniform_scale(scale.x, scale.y, scale.z));
    }

    pub fn set_instance_transform(&mut self, id: StringOrInt, transform: Mat4) {
        let Some(mi) = self.resolve(id) else { return };
        for m in self.group_members(mi) {
            self.model_info[m].instance_transform = transform;
        }
    }
}


type GltfParts = (
    Vec<Vertex>, Vec<u32>, Vec<u32>, Vec<Material>,
    Vec<Data>, Option<Skeleton>, Vec<AnimationClip>,
);

unsafe fn finish_model_load(
    asset: CachedAsset,
    label: &str,
    instance: &crate::Instance,
    device: &crate::Device,
    data: &mut crate::AppData,
) -> Result<Model> { unsafe {
    let skeleton = asset.skeleton.or_else(|| Some(dummy_root_skeleton()));

    let texture_offset = data.textures.len() as i32;
    let result = create_cached_textures(instance, device, data, &asset.textures)?;
    data.textures.extend(result);

    let mut materials = asset.materials;
    for material in &mut materials {
        if material.albedo_texture_index >= 0 {
            material.albedo_texture_index += texture_offset;
        }
    }

    info!("Loaded {} into memory", label);

    let mut model = Model {
        vertices: asset.vertices,
        indices: asset.indices,
        material_ids: asset.material_ids,
        materials,
        skeleton,
        animations: asset.animations,
    };
    model.partition_by_opacity();
    Ok(model)
}}

fn node_local_matrix(node: &gltf::Node) -> Mat4 {
    let (t, r, s) = node.transform().decomposed();
    Mat4::from_translation(vec3(t[0], t[1], t[2]))
        * Mat4::from(cgmath::Quaternion::new(r[3], r[0], r[1], r[2]))
        * Mat4::from_nonuniform_scale(s[0], s[1], s[2])
}

fn walk_node(
    node: &gltf::Node,
    parent_world: Mat4,
    buffers: &[gltf::buffer::Data],
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    material_ids: &mut Vec<u32>,
) {
    let world = parent_world * node_local_matrix(node);

    let normal_mat = world.invert().map(|m| m.transpose()).unwrap_or(world);

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                warn!("Skipping non-triangle primitive in mesh {:?}", mesh.name());
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<Vec3> = reader
                .read_positions()
                .expect("Primitive missing POSITION attribute")
                .map(|p| {
                    let world_p = world * cgmath::Vector4::new(p[0], p[1], p[2], 1.0);
                    vec3(world_p.x, world_p.y, world_p.z)
                })
                .collect();

            let normals = match reader.read_normals() {
                Some(iter) => iter
                    .map(|n| {
                        let world_n = normal_mat * cgmath::Vector4::new(n[0], n[1], n[2], 0.0);
                        vec3(world_n.x, world_n.y, world_n.z).normalize()
                    })
                    .collect(),
                None => {
                    let raw_indices: Vec<u32> = reader
                        .read_indices()
                        .map(|i| i.into_u32().collect())
                        .unwrap_or_else(|| (0..positions.len() as u32).collect());
                    compute_vertex_normals(&positions, &raw_indices)
                }
            };

                        let uvs: Vec<Option<cgmath::Vector2<f32>>> = match reader.read_tex_coords(0) {
                Some(read_tex_coords) => read_tex_coords
                    .into_f32()
                    .map(|uv| Some(cgmath::vec2(uv[0], uv[1])))
                    .collect(),
                None => vec![None; positions.len()],
            };
 
            let joint_indices: Vec<[u16; 4]> = match reader.read_joints(0) {
                Some(read_joints) => read_joints.into_u16().collect(),
                None => vec![[0, 0, 0, 0]; positions.len()],
            };
 
            let joint_weights: Vec<[f32; 4]> = match reader.read_weights(0) {
                Some(read_weights) => read_weights.into_f32().collect(),
                None => vec![[1.0, 0.0, 0.0, 0.0]; positions.len()],
            };
 
 
            let vertex_offset = vertices.len() as u32;
            vertices.extend(
                positions.iter()
                    .zip(normals.iter())
                    .zip(uvs.iter())
                    .zip(joint_indices.iter())
                    .zip(joint_weights.iter())
                    .map(|((((&p, &n), &uv), &ji), &jw)| Vertex::new(p, n, uv, Some(ji), Some(jw))),
            );
 
            let prim_indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().map(|i| i + vertex_offset).collect(),
                None => (0..positions.len() as u32).map(|i| i + vertex_offset).collect(),
            };
 
            let mat_id = primitive.material().index().unwrap_or(0) as u32;
            let triangle_count = prim_indices.len() / 3;
            material_ids.extend(std::iter::repeat(mat_id).take(triangle_count));
 
            indices.extend(prim_indices);
        }
    }

    for child in node.children() {
        walk_node(&child, world, buffers, vertices, indices, material_ids);
    }
}

fn convert_materials(document: &Document) -> Vec<Material> {
    document
        .materials()
        .map(|m| {
            let pbr = m.pbr_metallic_roughness();
            
            // Albedo
            let base_color = pbr.base_color_factor();
            let albedo = Vec3::new(base_color[0], base_color[1], base_color[2]);

            let albedo_texture_index = pbr
                .base_color_texture()
                .map(|tex| tex.texture().index() as i32)
                .unwrap_or(-1);

            let metallic = pbr.metallic_factor();
            let roughness = pbr.roughness_factor();

            // Emission
            let emissive_factor = m.emissive_factor();
            let emissive_strength = m
                .emissive_strength()
                .unwrap_or(1.0);
            let emission = Vec3::new(
                emissive_factor[0] * emissive_strength,
                emissive_factor[1] * emissive_strength,
                emissive_factor[2] * emissive_strength,
            );

            // Transmission
            let transmission = m
                .transmission()
                .map(|t| t.transmission_factor())
                .unwrap_or(0.0);

            // Ior
            let ior = m.ior().unwrap_or(1.5);

            // Absorption color
            let absorption_color = m
                .volume()
                .map(|v| {
                    let c = v.attenuation_color();
                    let d = v.attenuation_distance();
                    if d.is_finite() && d > 0.0 {
                        let sigma = |x: f32| -x.max(1e-4).ln() / d;
                        Vec3::new(sigma(c[0]), sigma(c[1]), sigma(c[2]))
                    } else {
                        Vec3::new(0.0, 0.0, 0.0)
                    }
                })
                .unwrap_or(Vec3::new(0.0, 0.0, 0.0));

            // Dispersion
            let dispersion = m
                .extension_value("KHR_materials_dispersion")
                .and_then(|v| v.get("dispersion"))
                .and_then(|d| d.as_f64())
                .map(|d| d as f32)
                .unwrap_or(0.0);

            // Specular
            let specular = m
                .specular()
                .map(|s| s.specular_factor())
                .unwrap_or(1.0);

            // Clearcoat
            let (clearcoat, clearcoat_roughness) = m
                .clearcoat()
                .map(|c| (c.clearcoat_factor(), c.clearcoat_roughness_factor()))
                .unwrap_or((0.0, 0.0));

            // Sheen
            let (sheen_color, sheen) = m
                .sheen()
                .map(|s| {
                    let color = s.sheen_color_factor();
                    (Vec3::new(color[0], color[1], color[2]), 1.0)
                })
                .unwrap_or((Vec3::new(0.0, 0.0, 0.0), 0.0));
            
            // Alpha mode
            let (alpha_mode, alpha_cutoff) = match m.alpha_mode() {
                gltf::material::AlphaMode::Opaque => (0u32, 0.0f32),
                gltf::material::AlphaMode::Mask => (1u32, m.alpha_cutoff().unwrap_or(0.5)),
                gltf::material::AlphaMode::Blend => (2u32, 0.0f32),
            };

            Material {
                albedo,
                albedo_texture_index,
                metallic,
                roughness,
                emission,
                transmission,
                ior,
                absorption_color,
                dispersion,
                specular,
                clearcoat,
                clearcoat_roughness,
                sheen,
                sheen_color,
                alpha_mode,
                alpha_cutoff,
            }
        })
        .collect()
}

fn compute_vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (v0, v1, v2) = (positions[i0], positions[i1], positions[i2]);

        let face_normal = (v1 - v0).cross(v2 - v0);

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    for n in &mut normals {
        let radicand = n.x * n.x + n.y * n.y + n.z * n.z;
        if radicand > 0.0 {
            let radical = radicand.sqrt();
            n.x /= radical;
            n.y /= radical;
            n.z /= radical;
        }
    }

    normals
}

unsafe fn create_texture_image(
    instance: &crate::Instance,
    device: &crate::Device,
    data: &crate::AppData,
    command_pool: crate::vk::CommandPool,
    queue: crate::vk::Queue,
    pixels: &[u8],
    width: u32,
    height: u32,
    format: crate::vk::Format,
) -> Result<(crate::vk::Image, crate::vk::DeviceMemory, crate::vk::ImageView)> { unsafe {
    let size = pixels.len() as u64;

    // Staging buffer
    let (staging_buffer_raw, staging_memory) = crate::create_buffer(
        instance, device, data, size,
        crate::vk::BufferUsageFlags::TRANSFER_SRC,
        crate::vk::MemoryPropertyFlags::HOST_VISIBLE | crate::vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let staging = crate::StagingBuffer {device, buffer: staging_buffer_raw, memory: staging_memory};

    let mem = device.map_memory(staging_memory, 0, size, crate::vk::MemoryMapFlags::empty())?;
    
    crate::memcpy(pixels.as_ptr(), mem.cast::<u8>(), size as usize);
    device.unmap_memory(staging_memory);

    // Device local image
    let image_info = crate::vk::ImageCreateInfo::builder()
        .image_type(crate::vk::ImageType::_2D)
        .format(format) // sRGB — matches glTF's baseColor color space
        .extent(crate::vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1) // no mipmaps yet — fine for now, add later if aliasing shows up
        .array_layers(1)
        .samples(crate::vk::SampleCountFlags::_1)
        .tiling(crate::vk::ImageTiling::OPTIMAL)
        .usage(crate::vk::ImageUsageFlags::TRANSFER_DST | crate::vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(crate::vk::SharingMode::EXCLUSIVE)
        .initial_layout(crate::vk::ImageLayout::UNDEFINED);

    let image = device.create_image(&image_info, None)?;
    let requirements = device.get_image_memory_requirements(image);

    let memory_info = crate::vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(crate::get_memory_type_index(
            instance, data, crate::vk::MemoryPropertyFlags::DEVICE_LOCAL, requirements,
        )?);

    let memory = device.allocate_memory(&memory_info, None)?;
    device.bind_image_memory(image, memory, 0)?;


    let alloc_info = crate::vk::CommandBufferAllocateInfo::builder()
        .level(crate::vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc_info)?[0];
    device.begin_command_buffer(cmd, &crate::vk::CommandBufferBeginInfo::builder()
        .flags(crate::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

    let subresource = crate::vk::ImageSubresourceRange::builder()
        .aspect_mask(crate::vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
        .build();

    let to_transfer_barrier = crate::vk::ImageMemoryBarrier::builder()
        .old_layout(crate::vk::ImageLayout::UNDEFINED)
        .new_layout(crate::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(crate::vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(crate::vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(crate::vk::AccessFlags::empty())
        .dst_access_mask(crate::vk::AccessFlags::TRANSFER_WRITE);

    device.cmd_pipeline_barrier(
        cmd, crate::vk::PipelineStageFlags::TOP_OF_PIPE, crate::vk::PipelineStageFlags::TRANSFER,
        crate::vk::DependencyFlags::empty(), &[] as &[crate::vk::MemoryBarrier], &[] as &[crate::vk::BufferMemoryBarrier],
        &[to_transfer_barrier],
    );

    let region = crate::vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(crate::vk::ImageSubresourceLayers::builder()
            .aspect_mask(crate::vk::ImageAspectFlags::COLOR)
            .mip_level(0).base_array_layer(0).layer_count(1).build())
        .image_extent(crate::vk::Extent3D { width, height, depth: 1 });

    device.cmd_copy_buffer_to_image(
        cmd, staging.buffer, image, crate::vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region],
    );

    let to_shader_read_barrier = crate::vk::ImageMemoryBarrier::builder()
        .old_layout(crate::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(crate::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(crate::vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(crate::vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(crate::vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(crate::vk::AccessFlags::SHADER_READ);

    device.cmd_pipeline_barrier(
        cmd, crate::vk::PipelineStageFlags::TRANSFER, crate::vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        crate::vk::DependencyFlags::empty(), &[] as &[crate::vk::MemoryBarrier], &[] as &[crate::vk::BufferMemoryBarrier],
        &[to_shader_read_barrier],
    );

    device.end_command_buffer(cmd)?;
    device.queue_submit(queue, &[crate::vk::SubmitInfo::builder().command_buffers(&[cmd])], crate::vk::Fence::null())?;
    device.queue_wait_idle(queue)?;
    device.free_command_buffers(command_pool, &[cmd]);

    let view_info = crate::vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(crate::vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource);
    let view = device.create_image_view(&view_info, None)?;

    Ok((image, memory, view))
}}

fn extract_skeleton(document: &Document, buffers: &[gltf::buffer::Data]) -> Option<Skeleton> {
    let skin = document.skins().next()?;
    let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));

    let inverse_bind_matrices: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
        Some(iter) => iter.map(|m| Mat4::from(m)).collect(),
        None => vec![Mat4::identity(); skin.joints().count()],
    };

    let joint_nodes: Vec<gltf::Node> = skin.joints().collect();

    let node_to_bone: HashMap<usize, usize> = joint_nodes
        .iter()
        .enumerate()
        .map(|(bone_index, node)| (node.index(), bone_index))
        .collect();

    let bones: Vec<Bone> = joint_nodes
        .iter()
        .enumerate()
        .map(|(bone_idx, node)| {
            let (translation, rotation, scale) = node.transform().decomposed();
            let local_transform =
                Mat4::from_translation(vec3(translation[0], translation[1], translation[2]))
                    * Mat4::from(cgmath::Quaternion::new(rotation[3], rotation[0], rotation[1], rotation[2]))
                    * Mat4::from_nonuniform_scale(scale[0], scale[1], scale[2]);

            let children: Vec<usize> = node.children()
                .filter_map(|child| node_to_bone.get(&child.index()).copied())
                .collect();

            Bone {
                node_index: node.index(),
                children,
                local_transform,
                inverse_bind_matrix: inverse_bind_matrices[bone_idx],
            }
        })
        .collect();

    let child_set: std::collections::HashSet<usize> =
        bones.iter().flat_map(|b| b.children.iter().copied()).collect();
    let root_bones: Vec<usize> = (0..bones.len())
        .filter(|i| !child_set.contains(i))
        .collect();

    Some(Skeleton {bones, root_bones})
}

pub fn dummy_root_skeleton() -> Skeleton {
    Skeleton {
        bones: vec![Bone {
            node_index: 0,
            children: vec![],
            local_transform: Mat4::identity(),
            inverse_bind_matrix: Mat4::identity(),
        }],
        root_bones: vec![0],
    }
}

fn extract_animations(document: &Document, buffers: &[gltf::buffer::Data], skeleton: &Skeleton) -> Vec<AnimationClip> {
    let node_to_bone: HashMap<usize, usize> = skeleton.bones
        .iter()
        .enumerate()
        .map(|(bone_idx, bone)| (bone.node_index, bone_idx))
        .collect();

    document.animations().filter_map(|anim| {
        let mut channels_by_bone: HashMap<usize, AnimationChannel> = HashMap::new();
        let mut max_time = 0.0f32;

        for channel in anim.channels() {
            let target_node = channel.target().node()?.index();
            let Some(&bone_index) = node_to_bone.get(&target_node) else {
                continue; // Animates a node that isn't part of this skeleton (e.g. a camera) — skip it
            };

            let reader = channel.reader(|b| Some(&buffers[b.index()]));
            let Some(inputs) = reader.read_inputs() else { continue };
            let times: Vec<f32> = inputs.collect();
            if let Some(&t) = times.last() {
                max_time = max_time.max(t);
            }

            let entry = channels_by_bone.entry(bone_index).or_insert_with(|| AnimationChannel {
                bone_index,
                translations: Vec::new(),
                rotations: Vec::new(),
                scales: Vec::new(),
            });

            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(vals)) => {
                    entry.translations = times.iter().copied()
                        .zip(vals.map(|v| cgmath::vec3(v[0], v[1], v[2])))
                        .collect();
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(vals)) => {
                    entry.rotations = times.iter().copied()
                        .zip(vals.into_f32().map(|r| cgmath::Quaternion::new(r[3], r[0], r[1], r[2])))
                        .collect();
                }
                Some(gltf::animation::util::ReadOutputs::Scales(vals)) => {
                    entry.scales = times.iter().copied()
                        .zip(vals.map(|v| cgmath::vec3(v[0], v[1], v[2])))
                        .collect();
                }
                _ => {}
            }
        }

        if channels_by_bone.is_empty() {
            return None;
        }
        
        Some(AnimationClip {
            name: anim.name().unwrap_or("unnamed_animation").to_string(),
            duration: max_time,
            channels: channels_by_bone.into_values().collect(),
        })
    }).collect()
}

fn load_gltf(path: &str) -> Result<GltfParts> {
    let (document, buffers, images) = gltf::import(path)?;
    process_gltf(document, buffers, images)
}

fn load_gltf_slice(bytes: &[u8]) -> Result<GltfParts> {
    let (document, buffers, images) = gltf::import_slice(bytes)?;
    process_gltf(document, buffers, images)
}

fn process_gltf(
    document: Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<Data>,
) -> Result<GltfParts> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut material_ids = Vec::new();

    for scene in document.scenes() {
        for node in scene.nodes() {
            walk_node(&node, Mat4::identity(), &buffers, &mut vertices, &mut indices, &mut material_ids);
        }
    }

    let skeleton = extract_skeleton(&document, &buffers);
    let materials = convert_materials(&document);
    let animations = match &skeleton {
        Some(skel) => extract_animations(&document, &buffers, skel),
        None => Vec::new(),
    };

    Ok((vertices, indices, material_ids, materials, images, skeleton, animations))
}

fn build_asset(parts: GltfParts) -> Result<CachedAsset> {
    let (vertices, indices, material_ids, materials, images, skeleton, animations) = parts;
    let textures = images.iter().map(convert_image).collect::<Result<Vec<_>>>()?;
    Ok(CachedAsset { vertices, indices, material_ids, materials, textures, skeleton, animations })
}


impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.albedo_texture_index == other.albedo_texture_index
            && self.albedo.x.to_bits() == other.albedo.x.to_bits()
            && self.albedo.y.to_bits() == other.albedo.y.to_bits()
            && self.albedo.z.to_bits() == other.albedo.z.to_bits()
            && self.metallic.to_bits() == other.metallic.to_bits()
            && self.roughness.to_bits() == other.roughness.to_bits()
            && self.alpha_mode == other.alpha_mode
            && self.alpha_cutoff.to_bits() == other.alpha_cutoff.to_bits()
    }
}

impl Eq for Material {}

impl Hash for Material {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.albedo_texture_index.hash(state);
        self.albedo.x.to_bits().hash(state);
        self.albedo.y.to_bits().hash(state);
        self.albedo.z.to_bits().hash(state);
        self.metallic.to_bits().hash(state);
        self.roughness.to_bits().hash(state);
        self.alpha_mode.hash(state);
        self.alpha_cutoff.to_bits().hash(state);
    }
}


#[derive(Clone)]
pub struct CachedTexture {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub pixels: Vec<u8>,
}

#[expect(unused)]
#[derive(Clone, Copy)]
pub enum TextureFormat {
    R8,
    R8G8,
    R8G8B8,
    R8G8B8A8,

    R16,
    R16G16,
    R16G16B16,
    R16G16B16A16,
}

#[expect(unused)]
impl TextureFormat {
    fn to_u8(self) -> u8 {
        match self {
            Self::R8 => 0,
            Self::R8G8 => 1,
            Self::R8G8B8 => 2,
            Self::R8G8B8A8 => 3,
            Self::R16 => 4,
            Self::R16G16 => 5,
            Self::R16G16B16 => 6,
            Self::R16G16B16A16 => 7,
        }
    }

    fn from_u8(value: u8) -> Result<Self> {
        Ok(match value {
            0 => Self::R8,
            1 => Self::R8G8,
            2 => Self::R8G8B8,
            3 => Self::R8G8B8A8,
            4 => Self::R16,
            5 => Self::R16G16,
            6 => Self::R16G16B16,
            7 => Self::R16G16B16A16,
            _ => return Err(anyhow!("Unknown cached texture format {}", value)),
        })
    }
}

pub struct CachedAsset {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<Material>,
    pub textures: Vec<CachedTexture>,
    pub skeleton: Option<Skeleton>,
    pub animations: Vec<AnimationClip>,
}


unsafe fn create_cached_textures(
    instance: &crate::Instance,
    device: &crate::Device,
    data: &crate::AppData,
    textures: &[CachedTexture],
) -> Result<Vec<(crate::vk::Image, crate::vk::DeviceMemory, crate::vk::ImageView)>> { unsafe {
    let mut result = Vec::with_capacity(textures.len());

    for texture in textures {
        let vk_format = match texture.format {
            TextureFormat::R8 =>
                crate::vk::Format::R8_UNORM,

            TextureFormat::R8G8 =>
                crate::vk::Format::R8G8_UNORM,

            TextureFormat::R8G8B8 => {
                return Err(anyhow!(
                    "R8G8B8 cached textures must be expanded before upload"
                ));
            }

            TextureFormat::R8G8B8A8 =>
                crate::vk::Format::R8G8B8A8_SRGB,

            TextureFormat::R16 =>
                crate::vk::Format::R16_UNORM,

            TextureFormat::R16G16 =>
                crate::vk::Format::R16G16_UNORM,

            TextureFormat::R16G16B16 => {
                return Err(anyhow!(
                    "R16G16B16 cached textures must be expanded before upload"
                ));
            }

            TextureFormat::R16G16B16A16 =>
                crate::vk::Format::R16G16B16A16_UNORM,
        };

        let (image, memory, view) = create_texture_image(
            instance,
            device,
            data,
            data.command_pool,
            data.graphics_queue,
            &texture.pixels,
            texture.width,
            texture.height,
            vk_format,
        )?;

        result.push((image, memory, view));
    }

    Ok(result)
}}

fn convert_image(img: &gltf::image::Data) -> Result<CachedTexture> {
    match img.format {
        gltf::image::Format::R8 => {
            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R8,
                pixels: img.pixels.clone(),
            })
        }

        gltf::image::Format::R8G8 => {
            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R8G8,
                pixels: img.pixels.clone(),
            })
        }

        gltf::image::Format::R8G8B8 => {
            let pixels = img.pixels
                .chunks_exact(3)
                .flat_map(|rgb| {
                    [rgb[0], rgb[1], rgb[2], 255]
                })
                .collect();

            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R8G8B8A8,
                pixels,
            })
        }

        gltf::image::Format::R8G8B8A8 => {
            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R8G8B8A8,
                pixels: img.pixels.clone(),
            })
        }

        gltf::image::Format::R16 => {
            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R16,
                pixels: img.pixels.clone(),
            })
        }

        gltf::image::Format::R16G16 => {
            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R16G16,
                pixels: img.pixels.clone(),
            })
        }

        gltf::image::Format::R16G16B16 => {
            let input: &[u16] = bytemuck::cast_slice(&img.pixels);

            let pixels = input
                .chunks_exact(3)
                .flat_map(|rgb| {
                    [rgb[0], rgb[1], rgb[2], u16::MAX]
                })
                .flat_map(u16::to_ne_bytes)
                .collect();

            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R16G16B16A16,
                pixels,
            })
        }

        gltf::image::Format::R16G16B16A16 => {
            Ok(CachedTexture {
                width: img.width,
                height: img.height,
                format: TextureFormat::R16G16B16A16,
                pixels: img.pixels.clone(),
            })
        }

        other => Err(anyhow!(
            "Unsupported glTF image format: {:?}",
            other
        )),
    }
}

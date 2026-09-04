// Some aspects of code are not used yet here. They will be used for though
#![expect(dead_code)]


use gltf::Document;
use gltf::image::Data;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use cgmath::SquareMatrix;
use cgmath::vec3;

use anyhow::{anyhow, Result};
use log::*;

use vulkanalia::vk::DeviceV1_0;
use vulkanalia::vk::HasBuilder;
use vulkanalia::vk::Handle;


use crate::common::Vertex;
use crate::common::Material;
use crate::common::Vec3;
use crate::common::Mat4;
use crate::common::Skeleton;
use crate::common::Bone;
use crate::common::{AnimationChannel, AnimationClip, AnimationPlayer};


pub struct UintRange {
    pub min: u32,
    pub max: u32,
}


pub struct ModelInfo {
    pub model_vertex_range: UintRange,
    pub model_index_range: UintRange,
    pub model_material_mappings: Vec<u32>,
    pub skeleton: Option<Skeleton>,
}

impl ModelInfo {
    fn new() -> Self {
        Self {model_vertex_range: UintRange {min: 0, max: 0}, model_index_range: UintRange {min: 0, max: 0}, model_material_mappings: Vec::new(), skeleton: None}
    }
}


pub struct Model {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    material_ids: Vec<u32>,
    materials: Vec<Material>,

    skeleton: Option<Skeleton>,
    animations: Vec<AnimationClip>,
}


pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<Material>,

    material_map: HashMap<Material, u32>,

    pub model_info: Vec<ModelInfo>,
    pub transform_matrices: Vec<Mat4>,

    pub animations: Vec<AnimationPlayer>,
}

impl Scene {
    pub fn new() -> Self {
        info!("Scene initiated!");
        Self {
            vertices: Vec::new(), indices: Vec::new(),
            material_ids: Vec::new(), materials: Vec::new(),
            material_map: HashMap::new(),
            model_info: Vec::new(),
            transform_matrices: Vec::new(),
            animations: Vec::new(),
        }
    }

    pub unsafe fn load_model_into_memory(path: &str, instance: &crate::Instance, device: &crate::Device, data: &mut crate::AppData) -> Result<Model> {
        let (vertices, indices, material_ids, mut materials, images, skeleton, animations) = load_gltf(path)?;

        unsafe {
            let texture_offset = data.textures.len() as i32;
            let result = create_gltf_textures(instance, device, data, &images)?;
            data.textures.extend(result);

            for material in &mut materials {
                if material.albedo_texture_index >= 0 {
                    material.albedo_texture_index += texture_offset;
                }
            }
        }
        
        info!("Loaded {} into memory", path);
        Ok(Model {vertices, indices, material_ids, materials, skeleton, animations})
    }

    pub fn add_model_to_scene(&mut self, model: &Model) {
        let vertex_offset = self.vertices.len() as u32;
        let index_offset = self.indices.len() as u32;

        // Default matrix
        let transform_matrix = Mat4::identity();
        self.transform_matrices.push(transform_matrix);

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

                self.materials.push(*material);
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
        
        // Save model info
        self.model_info.push(ModelInfo {
            model_vertex_range: UintRange {min: vertex_offset, max: vertex_offset + model.vertices.len() as u32},
            model_index_range: UintRange {min: index_offset, max: index_offset + model.indices.len() as u32},
            model_material_mappings: material_mapping,
            skeleton: model.skeleton.clone(),
        });

        self.animations.push(AnimationPlayer::new(model.animations.clone()));
    }

    pub fn translate_model(&mut self, model_id: usize, offset: Vec3) {
        self.transform_matrices[model_id] = Mat4::from_translation(offset) * self.transform_matrices[model_id];
    }

    pub fn scale_model(&mut self, model_id: usize, scale: Vec3) {
        self.transform_matrices[model_id] = Mat4::from_nonuniform_scale(scale.x, scale.y, scale.z) * self.transform_matrices[model_id];
    }

    pub fn set_transform(&mut self, model_id: usize, transform: Mat4) {
        self.transform_matrices[model_id] = transform;
    }
}


fn load_gltf(path: &str) -> Result<(Vec<Vertex>, Vec<u32>, Vec<u32>, Vec<Material>,Vec<Data>, Option<Skeleton>, Vec<AnimationClip>)> {
    let (document, buffers, images) = gltf::import(path)?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut material_ids = Vec::new();


    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                warn!("Skipping non-triangle primitive in mesh {:?}", mesh.name());
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<Vec3> = reader
                .read_positions()
                .ok_or_else(|| anyhow!("Primitive missing POSITION attribute"))?
                .map(|p| vec3(p[0], p[1], p[2]))
                .collect();

            let normals = match reader.read_normals() {
                Some(iter) => iter.map(|n| vec3(n[0], n[1], n[2])).collect(),
                None => {
                    // Fall back to flat shading
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
                None => vec![[0.0, 0.0, 0.0, 0.0]; positions.len()],
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


    let skeleton = extract_skeleton(&document, &buffers);
    let materials = convert_materials(&document);
    let animations = match &skeleton {
        Some(skel) => extract_animations(&document, &buffers, skel),
        None => Vec::new(),
    };

    
    Ok((vertices, indices, material_ids, materials, images, skeleton, animations))
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

            Material {
                albedo,
                albedo_texture_index,
                metallic,
                roughness,
                emission,
                transmission,
                ior,
                specular,
                clearcoat,
                clearcoat_roughness,
                sheen,
                sheen_color,
            }
        })
        .collect()
}

fn compute_vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];

    for tri in indices.chunks(3) {
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

unsafe fn create_gltf_textures(
    instance: &crate::Instance,
    device: &crate::Device,
    data: &crate::AppData,
    images: &[gltf::image::Data],
) -> Result<Vec<(crate::vk::Image, crate::vk::DeviceMemory, crate::vk::ImageView)>> { unsafe {
    let mut textures = Vec::new();

    for img in images {
        let (vk_format, _bpp) = gltf_to_vulkan(img.format)
            .ok_or_else(|| anyhow!("Unsupported glTF image format: {:?}", img.format))?;

        let pixels: Vec<u8> = match img.format {
            gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
            gltf::image::Format::R8G8B8 => img.pixels
                .chunks(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255u8])
                .collect(),
            gltf::image::Format::R16G16B16A16 => img.pixels.clone(),
            gltf::image::Format::R16G16B16 => {
                let u16_pixels: &[u16] = bytemuck::cast_slice(&img.pixels);
                u16_pixels
                    .chunks(3)
                    .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], u16::MAX])
                    .collect::<Vec<u16>>()
                    .into_iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect()
            }
            other => return Err(anyhow!("Unsupported glTF image format: {:?}", other)),
        };

        let (image, memory, view) = create_texture_image(
            instance, device, data,
            data.command_pool, data.graphics_queue,
            &pixels, img.width, img.height, vk_format,
        )?;

        textures.push((image, memory, view));
    }

    Ok(textures)
}}

fn gltf_to_vulkan(format: gltf::image::Format) -> Option<(crate::vk::Format, u32)> {
    match format {
        crate::R8G8B8A8 => Some((crate::vk::Format::R8G8B8A8_SRGB, 4)),
        crate::R8G8B8 => Some((crate::vk::Format::R8G8B8A8_SRGB, 4)),
        crate::R16G16B16A16 => Some((crate::vk::Format::R16G16B16A16_UNORM, 8)),
        crate::R16G16B16 => Some((crate::vk::Format::R16G16B16A16_UNORM, 8)),
        _ => None,
    }
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
                name: node.name().unwrap_or("unnamed_bone").to_string(),
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


impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.albedo_texture_index == other.albedo_texture_index
            && self.albedo.x.to_bits() == other.albedo.x.to_bits()
            && self.albedo.y.to_bits() == other.albedo.y.to_bits()
            && self.albedo.z.to_bits() == other.albedo.z.to_bits()
            && self.metallic.to_bits() == other.metallic.to_bits()
            && self.roughness.to_bits() == other.roughness.to_bits()
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
    }
}
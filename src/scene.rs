use gltf::Document;
use gltf::image::Data;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::fs::{self, File};

use indexmap::IndexSet;

use cgmath::SquareMatrix;
use cgmath::vec3;
use cgmath::Quaternion;


use anyhow::{anyhow, Context, Result};
use log::*;

use vulkanalia::vk::DeviceV1_0;
use vulkanalia::vk::HasBuilder;
use vulkanalia::vk::Handle;


use crate::common::Vertex;
use crate::common::Material;
use crate::common::Vec2;
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
}


#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ModelClass {
    Static,
    SemiDynamic,
    Dynamic, 
}


pub struct Model {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    material_ids: Vec<u32>,
    materials: Vec<Material>,

    skeleton: Option<Skeleton>,
    animations: Vec<AnimationClip>,
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
    pub transform_matrices: Vec<Mat4>,

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
            transform_matrices: Vec::new(),
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
    ) -> Result<Model> {
        use std::path::Path;

        let source_path = Path::new(path);

        let asset = if let Some(cache_path) = find_valid_cache(source_path)? {
            info!("Loading model from cache: {}", cache_path.display());

            load_cached_asset(&cache_path)?
        } else {
            info!("Cache miss, parsing glTF: {}", path);

            let (
                vertices,
                indices,
                material_ids,
                materials,
                images,
                skeleton,
                animations,
            ) = load_gltf(path)?;

            let textures = images
                .iter()
                .map(convert_image)
                .collect::<Result<Vec<_>>>()?;

            let asset = CachedAsset {
                vertices,
                indices,
                material_ids,
                materials,
                textures,
                skeleton,
                animations,
            };

            let cache_path = cache_path_for(source_path);
            let dependencies = find_gltf_dependencies(source_path)?;

            if let Err(e) = save_cache(&cache_path, &asset, &dependencies) {
                warn!(
                    "Failed to save cache {}: {:?}",
                    cache_path.display(),
                    e
                );
            } else {
                info!("Saved model cache: {}", cache_path.display());
            }

            asset
        };

        // Models without a skin still need the dummy root.
        let skeleton = asset
            .skeleton
            .or_else(|| Some(dummy_root_skeleton()));

        // Upload textures to Vulkan.
        let texture_offset = data.textures.len() as i32;

        let result = create_cached_textures(
            instance,
            device,
            data,
            &asset.textures,
        )?;

        data.textures.extend(result);

        // Texture indices coming from glTF are local to this model.
        // Make them point into the global AppData texture array.
        let mut materials = asset.materials;

        for material in &mut materials {
            if material.albedo_texture_index >= 0 {
                material.albedo_texture_index += texture_offset;
            }
        }

        info!("Loaded {} into memory", path);

        Ok(Model {
            vertices: asset.vertices,
            indices: asset.indices,
            material_ids: asset.material_ids,
            materials,
            skeleton,
            animations: asset.animations,
        })
    }

    pub fn add_model_to_scene(&mut self, model: &Model, model_class: ModelClass, model_name: Option<String>) {
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
            skeleton: model.skeleton.clone(),
            model_class,
        });

        match model_name {
            Some(mn) => self.model_names.insert(mn),
            None => self.model_names.insert(self.model_info.len().to_string()),
        };
        self.animations.push(AnimationPlayer::new(model.animations.clone()));
    }


    pub fn search_for_model_id(&self, model_name: String) -> Option<usize> {
        if let Some(index) = self.model_names.get_index_of(&model_name) {
            Some(index);
        }
        None
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

unsafe fn create_gltf_textures(
    instance: &crate::Instance,
    device: &crate::Device,
    data: &crate::AppData,
    images: &[gltf::image::Data],
) -> Result<Vec<(crate::vk::Image, crate::vk::DeviceMemory, crate::vk::ImageView)>> {
    unsafe {
        let mut textures = Vec::new();

        for img in images {
            let vk_format = gltf_to_vulkan(img.format)
                .ok_or_else(|| anyhow!("Unsupported glTF image format: {:?}", img.format))?;

            let pixels: Vec<u8> = match img.format {
                // 8 bit
                gltf::image::Format::R8 => img.pixels.clone(),

                gltf::image::Format::R8G8B8A8 => img.pixels.clone(),

                gltf::image::Format::R8G8B8 => img.pixels
                    .chunks_exact(3)
                    .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255u8])
                    .collect(),

                // 16 bit
                gltf::image::Format::R16 => img.pixels.clone(),

                gltf::image::Format::R16G16B16A16 => img.pixels.clone(),

                gltf::image::Format::R16G16B16 => {
                    let u16_pixels: &[u16] = bytemuck::cast_slice(&img.pixels);

                    u16_pixels
                        .chunks_exact(3)
                        .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], u16::MAX])
                        .flat_map(|v| v.to_ne_bytes())
                        .collect()
                }

                other => {
                    return Err(anyhow!(
                        "Unsupported glTF image format: {:?}",
                        other
                    ));
                }
            };

            let (image, memory, view) = create_texture_image(
                instance,
                device,
                data,
                data.command_pool,
                data.graphics_queue,
                &pixels,
                img.width,
                img.height,
                vk_format,
            )?;

            textures.push((image, memory, view));
        }

        Ok(textures)
    }
}

fn gltf_to_vulkan(format: gltf::image::Format) -> Option<crate::vk::Format> {
    match format {
        crate::R8 =>
            Some(crate::vk::Format::R8_UNORM),

        crate::R8G8B8 =>
            Some(crate::vk::Format::R8G8B8A8_SRGB),

        crate::R8G8B8A8 =>
            Some(crate::vk::Format::R8G8B8A8_SRGB),

        crate::R16 =>
            Some(crate::vk::Format::R16_UNORM),

        crate::R16G16B16 =>
            Some(crate::vk::Format::R16G16B16A16_UNORM),

        crate::R16G16B16A16 =>
            Some(crate::vk::Format::R16G16B16A16_UNORM),

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

fn dummy_root_skeleton() -> Skeleton {
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


// Model caching
const CACHE_MAGIC: &[u8; 8] = b"JMCACHE\0";
const CACHE_VERSION: u32 = 2;

#[derive(Clone)]
pub struct CachedTexture {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub pixels: Vec<u8>,
}

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

#[derive(Clone, Debug)]
struct CacheDependency {
    path: PathBuf,
    size: u64,
    modified: std::time::SystemTime,
}


fn write_u8(w: &mut impl Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn write_u16(w: &mut impl Write, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_i32(w: &mut impl Write, v: i32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}


fn write_bytes(w: &mut impl Write, data: &[u8]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    w.write_all(data)
}

fn read_bytes(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u64(r)?;

    if len > usize::MAX as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cached allocation too large",
        ));
    }

    let mut data = vec![0u8; len as usize];
    r.read_exact(&mut data)?;
    Ok(data)
}

fn write_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    write_bytes(w, s.as_bytes())
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let data = read_bytes(r)?;
    String::from_utf8(data)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8"))
}

fn write_mat4(w: &mut impl Write, m: &Mat4) -> io::Result<()> {
    for column in 0..4 {
        for row in 0..4 {
            write_f32(w, m[column][row])?;
        }
    }

    Ok(())
}

fn read_mat4(r: &mut impl Read) -> io::Result<Mat4> {
    let mut m = [[0.0f32; 4]; 4];

    for column in 0..4 {
        for row in 0..4 {
            m[column][row] = read_f32(r)?;
        }
    }

    Ok(Mat4::new(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ))
}


fn write_vertex(w: &mut impl Write, v: &Vertex) -> io::Result<()> {
    write_f32(w, v.pos.x)?;
    write_f32(w, v.pos.y)?;
    write_f32(w, v.pos.z)?;
    write_f32(w, v._pad0)?;

    write_f32(w, v.normal.x)?;
    write_f32(w, v.normal.y)?;
    write_f32(w, v.normal.z)?;
    write_f32(w, v._pad1)?;

    write_f32(w, v.uv.x)?;
    write_f32(w, v.uv.y)?;

    for x in v.joint_indices {
        write_u16(w, x)?;
    }

    for x in v.joint_weights {
        write_f32(w, x)?;
    }

    Ok(())
}

fn read_vertex(r: &mut impl Read) -> io::Result<Vertex> {
    let pos = Vec3::new(
        read_f32(r)?,
        read_f32(r)?,
        read_f32(r)?,
    );

    let pad0 = read_f32(r)?;

    let normal = Vec3::new(
        read_f32(r)?,
        read_f32(r)?,
        read_f32(r)?,
    );

    let pad1 = read_f32(r)?;

    let uv = Vec2::new(
        read_f32(r)?,
        read_f32(r)?,
    );

    let mut joint_indices = [0u16; 4];
    for x in &mut joint_indices {
        *x = read_u16(r)?;
    }

    let mut joint_weights = [0.0f32; 4];
    for x in &mut joint_weights {
        *x = read_f32(r)?;
    }

    Ok(Vertex {
        pos,
        _pad0: pad0,
        normal,
        _pad1: pad1,
        uv,
        joint_indices,
        joint_weights,
    })
}


fn write_material(w: &mut impl Write, m: &Material) -> io::Result<()> {
    write_f32(w, m.albedo.x)?;
    write_f32(w, m.albedo.y)?;
    write_f32(w, m.albedo.z)?;
    write_i32(w, m.albedo_texture_index)?;

    write_f32(w, m.metallic)?;
    write_f32(w, m.roughness)?;

    write_f32(w, m.emission.x)?;
    write_f32(w, m.emission.y)?;
    write_f32(w, m.emission.z)?;

    write_f32(w, m.transmission)?;
    write_f32(w, m.ior)?;
    write_f32(w, m.specular)?;
    write_f32(w, m.clearcoat)?;
    write_f32(w, m.clearcoat_roughness)?;

    write_f32(w, m.sheen)?;

    write_f32(w, m.sheen_color.x)?;
    write_f32(w, m.sheen_color.y)?;
    write_f32(w, m.sheen_color.z)?;

    Ok(())
}

fn read_material(r: &mut impl Read) -> io::Result<Material> {
    Ok(Material {
        albedo: Vec3::new(
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
        ),

        albedo_texture_index: read_i32(r)?,

        metallic: read_f32(r)?,
        roughness: read_f32(r)?,

        emission: Vec3::new(
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
        ),

        transmission: read_f32(r)?,
        ior: read_f32(r)?,

        specular: read_f32(r)?,
        clearcoat: read_f32(r)?,
        clearcoat_roughness: read_f32(r)?,

        sheen: read_f32(r)?,

        sheen_color: Vec3::new(
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
        ),
    })
}


fn write_skeleton(w: &mut impl Write, skeleton: &Skeleton) -> io::Result<()> {
    write_u64(w, skeleton.bones.len() as u64)?;

    for bone in &skeleton.bones {
        write_u64(w, bone.node_index as u64)?;

        write_u64(w, bone.children.len() as u64)?;
        for &child in &bone.children {
            write_u64(w, child as u64)?;
        }

        write_mat4(w, &bone.local_transform)?;
        write_mat4(w, &bone.inverse_bind_matrix)?;
    }

    write_u64(w, skeleton.root_bones.len() as u64)?;
    for &root in &skeleton.root_bones {
        write_u64(w, root as u64)?;
    }

    Ok(())
}

fn read_skeleton(r: &mut impl Read) -> io::Result<Skeleton> {
    let bone_count = read_u64(r)? as usize;

    let mut bones = Vec::with_capacity(bone_count);

    for _ in 0..bone_count {
        let node_index = read_u64(r)? as usize;

        let child_count = read_u64(r)? as usize;
        let mut children = Vec::with_capacity(child_count);

        for _ in 0..child_count {
            children.push(read_u64(r)? as usize);
        }

        let local_transform = read_mat4(r)?;
        let inverse_bind_matrix = read_mat4(r)?;

        bones.push(Bone {
            node_index,
            children,
            local_transform,
            inverse_bind_matrix,
        });
    }

    let root_count = read_u64(r)? as usize;
    let mut root_bones = Vec::with_capacity(root_count);

    for _ in 0..root_count {
        root_bones.push(read_u64(r)? as usize);
    }

    Ok(Skeleton {
        bones,
        root_bones,
    })
}


fn write_animation_channel(
    w: &mut impl Write,
    channel: &AnimationChannel,
) -> io::Result<()> {
    write_u64(w, channel.bone_index as u64)?;

    write_u64(w, channel.translations.len() as u64)?;
    for &(time, value) in &channel.translations {
        write_f32(w, time)?;
        write_f32(w, value.x)?;
        write_f32(w, value.y)?;
        write_f32(w, value.z)?;
    }

    write_u64(w, channel.rotations.len() as u64)?;
    for &(time, value) in &channel.rotations {
        write_f32(w, time)?;
        write_f32(w, value.v.x)?;
        write_f32(w, value.v.y)?;
        write_f32(w, value.v.z)?;
        write_f32(w, value.s)?;
    }

    write_u64(w, channel.scales.len() as u64)?;
    for &(time, value) in &channel.scales {
        write_f32(w, time)?;
        write_f32(w, value.x)?;
        write_f32(w, value.y)?;
        write_f32(w, value.z)?;
    }

    Ok(())
}

fn read_animation_channel(
    r: &mut impl Read,
) -> io::Result<AnimationChannel> {
    let bone_index = read_u64(r)? as usize;

    let translation_count = read_u64(r)? as usize;
    let mut translations = Vec::with_capacity(translation_count);

    for _ in 0..translation_count {
        let time = read_f32(r)?;

        let value = Vec3::new(
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
        );

        translations.push((time, value));
    }

    let rotation_count = read_u64(r)? as usize;
    let mut rotations = Vec::with_capacity(rotation_count);

    for _ in 0..rotation_count {
        let time = read_f32(r)?;

        let x = read_f32(r)?;
        let y = read_f32(r)?;
        let z = read_f32(r)?;
        let w = read_f32(r)?;

        rotations.push((
            time,
            Quaternion::new(w, x, y, z),
        ));
    }

    let scale_count = read_u64(r)? as usize;
    let mut scales = Vec::with_capacity(scale_count);

    for _ in 0..scale_count {
        let time = read_f32(r)?;

        let value = Vec3::new(
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
        );

        scales.push((time, value));
    }

    Ok(AnimationChannel {
        bone_index,
        translations,
        rotations,
        scales,
    })
}

fn write_animation_clip(
    w: &mut impl Write,
    clip: &AnimationClip,
) -> io::Result<()> {
    write_string(w, &clip.name)?;
    write_f32(w, clip.duration)?;

    write_u64(w, clip.channels.len() as u64)?;

    for channel in &clip.channels {
        write_animation_channel(w, channel)?;
    }

    Ok(())
}

fn read_animation_clip(
    r: &mut impl Read,
) -> io::Result<AnimationClip> {
    let name = read_string(r)?;
    let duration = read_f32(r)?;

    let channel_count = read_u64(r)? as usize;
    let mut channels = Vec::with_capacity(channel_count);

    for _ in 0..channel_count {
        channels.push(read_animation_channel(r)?);
    }

    Ok(AnimationClip {
        name,
        duration,
        channels,
    })
}


unsafe fn create_cached_textures(
    instance: &crate::Instance,
    device: &crate::Device,
    data: &crate::AppData,
    textures: &[CachedTexture],
) -> Result<Vec<(crate::vk::Image, crate::vk::DeviceMemory, crate::vk::ImageView)>> {
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
}

fn write_texture(
    w: &mut impl Write,
    texture: &CachedTexture,
) -> io::Result<()> {
    write_u32(w, texture.width)?;
    write_u32(w, texture.height)?;
    write_u8(w, texture.format.to_u8())?;
    write_bytes(w, &texture.pixels)?;

    Ok(())
}

fn read_texture(
    r: &mut impl Read,
) -> io::Result<CachedTexture> {
    let width = read_u32(r)?;
    let height = read_u32(r)?;
    let format = TextureFormat::from_u8(read_u8(r)?)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let pixels = read_bytes(r)?;

    Ok(CachedTexture {
        width,
        height,
        format,
        pixels,
    })
}

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


fn find_gltf_dependencies(source: &Path) -> Result<Vec<CacheDependency>> {
    let gltf = gltf::Gltf::open(source)
        .with_context(|| format!("Opening {}", source.display()))?;

    let base_dir = source.parent().unwrap_or_else(|| Path::new("."));

    let mut paths = vec![source.to_path_buf()];

    // External buffers
    for buffer in gltf.document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source() {
           paths.push(normalize_path(base_dir.join(uri)));
        }
    }

    // External images
    for image in gltf.document.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source() {
            paths.push(normalize_path(base_dir.join(uri)));
        }
    }

    // Remove duplicates
    paths.sort();
    paths.dedup();

    let mut dependencies = Vec::with_capacity(paths.len());

    for path in paths {
        let (size, modified) = get_file_info(&path)?;

        dependencies.push(CacheDependency {
            path,
            size,
            modified,
        });
    }

    Ok(dependencies)
}


fn save_cache(
    path: &Path,
    asset: &CachedAsset,
    dependencies: &[CacheDependency],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Creating cache directory {}", parent.display()))?;
    }

    let temp_path = path.with_extension("cache.tmp");

    let mut file = File::create(&temp_path)
        .with_context(|| format!("Creating temporary cache {}", temp_path.display()))?;

    file.write_all(CACHE_MAGIC)?;
    write_u32(&mut file, CACHE_VERSION)?;

    write_u64(&mut file, dependencies.len() as u64)?;

    for dependency in dependencies {
        write_string(
            &mut file,
            &dependency.path.to_string_lossy(),
        )?;

        write_u64(&mut file, dependency.size)?;

        let modified = dependency
            .modified
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        write_u64(&mut file, modified.as_secs())?;
        write_u32(&mut file, modified.subsec_nanos())?;
    }

    file.write_all(CACHE_MAGIC)?;
    write_u32(&mut file, CACHE_VERSION)?;

    write_u64(&mut file, asset.vertices.len() as u64)?;
    for vertex in &asset.vertices {
        write_vertex(&mut file, vertex)?;
    }

    write_u64(&mut file, asset.indices.len() as u64)?;
    for &index in &asset.indices {
        write_u32(&mut file, index)?;
    }

    write_u64(&mut file, asset.material_ids.len() as u64)?;
    for &id in &asset.material_ids {
        write_u32(&mut file, id)?;
    }

    write_u64(&mut file, asset.materials.len() as u64)?;
    for material in &asset.materials {
        write_material(&mut file, material)?;
    }

    match &asset.skeleton {
        Some(skeleton) => {
            write_u8(&mut file, 1)?;
            write_skeleton(&mut file, skeleton)?;
        }

        None => {
            write_u8(&mut file, 0)?;
        }
    }

    write_u64(&mut file, asset.animations.len() as u64)?;
    for animation in &asset.animations {
        write_animation_clip(&mut file, animation)?;
    }

    write_u64(&mut file, asset.textures.len() as u64)?;
    for texture in &asset.textures {
        write_texture(&mut file, texture)?;
    }

    file.flush()?;
    drop(file);

    std::fs::rename(&temp_path, path)
        .with_context(|| format!("Installing cache {}", path.display()))?;

    Ok(())
}

pub fn load_cached_asset(path: &std::path::Path) -> Result<CachedAsset> {
    let mut file = File::open(path)
        .with_context(|| format!("Opening cache {}", path.display()))?;

    // Magic
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;

    if &magic != CACHE_MAGIC {
        return Err(anyhow!("Invalid cache magic"));
    }

    // Version
    let version = read_u32(&mut file)?;

    if version != CACHE_VERSION {
        return Err(anyhow!(
            "Unsupported cache version {} (expected {})",
            version,
            CACHE_VERSION
        ));
    }

    // Dependencies
    let dependency_count = read_u64(&mut file)? as usize;

    for _ in 0..dependency_count {
        let _path = read_string(&mut file)?;
        let _size = read_u64(&mut file)?;
        let _seconds = read_u64(&mut file)?;
        let _nanos = read_u32(&mut file)?;
    }

    // Vertices
    let vertex_count = read_u64(&mut file)? as usize;
    let mut vertices = Vec::with_capacity(vertex_count);

    for _ in 0..vertex_count {
        vertices.push(read_vertex(&mut file)?);
    }

    // Indices
    let index_count = read_u64(&mut file)? as usize;
    let mut indices = Vec::with_capacity(index_count);

    for _ in 0..index_count {
        indices.push(read_u32(&mut file)?);
    }

    // Material Ids
    let material_id_count = read_u64(&mut file)? as usize;
    let mut material_ids = Vec::with_capacity(material_id_count);

    for _ in 0..material_id_count {
        material_ids.push(read_u32(&mut file)?);
    }

    // Materials
    let material_count = read_u64(&mut file)? as usize;
    let mut materials = Vec::with_capacity(material_count);

    for _ in 0..material_count {
        materials.push(read_material(&mut file)?);
    }

    // Skeleton
    let has_skeleton = read_u8(&mut file)?;

    let skeleton = match has_skeleton {
        0 => None,
        1 => Some(read_skeleton(&mut file)?),
        _ => return Err(anyhow!("Invalid skeleton flag {}", has_skeleton)),
    };

    // Animations
    let animation_count = read_u64(&mut file)? as usize;
    let mut animations = Vec::with_capacity(animation_count);

    for _ in 0..animation_count {
        animations.push(read_animation_clip(&mut file)?);
    }

    // Textures
    let texture_count = read_u64(&mut file)? as usize;
    let mut textures = Vec::with_capacity(texture_count);

    for _ in 0..texture_count {
        textures.push(read_texture(&mut file)?);
    }

    Ok(CachedAsset {
        vertices,
        indices,
        material_ids,
        materials,
        textures,
        skeleton,
        animations,
    })
}

fn cache_path(source: &Path) -> PathBuf {
    let mut path = source.to_path_buf();
    path.set_extension("cache");
    path
}

fn cache_path_for(source: &Path) -> PathBuf {
    let parent = source.parent().unwrap_or_else(|| Path::new("."));

    let cache_dir = parent.join("cache");

    let stem = source
        .file_stem()
        .unwrap_or_else(|| std::ffi::OsStr::new("asset"));

    cache_dir.join(format!("{}.cache", stem.to_string_lossy()))
}

fn get_file_info(path: &Path) -> Result<(u64, std::time::SystemTime)> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Reading metadata for {}", path.display()))?;

    let modified = metadata
        .modified()
        .with_context(|| format!("Reading modification time for {}", path.display()))?;

    Ok((metadata.len(), modified))
}

pub fn find_valid_cache(source: &Path) -> Result<Option<PathBuf>> {
    let cache_path = cache_path_for(source);

    let mut file = match File::open(&cache_path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(e) => {
            return Err(e)
                .with_context(|| format!("Opening cache {}", cache_path.display()));
        }
    };

    // Check magic
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;

    if &magic != CACHE_MAGIC {
        warn!(
            "Invalid cache magic: {}",
            cache_path.display()
        );

        return Ok(None);
    }

    // Check version
    let version = read_u32(&mut file)?;

    if version != CACHE_VERSION {
        info!(
            "Cache version mismatch for {}",
            source.display()
        );

        return Ok(None);
    }

    // Read counts
    let dependency_count = read_u64(&mut file)? as usize;

    let mut dependencies = Vec::with_capacity(dependency_count);

    for _ in 0..dependency_count {
        let path = PathBuf::from(read_string(&mut file)?);
        let size = read_u64(&mut file)?;
        let seconds = read_u64(&mut file)?;
        let nanos = read_u32(&mut file)?;

        let modified =
            std::time::UNIX_EPOCH
                + std::time::Duration::new(seconds, nanos);

        dependencies.push(CacheDependency {
            path,
            size,
            modified,
        });
    }

    // Validate dependencies
    for dependency in &dependencies {
        let Ok((size, modified)) =
            get_file_info(&dependency.path)
        else {
            info!(
                "Cache dependency missing: {}",
                dependency.path.display()
            );

            return Ok(None);
        };

        if size != dependency.size ||
           modified != dependency.modified
        {
            info!(
                "Cache dependency changed: {}",
                dependency.path.display()
            );

            return Ok(None);
        }
    }

    Ok(Some(cache_path))
}

fn normalize_path(path: PathBuf) -> PathBuf {
    if let Ok(path) = path.canonicalize() {
        path
    } else {
        path
    }
}

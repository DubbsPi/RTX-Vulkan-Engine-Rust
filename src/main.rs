use anyhow::{anyhow, Result};
use thiserror::Error;
use log::*;

use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::Version;
use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::InstanceV1_1;
use vulkanalia::vk::DeviceV1_2;
use vulkanalia::vk::KhrAccelerationStructureExtensionDeviceCommands;
use vulkanalia::vk::KhrRayTracingPipelineExtensionDeviceCommands;

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::time::{Duration, Instant};
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use winit::window::{Window, WindowBuilder};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;

use cgmath::{Deg, Point3, vec3};
use cgmath::SquareMatrix;
use cgmath::InnerSpace;

mod common;
use common::Vertex;
use common::Material;
use common::Vec3;
use common::Mat4;

mod scene;
use scene::Scene;
use scene::Model;
use scene::ModelClass;


const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_ACCELERATION_STRUCTURE_EXTENSION.name,
    vk::KHR_RAY_TRACING_PIPELINE_EXTENSION.name,
    vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION.name,
    vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name,
    vk::EXT_DESCRIPTOR_INDEXING_EXTENSION.name,
];

const MAX_FRAMES_IN_FLIGHT: usize = 3;  // Don't touch me!!

const MAX_TEXTURES: u32 = 4096;  // Almost free to increase, but to make dynamic is extreamly hard

const BLAS_REBUILD_INTERVAL: u32 = 240;

const ENABLE_CUTOUT_SHADER: bool = true;


unsafe fn create_scene(instance: &Instance, device: &Device, data: &mut AppData,) -> Result<Scene> {
    let mut scene = Scene::new();
    
    unsafe {
        let magazine = Scene::load_model_into_memory(
            &mut scene,
            "models/Magazine.glb",
            &instance, &device, data,
        )?;
        let protogen = Scene::load_model_into_memory(
            &mut scene,
            "models/Xenon.glb",
            &instance, &device, data,
        )?;
        let room = Scene::load_model_into_memory(
            &mut scene,
            "models/Test_Room.glb",
            &instance, &device, data,
        )?;


        scene.add_model_to_scene(&magazine, ModelClass::Static, None);
        scene.add_model_to_scene(&magazine, ModelClass::Static, None);

        scene.scale_model(StringOrInt::Int(1), vec3(5.0, 5.0, 5.0));
        scene.translate_model(StringOrInt::Int(1), vec3(1.0, 0.5, 0.0));

        scene.scale_model(StringOrInt::Int(0), vec3(20.0, 5.0, 10.0));
        scene.translate_model(StringOrInt::Int(0), vec3(3.0, -2.0, 0.0));

        scene.add_model_to_scene(&protogen, ModelClass::Dynamic, Some("Xenon".to_owned()));
        scene.translate_model(StringOrInt::Str("Xenon".to_owned()), vec3(-4.0, 0.0, 0.0));

        scene.add_model_to_scene(&room, ModelClass::Dynamic, Some("Room".to_owned()));
        scene.translate_model(StringOrInt::Str("Room".to_owned()), vec3(0.0, -5.0, 8.0));
    }

    Ok(scene)
}


#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> { unsafe {
        let properties = instance
            .get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);
        
        let mut present = None;
        for (index, _properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self {graphics, present})
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families")))
        }
    }}
}


#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> { unsafe {
        Ok(
            Self {
                capabilities: instance
                    .get_physical_device_surface_capabilities_khr(
                        physical_device, data.surface)?,
                formats: instance
                    .get_physical_device_surface_formats_khr(
                        physical_device, data.surface)?,
                present_modes: instance
                    .get_physical_device_surface_present_modes_khr(
                        physical_device, data.surface)?
            }
        )
    }}
}


struct App {
    _entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,

    frame: usize,
    resized: bool,

    last_frame: Instant,
    frame_time: f64,
    start: Instant,

    camera: Camera,
    input: InputState,
    controls_enabled: bool,

    scene: Scene,
}

impl App {
    unsafe fn create(window: &Window) -> Result<Self> { unsafe {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let _entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();

        let instance = create_instance(window, &_entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;

        pick_physical_device(&instance, &mut data)?;

        let device = create_logical_device(&_entry, &instance, &mut data)?;

        let query_pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * MAX_FRAMES_IN_FLIGHT as u32);
        data.queries_valid = [false; MAX_FRAMES_IN_FLIGHT];

        data.timestamp_query_pool = device.create_query_pool(&query_pool_info, None)?;

        let properties = instance.get_physical_device_properties(data.physical_device);
        data.timestamp_period = properties.limits.timestamp_period;

        create_swapchain(window, &instance, &device, &mut data)?;

        create_command_pool(&instance, &device, &mut data)?;

        create_storage_image(&instance, &device, &mut data)?;
        create_accum_image(&instance, &device, &mut data)?;


        // Scene setup
        let scene = create_scene(&instance, &device, &mut data)?;

        data.material_refcounts = vec![0u32; scene.materials.len()];
        for &mat_id in &scene.material_ids {
            data.material_refcounts[mat_id as usize] += 1;
        }


        data.texture_sampler = create_texture_sampler(&device)?;
        info!("Triangles: {}, Vertices: {}", scene.indices.len() / 3, scene.vertices.len());

        create_descriptor_set_layout(&device, &mut data)?;
        create_rt_pipeline(&device, &mut data)?;
        create_denoise_pipeline(&device, &mut data)?;

        // Create index and vertex buffers
        let vertex_size = (size_of::<Vertex>() * scene.vertices.len()) as u64;
        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            &instance, &device, &data, vertex_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let mem = device.map_memory(vertex_buffer_memory, 0, vertex_size, vk::MemoryMapFlags::empty())?;
        memcpy(scene.vertices.as_ptr().cast::<u8>(), mem.cast::<u8>(), vertex_size as usize);
        device.unmap_memory(vertex_buffer_memory);

        let index_size = (size_of::<u32>() * scene.indices.len()) as u64;
        let (index_buffer, index_buffer_memory) = create_buffer(
            &instance, &device, &data, index_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let mem = device.map_memory(index_buffer_memory, 0, index_size, vk::MemoryMapFlags::empty())?;
        memcpy(scene.indices.as_ptr().cast::<u8>(), mem.cast::<u8>(), index_size as usize);
        device.unmap_memory(index_buffer_memory);

        data.vertex_buffer = vertex_buffer;
        data.vertex_buffer_memory = vertex_buffer_memory;
        data.index_buffer = index_buffer;
        data.index_buffer_memory = index_buffer_memory;


        let vertex_address = get_buffer_device_address(&device, data.vertex_buffer);
        let index_address = get_buffer_device_address(&device, data.index_buffer);
        if vertex_address == 0 || index_address == 0 {
            return Err(anyhow!("Vertex or index buffer has a zero device address"));
        }


        let rigged_model_count = scene.model_info
            .iter()
            .filter(|m| m.skeleton.is_some())
            .count() as u32;

        create_skinning_descriptor_pool(&device, &mut data, rigged_model_count)?;
        create_skinning_pipeline(&device, &mut data)?;
        
        let mut skinned_vertex_buffers = Vec::new();
        let mut skinned_vertex_buffers_memory = Vec::new();
        let mut joint_matrix_buffers = Vec::new();
        let mut joint_matrix_buffers_memory = Vec::new();
        let mut joint_matrix_buffers_mapped: Vec<*mut u8> = Vec::new();
        let mut skinning_descriptor_sets = Vec::new();
        let mut skinned_vertex_counts = Vec::new();
        let mut joint_counts = Vec::new();
        let mut skinned_triangle_counts = Vec::new();

        for model_info in &scene.model_info {
            let tri_count = (model_info.model_index_range.max - model_info.model_index_range.min) / 3;
            skinned_triangle_counts.push(tri_count);
            
            match &model_info.skeleton {
                Some(skeleton) => {
                    let vertex_count = model_info.model_vertex_range.max - model_info.model_vertex_range.min;

                    // Skinned output buffer
                    let skinned_size = (size_of::<Vertex>() as u32 * vertex_count) as u64;
                    let (skinned_buffer, skinned_memory) = create_buffer(
                        &instance, &device, &data, skinned_size,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )?;

                    // Joint matrix buffer
                    let joint_count = skeleton.bones.len();
                    let joint_buffer_size = (size_of::<Mat4>() * joint_count.max(1)) as u64;
                    let (joint_buffer, joint_memory) = create_buffer(
                        &instance, &device, &data, joint_buffer_size,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    )?;
                    let joint_mapped = device.map_memory(joint_memory, 0, joint_buffer_size, vk::MemoryMapFlags::empty())?
                        as *mut u8;

                    
                    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(data.skinning_descriptor_pool)
                        .set_layouts(std::slice::from_ref(&data.skinning_descriptor_set_layout));
                    let descriptor_set = device.allocate_descriptor_sets(&alloc_info)?[0];

                    let rest_info = vk::DescriptorBufferInfo::builder()
                        .buffer(vertex_buffer) // your shared merged vertex buffer, local var from earlier in create()
                        .offset((model_info.model_vertex_range.min as u64) * size_of::<Vertex>() as u64)
                        .range((vertex_count as u64) * size_of::<Vertex>() as u64)
                        .build();
                    let skinned_info = vk::DescriptorBufferInfo::builder()
                        .buffer(skinned_buffer).offset(0).range(vk::WHOLE_SIZE).build();
                    let joints_info = vk::DescriptorBufferInfo::builder()
                        .buffer(joint_buffer).offset(0).range(vk::WHOLE_SIZE).build();

                    let writes = &[
                        vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set).dst_binding(0).dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&rest_info)).build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set).dst_binding(1).dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&skinned_info)).build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set).dst_binding(2).dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&joints_info)).build(),
                    ];
                    device.update_descriptor_sets(writes, &[] as &[vk::CopyDescriptorSet]);

                    // Blank matrices for the default
                    let identity_matrices = vec![Mat4::identity(); joint_count.max(1)];
                    memcpy(identity_matrices.as_ptr().cast::<u8>(), joint_mapped, joint_buffer_size as usize);

                    skinned_vertex_buffers.push(Some(skinned_buffer));
                    skinned_vertex_buffers_memory.push(Some(skinned_memory));
                    joint_matrix_buffers.push(Some(joint_buffer));
                    joint_matrix_buffers_memory.push(Some(joint_memory));
                    joint_matrix_buffers_mapped.push(joint_mapped);
                    skinning_descriptor_sets.push(Some(descriptor_set));
                    skinned_vertex_counts.push(vertex_count);
                    joint_counts.push(skeleton.bones.len());
                }
                None => {
                    // Boneless model
                    skinned_vertex_buffers.push(None);
                    skinned_vertex_buffers_memory.push(None);
                    joint_matrix_buffers.push(None);
                    joint_matrix_buffers_memory.push(None);
                    joint_matrix_buffers_mapped.push(std::ptr::null_mut());
                    skinning_descriptor_sets.push(None);
                    skinned_vertex_counts.push(0);
                    joint_counts.push(0);
                }
            }
        }

        data.skinned_vertex_buffers = skinned_vertex_buffers;
        data.skinned_vertex_buffers_memory = skinned_vertex_buffers_memory;
        data.joint_matrix_buffers = joint_matrix_buffers;
        data.joint_matrix_buffers_memory = joint_matrix_buffers_memory;
        data.joint_matrix_buffers_mapped = joint_matrix_buffers_mapped;
        data.skinning_descriptor_sets = skinning_descriptor_sets;
        data.skinned_vertex_counts = skinned_vertex_counts;
        data.joint_counts = joint_counts;
        data.skinned_triangle_counts = skinned_triangle_counts;


        let mut skinned_index_buffers = Vec::new();
        let mut skinned_index_buffers_memory = Vec::new();
        let mut skinned_index_addresses = Vec::new();
        for model_info in &scene.model_info {
            if model_info.skeleton.is_some() {
                let indices: Vec<u32> = scene.indices[model_info.model_index_range.min as usize..model_info.model_index_range.max as usize]
                    .iter()
                    .map(|&index| index - model_info.model_vertex_range.min)
                    .collect();
                let index_size = (size_of::<u32>() * indices.len()) as u64;
                let (index_buffer, index_memory) = create_buffer(
                    &instance, &device, &data, index_size,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )?;
                let mapped = device.map_memory(index_memory, 0, index_size, vk::MemoryMapFlags::empty())?;
                memcpy(indices.as_ptr().cast::<u8>(), mapped.cast::<u8>(), index_size as usize);
                device.unmap_memory(index_memory);

                skinned_index_addresses.push(Some(get_buffer_device_address(&device, index_buffer)));
                skinned_index_buffers.push(Some(index_buffer));
                skinned_index_buffers_memory.push(Some(index_memory));
            } else {
                skinned_index_addresses.push(None);
                skinned_index_buffers.push(None);
                skinned_index_buffers_memory.push(None);
            }
        }
        data.skinned_index_buffers = skinned_index_buffers;
        data.skinned_index_buffers_memory = skinned_index_buffers_memory;
        data.skinned_index_addresses = skinned_index_addresses;


        // Create blases
        info!("model_info.len()={}, skinning_descriptor_sets.len()={}", scene.model_info.len(), data.skinning_descriptor_sets.len());
        prime_skin_all(&device, &data, &scene)?;

        let mut static_members = Vec::new();
        let mut semi_members = Vec::new();
        let mut dynamic_members = Vec::new();
        for (i, m) in scene.model_info.iter().enumerate() {
            match m.model_class {
                ModelClass::Static => static_members.push(i),
                ModelClass::SemiDynamic => semi_members.push(i),
                ModelClass::Dynamic => dynamic_members.push(i),
            }
        }

        let static_base = 0u32;
        let semi_base = static_members.len() as u32;
        let dynamic_base = semi_base + semi_members.len() as u32;

        data.static_group = rebuild_group_cold(&instance, &device, &data, static_members, static_base)?;
        data.semi_group = rebuild_group_cold(&instance, &device, &data, semi_members, semi_base)?;
        data.dynamic_group = rebuild_group_cold(&instance, &device, &data, dynamic_members, dynamic_base)?;
        data.dynamic_group_frames_since_rebuild = 0;

        create_tlas(&instance, &device, &mut data)?;

        create_scene_buffers(
            &instance, &device, &mut data,
            &scene.materials, &scene.material_ids,
            &scene.model_info,
        )?;
        
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;


        let mut object_descs = Vec::with_capacity(scene.model_info.len());
        for group in [&data.static_group, &data.semi_group, &data.dynamic_group] {
            for &mi in &group.members {
                let model = &scene.model_info[mi];
                let src = geom_source_for(&device, &data, mi);

                object_descs.push(ObjectDesc {
                    vertex_address: src.vertex_address,
                    index_address: src.index_address,
                    material_id: model.model_index_range.min / 3,
                    _pad: 0,
                });
            }
        }


        let object_descs_size = (size_of::<ObjectDesc>() * object_descs.len()) as u64;
        let mapped = device.map_memory(data.object_descs_buffer_memory, 0, object_descs_size, vk::MemoryMapFlags::empty())?;
        memcpy(object_descs.as_ptr().cast::<u8>(), mapped.cast::<u8>(), object_descs_size as usize);
        device.unmap_memory(data.object_descs_buffer_memory);


        create_shader_binding_table(&instance, &device, &mut data)?;

        create_command_buffers(&device, &mut data)?;

        create_sync_objects(&device, &mut data)?;
        
        let frame = 0;
        let resized = false;
        let last_frame = Instant::now();
        let frame_time = 0.0;

        let camera = Camera::new(Point3::new(0.0, 1.0, 0.0));
        let input = InputState::default();
        let controls_enabled = true;
        let start = Instant::now();

        Ok(Self {
            _entry, instance: instance, data, device,
            frame, resized,
            last_frame, frame_time, start,
            camera, input, controls_enabled,
            scene
        })
    }}

    unsafe fn destroy(&mut self) { unsafe {
        self.destroy_swapchain();

        // SBT
        self.device.destroy_buffer(self.data.sbt_buffer, None);
        self.device.free_memory(self.data.sbt_buffer_memory, None);

        // Tlas
        self.device.destroy_acceleration_structure_khr(self.data.tlas, None);
        self.device.destroy_buffer(self.data.tlas_buffer, None);
        self.device.free_memory(self.data.tlas_buffer_memory, None);
        
        self.device.destroy_buffer(self.data.tlas_scratch_buffer, None);
        self.device.free_memory(self.data.tlas_scratch_buffer_memory, None);
        
        self.device.destroy_buffer(self.data.instance_buffer, None);
        self.device.free_memory(self.data.instance_buffer_memory, None);

        // Blases
        for group in [&self.data.static_group, &self.data.semi_group, &self.data.dynamic_group] {
            if group.blas != vk::AccelerationStructureKHR::null() {
                self.device.destroy_acceleration_structure_khr(group.blas, None);
            }
            if group.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(group.buffer, None);
                self.device.free_memory(group.buffer_memory, None);
            }
            if group.scratch_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(group.scratch_buffer, None);
                self.device.free_memory(group.scratch_buffer_memory, None);
            }
        }

        // Geometry buffers
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);

        // Material buffers
        self.device.destroy_buffer(self.data.materials_buffer, None);
        self.device.free_memory(self.data.materials_buffer_memory, None);
        self.device.destroy_buffer(self.data.object_descs_buffer, None);
        self.device.free_memory(self.data.object_descs_buffer_memory, None);

        self.device.destroy_buffer(self.data.material_ids_buffer, None);
        self.device.free_memory(self.data.material_ids_buffer_memory, None);

        // Textures
        self.device.destroy_sampler(self.data.texture_sampler, None);

        for (image, memory, view) in self.data.textures.drain(..) {
            self.device.destroy_image_view(view, None);
            self.device.destroy_image(image, None);
            self.device.free_memory(memory, None);
        }

        // Rigging
        for i in 0..self.data.skinned_vertex_buffers.len() {
            if let Some(buffer) = self.data.skinned_vertex_buffers[i] {
                self.device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.data.skinned_vertex_buffers_memory[i] {
                self.device.free_memory(memory, None);
            }

            if let Some(buffer) = self.data.joint_matrix_buffers[i] {
                self.device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.data.joint_matrix_buffers_memory[i] {
                // unmap before freeing — it was left persistently mapped
                self.device.unmap_memory(memory);
                self.device.free_memory(memory, None);
            }

            if let Some(buffer) = self.data.skinned_index_buffers[i] {
                self.device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.data.skinned_index_buffers_memory[i] {
                self.device.free_memory(memory, None);
            }
        }

        // Skinning
        self.device.destroy_pipeline(self.data.skinning_pipeline, None);
        self.device.destroy_pipeline_layout(self.data.skinning_pipeline_layout, None);
        self.device.destroy_descriptor_set_layout(self.data.skinning_descriptor_set_layout, None);
        self.device.destroy_descriptor_pool(self.data.skinning_descriptor_pool, None);

        // Pools and descriptor sets
        self.device.destroy_query_pool(self.data.timestamp_query_pool, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);


        self.data.in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }}

    unsafe fn destroy_swapchain(&mut self) { unsafe {
        for i in 0..self.data.storage_images.len() {
            self.device.destroy_image_view(self.data.storage_image_views[i], None);
            self.device.destroy_image(self.data.storage_images[i], None);
            self.device.free_memory(self.data.storage_image_memories[i], None);
        }

        self.device.destroy_image_view(self.data.accum_image_view, None);
        self.device.destroy_image(self.data.accum_image, None);
        self.device.free_memory(self.data.accum_image_memory, None);
        

        self.device.destroy_pipeline(self.data.denoise_pipeline, None);
        self.device.destroy_pipeline_layout(self.data.denoise_pipeline_layout, None);

        self.device.destroy_pipeline(self.data.rt_pipeline, None);
        self.device.destroy_pipeline_layout(self.data.rt_pipeline_layout, None);


        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);

        for buffer in self.data.uniform_buffers.drain(..) {
            self.device.destroy_buffer(buffer, None);
        }
        for memory in self.data.uniform_buffers_memory.drain(..) {
            self.device.unmap_memory(memory);
            self.device.free_memory(memory, None);
        }

        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);

        for view in self.data.swapchain_image_views.drain(..) {
            self.device.destroy_image_view(view, None);
        }

        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }}


    unsafe fn render(&mut self, window: &Window) -> Result<()> { unsafe {
        self.device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame]],
            true,
            u64::MAX,
        )?;

        if self.data.queries_valid[self.frame] {
            let mut timestamps = [0u64; 2];
            let query_result = self.device.get_query_pool_results(
                self.data.timestamp_query_pool,
                (self.frame as u32) * 2,
                2,
                std::slice::from_raw_parts_mut(
                    timestamps.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of::<u64>() * 2,
                ),
                std::mem::size_of::<u64>() as u64,
                vk::QueryResultFlags::_64,
            );

            if query_result.is_ok() {
                let ticks = timestamps[1].saturating_sub(timestamps[0]);
                let gpu_ms = (ticks as f64 * self.data.timestamp_period as f64) / 1_000_000.0;
                self.frame_time = gpu_ms;
            }
        }


        // Check for scene ops
        if !self.data.pending_model_ops.is_empty() {
            self.device.device_wait_idle()?;
            self.apply_pending_model_ops()?;
        }


        let image_index = match self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores[self.frame],
                vk::Fence::null(),
            ) {
                Ok((index, _)) => index as usize,
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window)?;
                    return Ok(());
                }
                Err(e) => return Err(anyhow!(e)),
            };

        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] =
            self.data.in_flight_fences[self.frame];
        

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;


        self.update_camera(dt);
        self.update_uniform_buffer(image_index)?;

        // Rerecord command buffer
        let cmd = self.data.command_buffers[image_index];
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

        let begin_info = vk::CommandBufferBeginInfo::builder();
        self.device.begin_command_buffer(cmd, &begin_info)?;

        self.update_dynamic_models(cmd, dt);


        self.device.cmd_reset_query_pool(
            cmd,
            self.data.timestamp_query_pool,
            (image_index as u32) * 2,
            2,
        );

        self.device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            self.data.timestamp_query_pool,
            (image_index as u32) * 2,
        );

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            self.data.rt_pipeline,
        );
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            self.data.rt_pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );


        let current_view = self.camera.view_matrix();
        let camera_moved = match self.data.last_view_matrix {
            Some(last) => {
                let diff: f32 = (0..4).flat_map(|c| (0..4).map(move |r| (c, r)))
                    .map(|(c, r)| (current_view[c][r] - last[c][r]).abs())
                    .sum();
                diff > 0.0001
            }
            None => true,
        };

        self.data.frame_index = self.data.frame_index.wrapping_add(1);
        self.data.accumulated_samples = if camera_moved { 0 } else { self.data.accumulated_samples + 1 };
        self.data.last_view_matrix = Some(current_view);

        let push_constants = RtPushConstants {
            frame_index: self.data.frame_index,
            accumulated_samples: self.data.accumulated_samples,
        };

        self.device.cmd_push_constants(
            cmd,
            self.data.rt_pipeline_layout,
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            std::slice::from_raw_parts(
                (&push_constants as *const RtPushConstants).cast::<u8>(),
                size_of::<RtPushConstants>(),
            ),
        );

        self.device.cmd_trace_rays_khr(
            cmd,
            &self.data.sbt_raygen_region,
            &self.data.sbt_miss_region,
            &self.data.sbt_hit_region,
            &self.data.sbt_callable_region,
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
            1,
        );

        let denoise_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.data.accum_image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build()
            )
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[denoise_barrier],
        );

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.data.denoise_pipeline,
        );

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.data.denoise_pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );

        let group_x = (self.data.swapchain_extent.width + 7) / 8;
        let group_y = (self.data.swapchain_extent.height + 7) / 8;

        self.device.cmd_dispatch(cmd, group_x, group_y, 1);

        let storage_image_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.data.storage_images[image_index])
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

        let swapchain_image_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.data.swapchain_images[image_index])
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[storage_image_barrier, swapchain_image_barrier],
        );

        let blit_region = vk::ImageBlit::builder()
            .src_subresource(vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .src_offsets([
                vk::Offset3D {x: 0, y: 0, z: 0},
                vk::Offset3D {
                    x: self.data.swapchain_extent.width as i32,
                    y: self.data.swapchain_extent.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .dst_offsets([
                vk::Offset3D {x: 0, y: 0, z: 0},
                vk::Offset3D {
                    x: self.data.swapchain_extent.width as i32,
                    y: self.data.swapchain_extent.height as i32,
                    z: 1,
                },
            ]);

        self.device.cmd_blit_image(
            cmd,
            self.data.storage_images[image_index],
            vk::ImageLayout::GENERAL,
            self.data.swapchain_images[image_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit_region],
            vk::Filter::NEAREST,
        );

        let present_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.data.swapchain_images[image_index])
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[present_barrier],
        );

        self.device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.data.timestamp_query_pool,
            (image_index as u32) * 2 + 1,
        );

        self.device.end_command_buffer(cmd)?;

        // Submit
        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::TRANSFER];

        let command_buffers = &[cmd];
        let signal_semaphores = &[self.data.render_finished_semaphores[image_index]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        self.data.queries_valid[self.frame] = true;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);

        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }}


    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> { unsafe {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        info!("Recreating swapchain...");

        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;

        create_storage_image(&self.instance, &self.device, &mut self.data)?;
        create_accum_image(&self.instance, &self.device, &mut self.data)?;

        create_rt_pipeline(&self.device, &mut self.data)?;
        create_denoise_pipeline(&self.device, &mut self.data)?;

        create_shader_binding_table(&self.instance, &self.device, &mut self.data)?;
        
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;

        create_command_buffers(&self.device, &mut self.data)?;

        self.data.accumulated_samples = 0;
        self.data.last_view_matrix = None;

        self.data.queries_valid = [false; MAX_FRAMES_IN_FLIGHT];
        self.data.images_in_flight = vec![vk::Fence::null(); self.data.swapchain_images.len()];

        Ok(())
    }}


    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> { unsafe {
        let time = self.start.elapsed().as_secs_f32();
        let view = self.camera.view_matrix();
        let mut proj = cgmath::perspective(Deg(45.0), self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32, 0.1, 100.0);
        proj[1][1] *= -1.0;

        let ubo = CameraUniformBufferObject {
            view_inverse: view.invert().ok_or_else(|| anyhow!("Failed to invert view matrix"))?,
            proj_inverse: proj.invert().ok_or_else(|| anyhow!("Failed to invert proj matrix"))?,
            time,
        };

        memcpy(
            (&ubo as *const CameraUniformBufferObject).cast::<u8>(),
            self.data.uniform_buffers_mapped[image_index],
            size_of::<CameraUniformBufferObject>(),
        );
        Ok(())
    }}

    fn update_camera(&mut self, dt: f32) {
        if self.controls_enabled {
            self.camera.yaw += self.input.mouse_dx * self.camera.sensitivity;
            self.camera.pitch -= self.input.mouse_dy * self.camera.sensitivity;
            self.camera.pitch = self.camera.pitch.clamp(-1.55, 1.55);

            self.input.mouse_dx = 0.0;
            self.input.mouse_dy = 0.0;

            let forward = self.camera.forward();
            let right = self.camera.right();
            let mut delta = vec3(0.0, 0.0, 0.0);

            if self.input.forward { delta += forward; }
            if self.input.back    { delta -= forward; }
            if self.input.right   { delta += right; }
            if self.input.left    { delta -= right; }
            if self.input.up      { delta.y += 1.0; }
            if self.input.down    { delta.y -= 1.0; }

            if delta.magnitude() > 0.0001 {
                delta = delta.normalize() * self.camera.speed * dt;
                self.camera.position += delta;
            }
        }
    }


    unsafe fn update_dynamic_models(&mut self, cmd: vk::CommandBuffer, dt: f32) { unsafe {
        if self.data.dynamic_group.members.is_empty() { return; }

        let members = self.data.dynamic_group.members.clone(); // cheap, small list; sidesteps borrow issues
        for model_index in members {
            self.dispatch_skin(cmd, model_index, dt);
        }

        let compute_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            );
        self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(), &[compute_barrier], &[] as &[vk::BufferMemoryBarrier], &[] as &[vk::ImageMemoryBarrier]);

        let due_for_rebuild = self.data.dynamic_group_frames_since_rebuild >= BLAS_REBUILD_INTERVAL;
        let mode = if due_for_rebuild { vk::BuildAccelerationStructureModeKHR::BUILD } else { vk::BuildAccelerationStructureModeKHR::UPDATE };
        
        self.refit_group(cmd, GroupKind::Dynamic, mode);
        self.data.dynamic_group_frames_since_rebuild = if due_for_rebuild { 0 } else { self.data.dynamic_group_frames_since_rebuild + 1 };

        let as_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(), &[as_barrier], &[] as &[vk::BufferMemoryBarrier], &[] as &[vk::ImageMemoryBarrier]);
    }}

    unsafe fn dispatch_skin(&mut self, cmd: vk::CommandBuffer, model_index: usize, dt: f32) { unsafe {
        let Some(descriptor_set) = self.data.skinning_descriptor_sets[model_index] else { return; };
        let vertex_count = self.data.skinned_vertex_counts[model_index];

        if let Some(mapped) = self.data.joint_matrix_buffers_mapped.get(model_index).copied() {
            if !mapped.is_null() {
                if let Some(skeleton) = &self.scene.model_info[model_index].skeleton {
                    let world = self.scene.transform_matrices[model_index];
                    let joint_matrices: Vec<Mat4> = if skeleton.bones.len() == 1 && self.scene.animations[model_index].clips.is_empty() {
                        vec![world]
                    } else {
                        self.scene.animations[model_index].advance(dt);
                        self.scene.animations[model_index].sample(skeleton)
                            .into_iter()
                            .map(|m| world * m)
                            .collect()
                    };

                    let size = size_of::<Mat4>() * joint_matrices.len();
                    std::ptr::copy_nonoverlapping(joint_matrices.as_ptr().cast::<u8>(), mapped, size);
                }
            }
        }

        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.data.skinning_pipeline);
        self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.data.skinning_pipeline_layout, 0, &[descriptor_set], &[]);
        self.device.cmd_push_constants(cmd, self.data.skinning_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &vertex_count.to_ne_bytes());
        self.device.cmd_dispatch(cmd, (vertex_count + 127) / 128, 1, 1);
    }}

    unsafe fn apply_pending_model_ops(&mut self) -> Result<()> { unsafe {
        let pending_ops = std::mem::take(&mut self.data.pending_model_ops);

        let mut added_model_index = None;

        for op in pending_ops {
            match op {
                ModelOp::Add(model, model_class, transform, model_name) => {
                    let model_index = self.scene.model_info.len();

                    self.scene.add_model_to_scene(&model, model_class, model_name);
                    self.scene.set_transform(StringOrInt::Int(model_index), transform);
                    self.recompute_material_refcounts();

                    added_model_index = Some(model_index);
                }

                ModelOp::Remove(index) => {
                    free_skin_resources_for_model(
                        &self.device,
                        &mut self.data,
                        index
                    );
                    self.remove_model(index)?;
                }
            }
        }

        // Grow buffers
        let needed_verts = self.scene.vertices.len() as u64;
        let needed_indices = self.scene.indices.len() as u64;
        let needed_models = self.scene.model_info.len();

        let cap = &self.data.geometry_capacity;
        let must_grow = needed_verts > cap.vertex_capacity
            || needed_indices > cap.index_capacity
            || needed_models > cap.model_capacity;

        if must_grow {
            self.rebuild_geometry_buffers(needed_verts, needed_indices, needed_models)?;
            self.refresh_skinning_rest_bindings()?;
        } else {
            self.reupload_scene_data()?;
        }

        // Grow skinning buffer
        let rigged_count = self.scene.model_info.iter().filter(|m| m.skeleton.is_some()).count() as u32;
        if rigged_count > self.data.skinning_pool_capacity {
            let new_capacity = (rigged_count * 3 / 2).max(rigged_count);
            self.device.destroy_descriptor_pool(self.data.skinning_descriptor_pool, None);
            create_skinning_descriptor_pool(&self.device, &mut self.data, new_capacity)?;
            self.reallocate_all_skinning_descriptor_sets()?; // this already reads current self.data.vertex_buffer, safe now
        }

        if let Some(model_index) = added_model_index {
            let vertex_buffer = self.data.vertex_buffer;

            allocate_skin_resources_for_model(
                &self.instance,
                &self.device,
                &mut self.data,
                vertex_buffer,
                &self.scene.model_info[model_index],
                &self.scene.indices,
            )?;
        }

        let rigged_count = self.scene.model_info.iter().filter(|m| m.skeleton.is_some()).count() as u32;
        self.data.skinning_pool_capacity = self.data.skinning_pool_capacity.max(rigged_count);

        self.rebuild_scene_side_buffers()?;
        prime_skin_all(&self.device, &self.data, &self.scene)?;
        self.rebuild_groups_and_tlas()?;
        self.recreate_descriptor_sets_for_geometry()?;

        Ok(())
    }}

    unsafe fn refresh_skinning_rest_bindings(&mut self) -> Result<()> { unsafe {
        for i in 0..self.data.skinning_descriptor_sets.len() {
            let Some(descriptor_set) = self.data.skinning_descriptor_sets[i] else { continue; };
            let model_info = &self.scene.model_info[i];
            let vertex_count = self.data.skinned_vertex_counts[i];

            let rest_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.data.vertex_buffer)
                .offset((model_info.model_vertex_range.min as u64) * size_of::<Vertex>() as u64)
                .range((vertex_count as u64) * size_of::<Vertex>() as u64)
                .build();

            let write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set).dst_binding(0).dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&rest_info)).build();

            self.device.update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
        }
        Ok(())
    }}


    fn queue_add_model(&mut self, model: &Model, model_class: ModelClass, transform: Mat4, model_name: Option<String>) {
        self.data.pending_model_ops.push(ModelOp::Add(model.into(), model_class, transform, model_name));
        info!("Queued adding model {} to scene", self.scene.model_info.len() + 1)
    }

    fn queue_remove_model(&mut self, model_id: StringOrInt) {
        let model_index = match model_id {
            StringOrInt::Str(s) => {
                if let Some(i) = self.scene.search_for_model_id(s) {
                    i
                } else {
                    return
                }
            },
            StringOrInt::Int(i) => i,
        };
        
        self.data.pending_model_ops.push(ModelOp::Remove(model_index));
        info!("Queued removing model {} to scene", model_index)
    }


    unsafe fn rebuild_groups_and_tlas(&mut self) -> Result<()> { unsafe {
        if self.data.tlas != vk::AccelerationStructureKHR::null() {
            self.device.destroy_acceleration_structure_khr(
                self.data.tlas,
                None,
            );
        }

        if self.data.tlas_buffer != vk::Buffer::null() {
            self.device.destroy_buffer(
                self.data.tlas_buffer,
                None,
            );
            self.device.free_memory(
                self.data.tlas_buffer_memory,
                None,
            );
        }

        if self.data.tlas_scratch_buffer != vk::Buffer::null() {
            self.device.destroy_buffer(
                self.data.tlas_scratch_buffer,
                None,
            );
            self.device.free_memory(
                self.data.tlas_scratch_buffer_memory,
                None,
            );
        }

        if self.data.instance_buffer != vk::Buffer::null() {
            self.device.destroy_buffer(
                self.data.instance_buffer,
                None,
            );
            self.device.free_memory(
                self.data.instance_buffer_memory,
                None,
            );
        }

        let mut static_members = Vec::new();
        let mut semi_members = Vec::new();
        let mut dynamic_members = Vec::new();
        for (i, m) in self.scene.model_info.iter().enumerate() {
            match m.model_class {
                ModelClass::Static => static_members.push(i),
                ModelClass::SemiDynamic => semi_members.push(i),
                ModelClass::Dynamic => dynamic_members.push(i),
            }
        }

        // Destroy old groups
        for group in [&self.data.static_group, &self.data.semi_group, &self.data.dynamic_group] {
            if group.blas != vk::AccelerationStructureKHR::null() {
                self.device.destroy_acceleration_structure_khr(group.blas, None);
                self.device.destroy_buffer(group.buffer, None);
                self.device.free_memory(group.buffer_memory, None);
                self.device.destroy_buffer(group.scratch_buffer, None);
                self.device.free_memory(group.scratch_buffer_memory, None);
            }
        }

        let static_base = 0u32;
        let semi_base = static_members.len() as u32;
        let dynamic_base = semi_base + semi_members.len() as u32;

        self.data.static_group = rebuild_group_cold(&self.instance, &self.device, &self.data, static_members, static_base)?;
        self.data.semi_group = rebuild_group_cold(&self.instance, &self.device, &self.data, semi_members, semi_base)?;
        self.data.dynamic_group = rebuild_group_cold(&self.instance, &self.device, &self.data, dynamic_members, dynamic_base)?;
        self.data.dynamic_group_frames_since_rebuild = 0;

        create_tlas(&self.instance, &self.device, &mut self.data)?;

        Ok(())
    }}

    unsafe fn rebuild_geometry_buffers(&mut self, needed_verts: u64, needed_indices: u64, needed_models: usize) -> Result<()> { unsafe {
        let new_vertex_cap = (needed_verts * 3 / 2).max(64);
        let new_index_cap = (needed_indices * 3 / 2).max(64);
        let new_model_cap = (needed_models * 3 / 2).max(4);

        // Destroy old vertex and index buffers
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);

        let vertex_size = new_vertex_cap * size_of::<Vertex>() as u64;
        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            &self.instance, &self.device, &self.data, vertex_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let index_size = new_index_cap * size_of::<u32>() as u64;
        let (index_buffer, index_buffer_memory) = create_buffer(
            &self.instance, &self.device, &self.data, index_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        self.data.vertex_buffer = vertex_buffer;
        self.data.vertex_buffer_memory = vertex_buffer_memory;
        self.data.index_buffer = index_buffer;
        self.data.index_buffer_memory = index_buffer_memory;
        self.data.geometry_capacity.vertex_capacity = new_vertex_cap;
        self.data.geometry_capacity.index_capacity = new_index_cap;
        self.data.geometry_capacity.model_capacity = new_model_cap;

        self.reupload_scene_data()?;

        Ok(())
    }}

    unsafe fn reupload_scene_data(&mut self) -> Result<()> { unsafe {
        let vertex_bytes = (size_of::<Vertex>() * self.scene.vertices.len()) as u64;
        let mem = self.device.map_memory(self.data.vertex_buffer_memory, 0, vertex_bytes, vk::MemoryMapFlags::empty())?;
        memcpy(self.scene.vertices.as_ptr().cast::<u8>(), mem.cast::<u8>(), vertex_bytes as usize);
        self.device.unmap_memory(self.data.vertex_buffer_memory);

        let index_bytes = (size_of::<u32>() * self.scene.indices.len()) as u64;
        let mem = self.device.map_memory(self.data.index_buffer_memory, 0, index_bytes, vk::MemoryMapFlags::empty())?;
        memcpy(self.scene.indices.as_ptr().cast::<u8>(), mem.cast::<u8>(), index_bytes as usize);
        self.device.unmap_memory(self.data.index_buffer_memory);

        Ok(())
    }}

    fn remove_model(&mut self, model_index: usize) -> Result<()> {
        let removed = self.scene.model_info[model_index].clone();
        let vert_span = removed.model_vertex_range.max - removed.model_vertex_range.min;
        let index_span = removed.model_index_range.max - removed.model_index_range.min;
        
        let tri_start = (removed.model_index_range.min / 3) as usize;
        let tri_end = (removed.model_index_range.max / 3) as usize;
        let removed_material_ids = self.scene.material_ids[tri_start..tri_end].to_vec();

        // Cut data out
        self.scene.vertices.drain(
            removed.model_vertex_range.min as usize..removed.model_vertex_range.max as usize
        );
        self.scene.indices.drain(
            removed.model_index_range.min as usize..removed.model_index_range.max as usize
        );

        // Shift everything else down
        for index in self.scene.indices.iter_mut() {
            if *index >= removed.model_vertex_range.max {
                *index -= vert_span;
            }
        }

        self.scene.material_ids.drain(
            (removed.model_index_range.min / 3) as usize..(removed.model_index_range.max / 3) as usize
        );

        // Shift models down
        for model in self.scene.model_info.iter_mut() {
            if model.model_vertex_range.min > removed.model_vertex_range.min {
                model.model_vertex_range.min -= vert_span;
                model.model_vertex_range.max -= vert_span;
            }
            if model.model_index_range.min > removed.model_index_range.min {
                model.model_index_range.min -= index_span;
                model.model_index_range.max -= index_span;
            }
        }

        self.scene.model_info.remove(model_index);
        self.scene.transform_matrices.remove(model_index);

        self.decrement_materials(&removed_material_ids)?;

        Ok(())
    }

    fn decrement_materials(&mut self, removed_material_ids: &Vec<u32>) -> Result<()> {
        let used_material_ids: HashSet<u32> = removed_material_ids.iter().copied().collect();

        for &mat_id in &used_material_ids {
            self.data.material_refcounts[mat_id as usize] -= 1;
        }

        // Find materials with none in use
        let mut to_remove: Vec<u32> = self.data.material_refcounts.iter()
            .enumerate()
            .filter(|(_, count)| **count == 0)
            .map(|(i, _)| i as u32)
            .collect();
        to_remove.sort_unstable();

        for &mat_index in to_remove.iter().rev() {
            self.scene.materials.remove(mat_index as usize);
            self.data.material_refcounts.remove(mat_index as usize);

            for id in self.scene.material_ids.iter_mut() {
                if *id > mat_index {
                    *id -= 1;
                }
            }
        }

        Ok(())
    }

    unsafe fn move_semi_dynamic_model(&mut self, model_id: StringOrInt, transform: Mat4) -> Result<()> { unsafe {
        let model_index = match model_id {
            StringOrInt::Str(s) => {
                if let Some(i) = self.scene.search_for_model_id(s) {
                    i
                } else {
                    0
                }
            },
            StringOrInt::Int(i) => i,
        };
        
        debug_assert_eq!(self.scene.model_info[model_index].model_class, ModelClass::SemiDynamic);
        self.scene.set_transform(StringOrInt::Int(model_index), transform);

        if let Some(mapped) = self.data.joint_matrix_buffers_mapped.get(model_index).copied() {
            if !mapped.is_null() {
                memcpy((&transform as *const Mat4).cast::<u8>(), mapped, size_of::<Mat4>());
            }
        }

        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY).command_pool(self.data.command_pool).command_buffer_count(1);
        let cmd = self.device.allocate_command_buffers(&alloc_info)?[0];
        self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

        self.dispatch_skin(cmd, model_index, 0.0);

        let compute_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            );
        self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(), &[compute_barrier], &[] as &[vk::BufferMemoryBarrier], &[] as &[vk::ImageMemoryBarrier]);

        self.refit_group(cmd, GroupKind::SemiDynamic, vk::BuildAccelerationStructureModeKHR::BUILD);

        let as_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(), &[as_barrier], &[] as &[vk::BufferMemoryBarrier], &[] as &[vk::ImageMemoryBarrier]);

        self.device.end_command_buffer(cmd)?;
        let cmds = [cmd];
        self.device.queue_submit(self.data.graphics_queue, &[vk::SubmitInfo::builder().command_buffers(&cmds)], vk::Fence::null())?;
        self.device.queue_wait_idle(self.data.graphics_queue)?;
        self.device.free_command_buffers(self.data.command_pool, &[cmd]);

        // Rebuild tlas
        create_tlas(&self.instance, &self.device, &mut self.data)?;
        
        // Update descriptor sets to point to the new TLAS
        self.recreate_descriptor_sets_for_geometry()?;

        Ok(())
    }}


    unsafe fn refit_group(&mut self, cmd: vk::CommandBuffer, kind: GroupKind, mode: vk::BuildAccelerationStructureModeKHR) { unsafe {
        let group = self.group_ref(kind);
        if group.members.is_empty() { return; }

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
            .mode(mode)
            .geometries(&group.cached_geometries)
            .dst_acceleration_structure(group.blas)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: group.scratch_addr });

        if mode == vk::BuildAccelerationStructureModeKHR::UPDATE {
            build_info = build_info.src_acceleration_structure(group.blas);
        }

        let ranges_slice = group.cached_range_infos.as_slice();
        let range_refs = std::slice::from_ref(&ranges_slice);

        self.device.cmd_build_acceleration_structures_khr(cmd, &[build_info], range_refs);
    }}

    fn group_ref(&self, kind: GroupKind) -> &ClassGroup {
        match kind {
            GroupKind::Static => &self.data.static_group,
            GroupKind::SemiDynamic => &self.data.semi_group,
            GroupKind::Dynamic => &self.data.dynamic_group,
        }
    }

    unsafe fn rebuild_scene_side_buffers(&mut self) -> Result<()> { unsafe {
        self.device.destroy_buffer(self.data.materials_buffer, None);
        self.device.free_memory(self.data.materials_buffer_memory, None);
        self.device.destroy_buffer(self.data.object_descs_buffer, None);
        self.device.free_memory(self.data.object_descs_buffer_memory, None);
        self.device.destroy_buffer(self.data.material_ids_buffer, None);
        self.device.free_memory(self.data.material_ids_buffer_memory, None);

        create_scene_buffers(
            &self.instance, &self.device, &mut self.data,
            &self.scene.materials, &self.scene.material_ids,
            &self.scene.model_info,
        )?;

        Ok(())
    }}

    unsafe fn recreate_descriptor_sets_for_geometry(&mut self) -> Result<()> { unsafe {
        let texture_infos: Vec<vk::DescriptorImageInfo> = self.data.textures
            .iter()
            .map(|(_, _, view)| {
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(*view)
                    .sampler(self.data.texture_sampler)
                    .build()
            })
            .collect();

            
        for i in 0..self.data.swapchain_images.len() {
            let mut writes = Vec::new();

            let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                .acceleration_structures(std::slice::from_ref(&self.data.tlas))
                .build();
            let mut as_write = vk::WriteDescriptorSet::builder()
                .push_next(&mut as_write_info)
                .dst_set(self.data.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .build();
            as_write.descriptor_count = 1;
            writes.push(as_write);

            let object_descs_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.data.object_descs_buffer).offset(0).range(vk::WHOLE_SIZE).build();
            writes.push(vk::WriteDescriptorSet::builder()
                .dst_set(self.data.descriptor_sets[i]).dst_binding(3).dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&object_descs_info)).build());

            let materials_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.data.materials_buffer).offset(0).range(vk::WHOLE_SIZE).build();
            writes.push(vk::WriteDescriptorSet::builder()
                .dst_set(self.data.descriptor_sets[i]).dst_binding(4).dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&materials_info)).build());

            let material_ids_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.data.material_ids_buffer).offset(0).range(vk::WHOLE_SIZE).build();
            writes.push(vk::WriteDescriptorSet::builder()
                .dst_set(self.data.descriptor_sets[i]).dst_binding(5).dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&material_ids_info)).build());

            
            if !texture_infos.is_empty() {
                writes.push(vk::WriteDescriptorSet::builder()
                    .dst_set(self.data.descriptor_sets[i])
                    .dst_binding(7)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&texture_infos)
                    .build());
            }

            self.device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        Ok(())
    }}

    unsafe fn reallocate_all_skinning_descriptor_sets(&mut self) -> Result<()> { unsafe {
        for i in 0..self.data.skinning_descriptor_sets.len() {
            if self.data.skinning_descriptor_sets[i].is_none() {
                continue;
            }

            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.data.skinning_descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.data.skinning_descriptor_set_layout));
            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let model_info = &self.scene.model_info[i];
            let vertex_count = self.data.skinned_vertex_counts[i];
            let skinned_buffer = self.data.skinned_vertex_buffers[i].unwrap();
            let joint_buffer = self.data.joint_matrix_buffers[i].unwrap();

            let rest_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.data.vertex_buffer)
                .offset((model_info.model_vertex_range.min as u64) * size_of::<Vertex>() as u64)
                .range((vertex_count as u64) * size_of::<Vertex>() as u64)
                .build();
            let skinned_info = vk::DescriptorBufferInfo::builder()
                .buffer(skinned_buffer).offset(0).range(vk::WHOLE_SIZE).build();
            let joints_info = vk::DescriptorBufferInfo::builder()
                .buffer(joint_buffer).offset(0).range(vk::WHOLE_SIZE).build();

            let writes = &[
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set).dst_binding(0).dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&rest_info)).build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set).dst_binding(1).dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&skinned_info)).build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set).dst_binding(2).dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&joints_info)).build(),
            ];
            self.device.update_descriptor_sets(writes, &[] as &[vk::CopyDescriptorSet]);

            self.data.skinning_descriptor_sets[i] = Some(descriptor_set);
        }

        Ok(())
    }}

    fn recompute_material_refcounts(&mut self) {
        if self.scene.materials.is_empty() {
            self.data.material_refcounts.clear();
            return;
        }
        self.data.material_refcounts = vec![0u32; self.scene.materials.len()];
        for &mat_id in &self.scene.material_ids {
            self.data.material_refcounts[mat_id as usize] += 1;
        }
    }
}


#[derive(Default)]
struct AppData {
    surface: vk::SurfaceKHR,
    messenger: vk::DebugUtilsMessengerEXT,
    
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // Swapchain
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,

    uniform_buffers_mapped: Vec<*mut u8>,

    // Storage images
    storage_images: Vec<vk::Image>,
    storage_image_memories: Vec<vk::DeviceMemory>,
    storage_image_views: Vec<vk::ImageView>,

    // TA images
    accum_image: vk::Image,
    accum_image_memory: vk::DeviceMemory,
    accum_image_view: vk::ImageView,

    accumulated_samples: u32,
    frame_index: u32,
    last_view_matrix: Option<Mat4>,

    // Denoiser
    denoise_pipeline: vk::Pipeline,
    denoise_pipeline_layout: vk::PipelineLayout,

    // Geometry
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    // Resizing
    geometry_capacity: GeometryCapacity,
    pending_model_ops: Vec<ModelOp>,

    // Materials
    materials_buffer: vk::Buffer,
    materials_buffer_memory: vk::DeviceMemory,
    object_descs_buffer: vk::Buffer,
    object_descs_buffer_memory: vk::DeviceMemory,
    material_ids_buffer: vk::Buffer,
    material_ids_buffer_memory: vk::DeviceMemory,
    material_refcounts: Vec<u32>,

    // Textures
    textures: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
    texture_sampler: vk::Sampler,

    // Rigging
    skinned_vertex_buffers: Vec<Option<vk::Buffer>>,
    skinned_vertex_buffers_memory: Vec<Option<vk::DeviceMemory>>,
    joint_matrix_buffers: Vec<Option<vk::Buffer>>,
    joint_matrix_buffers_memory: Vec<Option<vk::DeviceMemory>>,
    joint_matrix_buffers_mapped: Vec<*mut u8>,
    skinning_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    skinned_vertex_counts: Vec<u32>,
    skinned_index_buffers: Vec<Option<vk::Buffer>>,
    skinned_index_buffers_memory: Vec<Option<vk::DeviceMemory>>,
    skinned_index_addresses: Vec<Option<vk::DeviceAddress>>,

    // Skinning
    skinning_descriptor_pool: vk::DescriptorPool,
    skinned_triangle_counts: Vec<u32>,
    skinning_pool_capacity: u32,

    // Skinning
    skinning_pipeline: vk::Pipeline,
    skinning_pipeline_layout: vk::PipelineLayout,
    skinning_descriptor_set_layout: vk::DescriptorSetLayout,

    // Blas
    static_group: ClassGroup,
    semi_group: ClassGroup,
    dynamic_group: ClassGroup,
    dynamic_group_frames_since_rebuild: u32,


    // Tlas
    tlas: vk::AccelerationStructureKHR,
    tlas_buffer: vk::Buffer,
    tlas_buffer_memory: vk::DeviceMemory,
    tlas_scratch_buffer: vk::Buffer,
    tlas_scratch_buffer_memory: vk::DeviceMemory,
    tlas_scratch_addr: vk::DeviceAddress,
    instance_buffer: vk::Buffer,
    instance_buffer_memory: vk::DeviceMemory,
    instance_buffer_mapped: *mut u8,
    instance_buffer_addr: vk::DeviceAddress,
    instance_count: u32,
    joint_counts: Vec<usize>,
    

    // Descriptors
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Pipeline
    rt_pipeline: vk::Pipeline,
    rt_pipeline_layout: vk::PipelineLayout,

    // SBT
    sbt_buffer: vk::Buffer,
    sbt_buffer_memory: vk::DeviceMemory,
    sbt_raygen_region: vk::StridedDeviceAddressRegionKHR,
    sbt_miss_region: vk::StridedDeviceAddressRegionKHR,
    sbt_hit_region: vk::StridedDeviceAddressRegionKHR,
    sbt_callable_region: vk::StridedDeviceAddressRegionKHR,

    // Camera UBO
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    // Commands and sync objects
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,

    // Frametime polling
    timestamp_query_pool: vk::QueryPool,
    timestamp_period: f32,
    queries_valid: [bool; MAX_FRAMES_IN_FLIGHT],
}


#[derive(Debug, Error)]
#[error("Missing {0}")]
pub struct SuitabilityError(pub &'static str);


#[derive(Clone, Copy, PartialEq, Eq)]
enum GroupKind {
    #[allow(dead_code)]
    Static,
    #[allow(dead_code)]
    SemiDynamic,
    #[allow(dead_code)]
    Dynamic
}

#[derive(Default)]
struct ClassGroup {
    blas: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    scratch_buffer: vk::Buffer,
    scratch_buffer_memory: vk::DeviceMemory,
    scratch_addr: vk::DeviceAddress,
    members: Vec<usize>,
    instance_custom_base: u32,
    cached_geometries: Vec<vk::AccelerationStructureGeometryKHR>,
    cached_range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}

struct GeomSource {
    vertex_address: vk::DeviceAddress,
    index_address: vk::DeviceAddress,
    max_vertex: u32,
    triangle_count: u32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ObjectDesc {
    vertex_address: u64,
    index_address: u64,
    material_id: u32,
    _pad: u32,
}


struct Camera {
    position: cgmath::Point3<f32>,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl Camera {
    fn new(position: Point3<f32>) -> Self {
        Self {
            position: position,
            yaw: 90.0_f32.to_radians(),
            pitch: 0.0,
            speed: 2.5,
            sensitivity: 0.0025,
        }
    }

    fn forward(&self) -> Vec3 {
        vec3(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(vec3(0.0, 1.0, 0.0)).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), vec3(0.0, 1.0, 0.0))
    }
}


#[derive(Default)]
struct InputState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_dx: f32,
    mouse_dy: f32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct CameraUniformBufferObject {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    time: f32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct RtAccelerationStructureInstance {
    transform: vk::TransformMatrixKHR,
    instance_custom_index_and_mask: u32,
    instance_sbt_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}


struct StagingBuffer<'a> {
    device: &'a Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl<'a> Drop for StagingBuffer<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}


#[derive(Default)]
struct GeometryCapacity {
    vertex_capacity: u64,
    index_capacity: u64,
    model_capacity: usize,
}

enum ModelOp {
    Add(Model, ModelClass, Mat4, Option<String>),
    Remove(usize),
}


enum StringOrInt {
    #[allow(dead_code)]
    Str(String),
    #[allow(dead_code)]
    Int(usize),
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct RtPushConstants {
    frame_index: u32,
    accumulated_samples: u32,
}


// Utility functions
fn pack_custom_index_and_mask(custom_index: u32, mask: u8) -> u32 {
    (custom_index & 0x00FF_FFFF) | ((mask as u32) << 24)
}

fn pack_sbt_offset_and_flags(sbt_offset: u32, flags: vk::GeometryInstanceFlagsKHR) -> u32 {
    (sbt_offset & 0x00FF_FFFF) | ((flags.bits() as u32) << 24)
}

unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData
) -> Result<Instance> { unsafe {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial\0")
        .application_version(vk::make_version(1, 3, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 3, 0))
        .api_version(vk::make_version(1, 3, 0));

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Required on macOS
    let flags = if 
        cfg!(target_os = "macos") && 
        entry.version()? >= PORTABILITY_MACOS_VERSION
    {
        info!("Enabling extensions for macOS");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };


    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported"));
    }


    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }
    

    extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        type_: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> vk::Bool32 {
        let data = unsafe { *data };
        let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            error!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            warn!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            debug!("({:?}) {}", type_, message);
        } else {
            trace!("({:?}) {}", type_, message);
        }

        vk::FALSE
    }
    

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .user_callback(Some(debug_callback));

        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}}

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &AppData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> { unsafe {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);

    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}}

fn mat4_to_vk_transform(m: &Mat4) -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
        ],
    }
}


// Device checking
unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> { unsafe {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    check_physical_device_rt_features(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> { unsafe {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required device extensions")))
    }
}}

unsafe fn check_physical_device_rt_features(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> { unsafe {
    let mut rt_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder();
    let mut accel_struct_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder();
    let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::builder();
    let mut scalar_block_layout_features = vk::PhysicalDeviceScalarBlockLayoutFeatures::builder();
    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder();

    let mut features2 = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut rt_pipeline_features)
        .push_next(&mut accel_struct_features)
        .push_next(&mut bda_features)
        .push_next(&mut scalar_block_layout_features)
        .push_next(&mut descriptor_indexing_features);

    instance.get_physical_device_features2(physical_device, &mut features2);

    if rt_pipeline_features.ray_tracing_pipeline != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Missing ray tracing pipeline feature")));
    }
    if accel_struct_features.acceleration_structure != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Missing acceleration structure feature")));
    }
    if scalar_block_layout_features.scalar_block_layout != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Missing scalarBlockLayout feature")));
    }
    if bda_features.buffer_device_address != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Missing buffer device address feature")));
    }

    Ok(())
}}


// Device functions
unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> { unsafe {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
        } else {
            info!("Selected physical device (`{}`)", properties.device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device"))
}}

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> { unsafe {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required for macOS
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Feature enabling
    let mut rt_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder()
        .ray_tracing_pipeline(true);

    let mut accel_struct_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
        .acceleration_structure(true);

    let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
        .buffer_device_address(true);

    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::builder()
        .dynamic_rendering(true);
    
    let mut scalar_block_layout_features = vk::PhysicalDeviceScalarBlockLayoutFeatures::builder()
        .scalar_block_layout(true);
    
    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
        .shader_sampled_image_array_non_uniform_indexing(true)
        .runtime_descriptor_array(true);

    let mut int16_features = vk::PhysicalDevice16BitStorageFeatures::builder()
        .storage_buffer_16bit_access(true);

    let base_features = vk::PhysicalDeviceFeatures::builder()
        .shader_int64(true)
        .shader_int16(true);
    
    let mut features2 = vk::PhysicalDeviceFeatures2::builder()
        .features(base_features)
        .push_next(&mut rt_pipeline_features)
        .push_next(&mut accel_struct_features)
        .push_next(&mut bda_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut scalar_block_layout_features)
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut int16_features);

    instance.get_physical_device_features2(data.physical_device, &mut features2);

    if features2.features.shader_int64 != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Missing shaderInt64 feature")));
    }

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .push_next(&mut features2);

    let device = instance.create_device(data.physical_device, &info, None)?;

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}}


// Swapchain utils
fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    info!("Using B8G8R8A8_SRGB swapchain format");
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(
    _present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    vk::PresentModeKHR::MAILBOX
}

fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}


unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}}


unsafe fn create_shader_module(
    device: &Device,
    bytecode: &[u8],
) -> Result<vk::ShaderModule> { unsafe {
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());

    Ok(device.create_shader_module(&info, None)?)
}}


// Scene grouping functions
unsafe fn geom_source_for(device: &Device, data: &AppData, model_index: usize) -> GeomSource { unsafe {
    let vbuf = data.skinned_vertex_buffers[model_index]
        .expect("Everything has an output due to having a root bone");
    GeomSource {
        vertex_address: get_buffer_device_address(device, vbuf),
        index_address: data.skinned_index_addresses[model_index]
            .expect("every model has skinned indices"),
        max_vertex: data.skinned_vertex_counts[model_index] - 1,
        triangle_count: data.skinned_triangle_counts[model_index],
    }
}}

fn triangles_geometry(s: &GeomSource) -> vk::AccelerationStructureGeometryKHR {
    let triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {device_address: s.vertex_address})
        .vertex_stride(size_of::<Vertex>() as u64)
        .max_vertex(s.max_vertex)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {device_address: s.index_address})
        .build();
    vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {triangles: triangles_data})
        .flags(vk::GeometryFlagsKHR::empty())
        .build()
}

unsafe fn prime_skin_all(device: &Device, data: &AppData, scene: &Scene) -> Result<()> { unsafe {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY).command_pool(data.command_pool).command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc_info)?[0];
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

    for i in 0..scene.model_info.len() {
        let Some(descriptor_set) = data.skinning_descriptor_sets[i] else { continue; };
        let vertex_count = data.skinned_vertex_counts[i];

        if let Some(skeleton) = &scene.model_info[i].skeleton {
            if let Some(mapped) = data.joint_matrix_buffers_mapped.get(i).copied() {
                if !mapped.is_null() {
                    let world = scene.transform_matrices[i];
                    let joint_matrices: Vec<Mat4> = if skeleton.bones.len() == 1 {
                        vec![world]
                    } else {
                        // rest pose, transformed by world — no animation advance on prime
                        scene.animations[i].sample(skeleton)
                            .into_iter()
                            .map(|m| world * m)
                            .collect()
                    };
                    let size = size_of::<Mat4>() * joint_matrices.len();
                    memcpy(joint_matrices.as_ptr().cast::<u8>(), mapped, size);
                }
            }
        }

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, data.skinning_pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, data.skinning_pipeline_layout, 0, &[descriptor_set], &[]);
        device.cmd_push_constants(cmd, data.skinning_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &vertex_count.to_ne_bytes());
        device.cmd_dispatch(cmd, (vertex_count + 127) / 128, 1, 1);
    }

    let barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR);
    device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::DependencyFlags::empty(), &[barrier], &[] as &[vk::BufferMemoryBarrier], &[] as &[vk::ImageMemoryBarrier]);

    device.end_command_buffer(cmd)?;
    let cmds = [cmd];
    device.queue_submit(data.graphics_queue, &[vk::SubmitInfo::builder().command_buffers(&cmds)], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;
    device.free_command_buffers(data.command_pool, &[cmd]);
    Ok(())
}}

unsafe fn allocate_skin_resources_for_model(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    vertex_buffer: vk::Buffer,
    model_info: &scene::ModelInfo,
    global_indices: &Vec<u32>,
) -> Result<()> {
    let tri_count = (model_info.model_index_range.max - model_info.model_index_range.min) / 3;
    data.skinned_triangle_counts.push(tri_count);
            
    match &model_info.skeleton {
        Some(skeleton) => {
            unsafe {
                let vertex_count = model_info.model_vertex_range.max - model_info.model_vertex_range.min;

                // Skinned output buffer
                let skinned_size = (size_of::<Vertex>() as u32 * vertex_count) as u64;
                let (skinned_buffer, skinned_memory) = create_buffer(
                    &instance, &device, &data, skinned_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;

                // Joint matrix buffer
                let joint_count = skeleton.bones.len();
                let joint_buffer_size = (size_of::<Mat4>() * joint_count.max(1)) as u64;
                let (joint_buffer, joint_memory) = create_buffer(
                    &instance, &device, &data, joint_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )?;
                let joint_mapped = device.map_memory(joint_memory, 0, joint_buffer_size, vk::MemoryMapFlags::empty())?
                    as *mut u8;

                        
                let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(data.skinning_descriptor_pool)
                    .set_layouts(std::slice::from_ref(&data.skinning_descriptor_set_layout));
                let descriptor_set = device.allocate_descriptor_sets(&alloc_info)?[0];

                let rest_info = vk::DescriptorBufferInfo::builder()
                    .buffer(vertex_buffer) // your shared merged vertex buffer, local var from earlier in create()
                    .offset((model_info.model_vertex_range.min as u64) * size_of::<Vertex>() as u64)
                    .range((vertex_count as u64) * size_of::<Vertex>() as u64)
                    .build();
                let skinned_info = vk::DescriptorBufferInfo::builder()
                    .buffer(skinned_buffer).offset(0).range(vk::WHOLE_SIZE).build();
                let joints_info = vk::DescriptorBufferInfo::builder()
                    .buffer(joint_buffer).offset(0).range(vk::WHOLE_SIZE).build();

                let writes = &[
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set).dst_binding(0).dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&rest_info)).build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set).dst_binding(1).dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&skinned_info)).build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set).dst_binding(2).dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&joints_info)).build(),
                ];
                device.update_descriptor_sets(writes, &[] as &[vk::CopyDescriptorSet]);

                // Blank matrices for the default
                let identity_matrices = vec![Mat4::identity(); joint_count.max(1)];
                memcpy(identity_matrices.as_ptr().cast::<u8>(), joint_mapped, joint_buffer_size as usize);

                data.skinned_vertex_buffers.push(Some(skinned_buffer));
                data.skinned_vertex_buffers_memory.push(Some(skinned_memory));
                data.joint_matrix_buffers.push(Some(joint_buffer));
                data.joint_matrix_buffers_memory.push(Some(joint_memory));
                data.joint_matrix_buffers_mapped.push(joint_mapped);
                data.skinning_descriptor_sets.push(Some(descriptor_set));
                data.skinned_vertex_counts.push(vertex_count);
                data.joint_counts.push(skeleton.bones.len());
            }
        }
        None => {
            // Boneless model
            data.skinned_vertex_buffers.push(None);
            data.skinned_vertex_buffers_memory.push(None);
            data.joint_matrix_buffers.push(None);
            data.joint_matrix_buffers_memory.push(None);
            data.joint_matrix_buffers_mapped.push(std::ptr::null_mut());
            data.skinning_descriptor_sets.push(None);
            data.skinned_vertex_counts.push(0);
            data.joint_counts.push(0);
        }
    }

    if model_info.skeleton.is_some() {
        let local_indices: Vec<u32> = global_indices[model_info.model_index_range.min as usize..model_info.model_index_range.max as usize]
            .iter()
            .map(|&index| index - model_info.model_vertex_range.min)
            .collect();

        unsafe {
            let index_size = (size_of::<u32>() * local_indices.len()) as u64;
            let (index_buffer, index_memory) = create_buffer(
                instance, device, data, index_size,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;
            let mapped = device.map_memory(index_memory, 0, index_size, vk::MemoryMapFlags::empty())?;
            memcpy(local_indices.as_ptr().cast::<u8>(), mapped.cast::<u8>(), index_size as usize);
            device.unmap_memory(index_memory);

            data.skinned_index_addresses.push(Some(get_buffer_device_address(device, index_buffer)));
            data.skinned_index_buffers.push(Some(index_buffer));
            data.skinned_index_buffers_memory.push(Some(index_memory));
        }
    } else {
        data.skinned_index_addresses.push(None);
        data.skinned_index_buffers.push(None);
        data.skinned_index_buffers_memory.push(None);
    }

    Ok(())
}

unsafe fn free_skin_resources_for_model(device: &Device, data: &mut AppData, model_index: usize) { unsafe {
    if let Some(buffer) = data.skinned_vertex_buffers[model_index] {
        device.destroy_buffer(buffer, None);
    }
    if let Some(memory) = data.skinned_vertex_buffers_memory[model_index] {
        device.free_memory(memory, None);
    }

    if let Some(buffer) = data.joint_matrix_buffers[model_index] {
        device.destroy_buffer(buffer, None);
    }
    if let Some(memory) = data.joint_matrix_buffers_memory[model_index] {        device.unmap_memory(memory);
        device.free_memory(memory, None);
    }

    if let Some(buffer) = data.skinned_index_buffers[model_index] {
        device.destroy_buffer(buffer, None);
    }
    if let Some(memory) = data.skinned_index_buffers_memory[model_index] {
        device.free_memory(memory, None);
    }

    if let Some(descriptor_set) = data.skinning_descriptor_sets[model_index] {
        let _ = device.free_descriptor_sets(data.skinning_descriptor_pool, &[descriptor_set]);
    }

    data.skinned_vertex_buffers.remove(model_index);
    data.skinned_vertex_buffers_memory.remove(model_index);
    data.joint_matrix_buffers.remove(model_index);
    data.joint_matrix_buffers_memory.remove(model_index);
    data.joint_matrix_buffers_mapped.remove(model_index);
    data.skinning_descriptor_sets.remove(model_index);
    data.skinned_vertex_counts.remove(model_index);
    data.skinned_index_buffers.remove(model_index);
    data.skinned_index_buffers_memory.remove(model_index);
    data.skinned_index_addresses.remove(model_index);
    data.skinned_triangle_counts.remove(model_index);
    if model_index < data.joint_counts.len() {
        data.joint_counts.remove(model_index);
    }
}}


// Command buffer functions
unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}}

unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.swapchain_images.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    Ok(())
}}


// Image functions
unsafe fn create_storage_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    data.storage_images.clear();
    data.storage_image_memories.clear();
    data.storage_image_views.clear();

    for _ in 0..data.swapchain_images.len() {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: data.swapchain_extent.width,
                height: data.swapchain_extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = device.create_image(&info, None)?;
        let requirements = device.get_image_memory_requirements(image);

        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(
                instance, data, vk::MemoryPropertyFlags::DEVICE_LOCAL, requirements,
            )?);

        let memory = device.allocate_memory(&memory_info, None)?;
        device.bind_image_memory(image, memory, 0)?;

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .subresource_range(subresource_range);

        let view = device.create_image_view(&view_info, None)?;

        data.storage_images.push(image);
        data.storage_image_memories.push(memory);
        data.storage_image_views.push(view);
    }

    // Transfer all images to the general layout
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let cmd = device.allocate_command_buffers(&alloc_info)?[0];
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

    for &image in &data.storage_images {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1)
                .build())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE);

        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );
    }

    device.end_command_buffer(cmd)?;
    device.queue_submit(data.graphics_queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd])], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;
    device.free_command_buffers(data.command_pool, &[cmd]);

    Ok(())
}}

unsafe fn create_accum_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .extent(vk::Extent3D {
            width: data.swapchain_extent.width,
            height: data.swapchain_extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            requirements,
        )?);

    let memory = device.allocate_memory(&memory_info, None)?;
    device.bind_image_memory(image, memory, 0)?;

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .subresource_range(subresource_range);

    let view = device.create_image_view(&view_info, None)?;

    data.accum_image = image;
    data.accum_image_memory = memory;
    data.accum_image_view = view;


    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let cmd = device.allocate_command_buffers(&alloc_info)?[0];

    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1)
                .build(),
        )
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE);

    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    device.end_command_buffer(cmd)?;

    device.queue_submit(
        data.graphics_queue,
        &[vk::SubmitInfo::builder().command_buffers(&[cmd])],
        vk::Fence::null(),
    )?;

    device.queue_wait_idle(data.graphics_queue)?;

    device.free_command_buffers(data.command_pool, &[cmd]);

    Ok(())
}}

// Texture functions
unsafe fn create_texture_sampler(device: &Device) -> Result<vk::Sampler> { unsafe {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    Ok(device.create_sampler(&info, None)?)
}}


// Sync functions
unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data.swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}}


// Buffer utils
unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> { unsafe {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);

    let mut flags_info = vk::MemoryAllocateFlagsInfo::builder()
        .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

    let needs_device_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

    let mut memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    if needs_device_address {
        memory_info = memory_info.push_next(&mut flags_info);
    }

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}}

unsafe fn get_buffer_device_address(
    device: &Device,
    buffer: vk::Buffer,
) -> vk::DeviceAddress { unsafe {
    let info = vk::BufferDeviceAddressInfo::builder()
        .buffer(buffer);

    device.get_buffer_device_address(&info)
}}

unsafe fn create_uniform_buffers(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();
    data.uniform_buffers_mapped.clear();

    for _ in 0..data.swapchain_images.len() {
        let (buf, mem) = create_buffer(
            instance, device, data,
            size_of::<CameraUniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let mapped = device.map_memory(mem, 0, size_of::<CameraUniformBufferObject>() as u64, vk::MemoryMapFlags::empty())? as *mut u8;

        data.uniform_buffers.push(buf);
        data.uniform_buffers_memory.push(mem);
        data.uniform_buffers_mapped.push(mapped);
    }
    Ok(())
}}

unsafe fn create_scene_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    materials: &[Material],
    prim_material_ids: &[u32],
    model_info: &[scene::ModelInfo],
) -> Result<()> { unsafe {
    // Material buffer
    let materials_size = (size_of::<Material>() * materials.len().max(1)).max(1) as u64;
    let (materials_buffer, materials_buffer_memory) = create_buffer(
        instance, device, data, materials_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    
    if !materials.is_empty() {
        let mem = device.map_memory(materials_buffer_memory, 0, materials_size, vk::MemoryMapFlags::empty())?;
        memcpy(materials.as_ptr().cast::<u8>(), mem.cast::<u8>(), materials_size as usize);
        device.unmap_memory(materials_buffer_memory);
    }

    // ObjectDesc buffer
    let object_descs: Vec<ObjectDesc> = {
        let mut descs = Vec::with_capacity(model_info.len());

        let mut push_group = |members: &[usize]| {
            for &mi in members {
                let model = &model_info[mi];
                let src = geom_source_for(device, data, mi);

                descs.push(ObjectDesc {
                    vertex_address: src.vertex_address,
                    index_address: src.index_address,
                    material_id: model.model_index_range.min / 3,
                    _pad: 0,
                });
            }
        };

        let mut static_members = Vec::new();
        let mut semi_members = Vec::new();
        let mut dynamic_members = Vec::new();

        for (i, model) in model_info.iter().enumerate() {
            match model.model_class {
                ModelClass::Static => static_members.push(i),
                ModelClass::SemiDynamic => semi_members.push(i),
                ModelClass::Dynamic => dynamic_members.push(i),
            }
        }

        push_group(&static_members);
        push_group(&semi_members);
        push_group(&dynamic_members);

        descs
    };

    let objects_size = (size_of::<ObjectDesc>() * object_descs.len()).max(1) as u64;
    let (objects_buffer, objects_buffer_memory) = create_buffer(
        instance, device, data, objects_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    let mem = device.map_memory(objects_buffer_memory, 0, objects_size, vk::MemoryMapFlags::empty())?;
    memcpy(object_descs.as_ptr().cast::<u8>(), mem.cast::<u8>(), objects_size as usize);
    device.unmap_memory(objects_buffer_memory);

    // Material Ids buffer
    let ids_size = (size_of::<u32>() * prim_material_ids.len()).max(1) as u64;
    let (ids_buffer, ids_buffer_memory) = create_buffer(
        instance, device, data, ids_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let mem = device.map_memory(ids_buffer_memory, 0, ids_size, vk::MemoryMapFlags::empty())?;
    memcpy(prim_material_ids.as_ptr().cast::<u8>(), mem.cast::<u8>(), ids_size as usize);
    device.unmap_memory(ids_buffer_memory);

    // Store
    data.materials_buffer = materials_buffer;
    data.materials_buffer_memory = materials_buffer_memory;
    data.object_descs_buffer = objects_buffer;
    data.object_descs_buffer_memory = objects_buffer_memory;
    data.material_ids_buffer = ids_buffer;
    data.material_ids_buffer_memory = ids_buffer_memory;

    Ok(())
}}


// Acceleration struct creation
unsafe fn rebuild_group_cold(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    members: Vec<usize>,
    instance_custom_base: u32,
) -> Result<ClassGroup> { unsafe {
    if members.is_empty() {
        return Ok(ClassGroup {members, instance_custom_base, ..Default::default()});
    }

    let sources: Vec<GeomSource> = members.iter().map(|&mi| geom_source_for(device, data, mi)).collect();
    let geometries: Vec<_> = sources.iter().map(triangles_geometry).collect();
    let triangle_counts: Vec<u32> = sources.iter().map(|s| s.triangle_count).collect();
    let range_infos: Vec<_> = sources.iter().map(|s| {
        vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(s.triangle_count).primitive_offset(0).first_vertex(0).transform_offset(0)
            .build()
    }).collect();

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    device.get_acceleration_structure_build_sizes_khr(
        vk::AccelerationStructureBuildTypeKHR::DEVICE, &build_info, &triangle_counts, &mut size_info,
    );

    let (as_buffer, as_buffer_memory) = create_buffer(
        instance, device, data, size_info.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(as_buffer).size(size_info.acceleration_structure_size)
        .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let blas = device.create_acceleration_structure_khr(&create_info, None)?;

    let scratch_alignment = 256u64;
    let scratch_needed = size_info.build_scratch_size.max(size_info.update_scratch_size);
    let (scratch_buffer, scratch_buffer_memory) = create_buffer(
        instance, device, data, scratch_needed + scratch_alignment,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let raw_scratch = get_buffer_device_address(device, scratch_buffer);
    let scratch_addr = (raw_scratch + scratch_alignment - 1) & !(scratch_alignment - 1);

    let build_info = build_info
        .dst_acceleration_structure(blas)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
    let range_refs: Vec<&[_]> = vec![&range_infos[..]];

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY).command_pool(data.command_pool).command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc_info)?[0];
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
    device.cmd_build_acceleration_structures_khr(cmd, &[build_info], &range_refs);
    device.end_command_buffer(cmd)?;

    let cmds = [cmd];
    device.queue_submit(data.graphics_queue, &[vk::SubmitInfo::builder().command_buffers(&cmds)], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;
    device.free_command_buffers(data.command_pool, &[cmd]);

    Ok(ClassGroup {
        blas, buffer: as_buffer, buffer_memory: as_buffer_memory,
        scratch_buffer, scratch_buffer_memory, scratch_addr, members,
        instance_custom_base,
        cached_geometries: geometries,
        cached_range_infos: range_infos,
    })
}}

unsafe fn create_tlas(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let mut instances = Vec::with_capacity(3);
    for group in [&data.static_group, &data.semi_group, &data.dynamic_group] {
        if group.members.is_empty() { continue; }
        let addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::builder().acceleration_structure(group.blas);
        let blas_address = device.get_acceleration_structure_device_address_khr(&addr_info);
        instances.push(RtAccelerationStructureInstance {
            transform: mat4_to_vk_transform(&Mat4::identity()),
            instance_custom_index_and_mask: pack_custom_index_and_mask(group.instance_custom_base, 0xFF),
            instance_sbt_record_offset_and_flags: pack_sbt_offset_and_flags(0, vk::GeometryInstanceFlagsKHR::empty()),
            acceleration_structure_reference: blas_address,
        });
    }

    let instance_count = instances.len() as u32;
    let instance_size = (size_of::<RtAccelerationStructureInstance>() * instances.len()) as u64;

    // Instance buffer creation
    let (instance_buffer, instance_buffer_memory) = create_buffer(
        instance, device, data, instance_size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let mapped = device.map_memory(instance_buffer_memory, 0, instance_size, vk::MemoryMapFlags::empty())?
        as *mut u8;
    memcpy(instances.as_ptr().cast::<u8>(), mapped, instance_size as usize);
    device.unmap_memory(instance_buffer_memory);

    let instance_buffer_addr = get_buffer_device_address(device, instance_buffer);
    if instance_buffer_addr == 0 {
        return Err(anyhow!("TLAS instance buffer has a zero device address"));
    }

    let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR { device_address: instance_buffer_addr })
        .build();

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances: instances_data })
        .build();

    let geometries = &[geometry];

    // Get sizes
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
        )
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    device.get_acceleration_structure_build_sizes_khr(
        vk::AccelerationStructureBuildTypeKHR::DEVICE,
        &build_info,
        &[instance_count],
        &mut size_info,
    );

    // Tlas storage
    let (as_buffer, as_buffer_memory) = create_buffer(
        instance, device, data,
        size_info.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(as_buffer)
        .size(size_info.acceleration_structure_size)
        .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

    let tlas = device.create_acceleration_structure_khr(&create_info, None)?;

    // Scratch buffer
    let scratch_size = size_info.build_scratch_size.max(size_info.update_scratch_size);
    let scratch_alignment = 256u64; // or query minAccelerationStructureScratchOffsetAlignment properly
    let (scratch_buffer, scratch_buffer_memory) = create_buffer(
        instance, device, data,
        scratch_size + scratch_alignment,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let raw_scratch_addr = get_buffer_device_address(device, scratch_buffer);
    let scratch_addr = (raw_scratch_addr + scratch_alignment - 1) & !(scratch_alignment - 1);

    let build_info = build_info
        .dst_acceleration_structure(tlas)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });

    let range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(instance_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0)
        .build();

    // Sumbit build
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc_info)?[0];

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(cmd, &begin_info)?;
    device.cmd_build_acceleration_structures_khr(cmd, &[build_info], &[&[range_info]]);
    device.end_command_buffer(cmd)?;

    let cmd_buffers = &[cmd];
    let submit_info = vk::SubmitInfo::builder().command_buffers(cmd_buffers);
    device.queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;
    device.free_command_buffers(data.command_pool, &[cmd]);

    // Store
    data.tlas = tlas;
    data.tlas_buffer = as_buffer;
    data.tlas_buffer_memory = as_buffer_memory;

    data.tlas_scratch_buffer = scratch_buffer;
    data.tlas_scratch_buffer_memory = scratch_buffer_memory;
    data.tlas_scratch_addr = scratch_addr;

    data.instance_buffer = instance_buffer;
    data.instance_buffer_memory = instance_buffer_memory;
    data.instance_buffer_mapped = mapped;
    data.instance_buffer_addr = instance_buffer_addr;
    data.instance_count = instance_count;

    Ok(())
}}


// Pipeline creation
unsafe fn create_rt_pipeline(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Load shaders
    let raygen_bytes = include_bytes!("../shaders/raygen.spv");
    let miss_bytes = include_bytes!("../shaders/miss.spv");
    let shadow_miss_bytes = include_bytes!("../shaders/shadow.spv");
    let chit_bytes = include_bytes!("../shaders/closesthit.spv");

    unsafe {
        let raygen_module = create_shader_module(device, &raygen_bytes[..])?;
        let miss_module = create_shader_module(device, &miss_bytes[..])?;
        let shadow_miss_module = create_shader_module(device, &shadow_miss_bytes[..])?;
        let chit_module = create_shader_module(device, &chit_bytes[..])?;

        let rahit_module = if ENABLE_CUTOUT_SHADER {
            Some(create_shader_module(
                device,
                include_bytes!("../shaders/cutout.spv"),
            )?)
        } else {
            None
        };

        let raygen_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(raygen_module)
            .name(b"main\0");

        let miss_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(miss_module)
            .name(b"main\0");

        let shadow_miss_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(shadow_miss_module)
            .name(b"main\0");

        let chit_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(chit_module)
            .name(b"main\0");

        let any_hit_stage = rahit_module.map(|module| {
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                .module(module)
                .name(b"main\0")
                .build()
        });


        let mut stages = vec![
            raygen_stage.build(),
            miss_stage.build(),
            shadow_miss_stage.build(),
            chit_stage.build(),
        ];

        if let Some(any_hit_stage) = any_hit_stage {
            stages.push(any_hit_stage);
        }

        // Setup shader groups
        let raygen_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);

        let miss_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);

        let shadow_miss_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(2)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);

        let any_hit_shader = if ENABLE_CUTOUT_SHADER {
            4
        } else {
            vk::SHADER_UNUSED_KHR
        };

        let hit_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .type_(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(3)
            .any_hit_shader(any_hit_shader)
            .intersection_shader(vk::SHADER_UNUSED_KHR);

        let groups = &[raygen_group, miss_group, shadow_miss_group, hit_group];

        // Layout pipeline
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .offset(0)
            .size(size_of::<RtPushConstants>() as u32);

        let set_layouts = &[data.descriptor_set_layout];
        let push_constant_ranges = &[push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);

        data.rt_pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        // Create pipeline
        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::builder()
            .stages(&stages)
            .groups(groups)
            .max_pipeline_ray_recursion_depth(2)
            .layout(data.rt_pipeline_layout);

        let pipelines = device.create_ray_tracing_pipelines_khr(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[pipeline_info],
            None,
        )?;

        data.rt_pipeline = pipelines.0[0];

        // Cleanup
        device.destroy_shader_module(raygen_module, None);
        device.destroy_shader_module(miss_module, None);
        device.destroy_shader_module(shadow_miss_module, None);
        device.destroy_shader_module(chit_module, None);
        match rahit_module {
            Some(ramod) => device.destroy_shader_module(ramod, None),
            None => {},
        }
    }

    Ok(())
}

unsafe fn create_denoise_pipeline(
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(std::slice::from_ref(&data.descriptor_set_layout));

    data.denoise_pipeline_layout =
        device.create_pipeline_layout(&pipeline_layout_info, None)?;

    let denoise_bytes = include_bytes!("../shaders/denoise.spv");
    let denoise_module = create_shader_module(device, &denoise_bytes[..])?;

    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(denoise_module)
        .name(b"main\0");

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .stage(stage)
        .layout(data.denoise_pipeline_layout);

    let pipelines = device.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[pipeline_info],
        None,
    )?;

    data.denoise_pipeline = pipelines.0[0];

    device.destroy_shader_module(denoise_module, None);

    Ok(())
}}

unsafe fn create_skinning_pipeline(device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let rest_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let skinned_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let joints_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let bindings = &[rest_binding, skinned_binding, joints_binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
    data.skinning_descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(size_of::<u32>() as u32);

    let set_layouts = &[data.skinning_descriptor_set_layout];
    let push_constant_ranges = &[push_constant_range];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);
    data.skinning_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    let skin_bytes = include_bytes!("../shaders/skin.spv");
    let skin_module = create_shader_module(device, &skin_bytes[..])?;

    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(skin_module)
        .name(b"main\0");

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .stage(stage)
        .layout(data.skinning_pipeline_layout);

    let pipelines = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?;
    data.skinning_pipeline = pipelines.0[0];

    device.destroy_shader_module(skin_module, None);
    Ok(())
}}


unsafe fn create_shader_binding_table(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    // Get alignment rules
    let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::builder()
        .push_next(&mut rt_props);

    instance.get_physical_device_properties2(data.physical_device, &mut props2);


    let handle_size = rt_props.shader_group_handle_size;
    let handle_alignment = rt_props.shader_group_handle_alignment;
    let base_alignment = rt_props.shader_group_base_alignment;

    let align_up = |size: u32, align: u32| -> u32 {
        (size + align - 1) & !(align - 1)
    };

    let handle_size_aligned = align_up(handle_size, handle_alignment);

    let group_count = 4u32;

    // Get group handles
    let handle_data_size = (handle_size * group_count) as usize;
    let mut handle_data = vec![0u8; handle_data_size];
        
    device.get_ray_tracing_shader_group_handles_khr(
        data.rt_pipeline,
        0,
        group_count,
        &mut handle_data,
    )?;

    // Region sizes
    let raygen_region_size = handle_size_aligned;
    let miss_region_size = handle_size_aligned * 2;
    let hit_region_size = handle_size_aligned;
    let miss_region_offset = align_up(raygen_region_size, base_alignment);
    let hit_region_offset = align_up(miss_region_offset + miss_region_size, base_alignment);

    let sbt_size = (hit_region_offset + hit_region_size) as u64;

    let (sbt_buffer, sbt_buffer_memory) = create_buffer(
        instance, device, data, sbt_size,
        vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Write to the right places
    let mem = device.map_memory(sbt_buffer_memory, 0, sbt_size, vk::MemoryMapFlags::empty())?;
    let mem_ptr = mem as *mut u8;

    // Raygen
    memcpy(handle_data.as_ptr(), mem_ptr, handle_size as usize);

    // Miss (sky)
    memcpy(
        handle_data.as_ptr().add(handle_size as usize),
        mem_ptr.add(miss_region_offset as usize),
        handle_size as usize,
    );

    // Miss (shadow)
    memcpy(
        handle_data.as_ptr().add((handle_size * 2) as usize),
        mem_ptr.add((miss_region_offset + handle_size_aligned) as usize),
        handle_size as usize,
    );

    // Hit
    memcpy(
        handle_data.as_ptr().add((handle_size * 3) as usize),
        mem_ptr.add(hit_region_offset as usize),
        handle_size as usize,
    );

    device.unmap_memory(sbt_buffer_memory);


    let sbt_address = get_buffer_device_address(device, sbt_buffer);

    let raygen_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(sbt_address)
        .stride(handle_size_aligned as u64)
        .size(raygen_region_size as u64);

    let miss_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(sbt_address + miss_region_offset as u64)
        .stride(handle_size_aligned as u64)
        .size(miss_region_size as u64);

    let hit_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(sbt_address + hit_region_offset as u64)
        .stride(handle_size_aligned as u64)
        .size(hit_region_size as u64);

    let callable_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(0)
        .stride(0)
        .size(0);

    // Store
    data.sbt_buffer = sbt_buffer;
    data.sbt_buffer_memory = sbt_buffer_memory;
    data.sbt_raygen_region = *raygen_region;
    data.sbt_miss_region = *miss_region;
    data.sbt_hit_region = *hit_region;
    data.sbt_callable_region = *callable_region;

    Ok(())
}}


// Descriptor creation
unsafe fn create_descriptor_set_layout(
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    let rt_stages = vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::MISS_KHR
        | vk::ShaderStageFlags::ANY_HIT_KHR
        | vk::ShaderStageFlags::COMPUTE;

    let as_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    let image_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(2)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    let object_descs_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(3)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    let materials_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(4)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    let material_ids_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(5)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    let accum_image_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(6)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(rt_stages);

    // MUST ALWAYS BE HIGHEST
    let textures_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(7)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(MAX_TEXTURES)
        .stage_flags(rt_stages);


    let bindings = &[
        as_binding, image_binding,
        ubo_binding, 
        object_descs_binding, materials_binding, material_ids_binding,
        accum_image_binding,
        textures_binding,
    ];

    let binding_flags = &[
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::empty(),
        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
    ];
    let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
        .binding_flags(binding_flags);

    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings)
        .push_next(&mut binding_flags_info);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}}

unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let as_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(data.swapchain_images.len() as u32);

    let image_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(data.swapchain_images.len() as u32);

    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let object_descs_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let materials_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let storage_buffer_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count((data.swapchain_images.len() * 3) as u32);

    let textures_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32 * MAX_TEXTURES);

    let accum_image_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(data.swapchain_images.len() as u32);


    let pool_sizes = &[
        as_size, image_size, ubo_size, object_descs_size, materials_size, 
        storage_buffer_size, textures_size,
        accum_image_size,
    ];

    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}}

unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    
    let texture_counts = vec![MAX_TEXTURES; data.swapchain_images.len()];
    let mut variable_count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
        .descriptor_counts(&texture_counts);

    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts)
        .push_next(&mut variable_count_info);

    let texture_infos: Vec<vk::DescriptorImageInfo> = data.textures
        .iter()
        .map(|(_, _, view)| {
            vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(*view)
                .sampler(data.texture_sampler)
                .build()
        })
        .collect();

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        let mut writes = Vec::new();

        // Acceleration structure write
        let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
            .acceleration_structures(std::slice::from_ref(&data.tlas))
            .build();

        let mut as_write = vk::WriteDescriptorSet::builder()
            .push_next(&mut as_write_info)
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .build();
        as_write.descriptor_count = 1;
        writes.push(as_write);


        // Image write
        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(data.storage_image_views[i])
            .build();

        let image_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info))
            .build();
        writes.push(image_write);


        // UBO write
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<CameraUniformBufferObject>() as u64)
            .build();

        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info))
            .build();
        writes.push(ubo_write);


        // Object descriptions write
        let object_descs_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.object_descs_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build();

        let object_descs_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&object_descs_info))
            .build();
        writes.push(object_descs_write);


        // Materials write
        let materials_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.materials_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build();

        let materials_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(4)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&materials_info))
            .build();
        writes.push(materials_write);


        // Material id write
        let material_ids_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.material_ids_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build();

        let material_ids_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(5)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&material_ids_info))
            .build();
        writes.push(material_ids_write);


        // TA write
        let accum_image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(data.accum_image_view)
            .build();

        let accum_image_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(6)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&accum_image_info))
            .build();
        writes.push(accum_image_write);


        // Texture write
        if !texture_infos.is_empty() {
            let textures_write = vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(7)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&texture_infos)
                .build();
            writes.push(textures_write)
        }


        device.update_descriptor_sets(
            &writes,
            &[] as &[vk::CopyDescriptorSet],
        );
    }

    Ok(())
}}

unsafe fn create_skinning_descriptor_pool(device: &Device, data: &mut AppData, rigged_model_count: u32) -> Result<()> { unsafe {
    let pool_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(rigged_model_count * 3);

    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(std::slice::from_ref(&pool_size))
        .max_sets(rigged_model_count.max(1))
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    data.skinning_descriptor_pool = device.create_descriptor_pool(&info, None)?;
    data.skinning_pool_capacity = rigged_model_count.max(1);
    Ok(())
}}


fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;

    let window = WindowBuilder::new()
        .with_title("Vulkan")
        .build(&event_loop)?;

    let mut app = unsafe {App::create(&window)?};

    let mut last_fps_update = Instant::now();
    let mut frames = 0u32;
    
    let mut minimized = false;

    let mut cursor_grabbed = true;
    window.set_cursor_visible(false);
    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_e| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
        .ok();


    let protogen;
    unsafe {
        protogen = Some(Scene::load_model_into_memory(
            &mut app.scene,
            "models/Protogen.glb",
            &app.instance, &app.device, &mut app.data,
        )?);
    }
    
    
    event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => {
                if !minimized {
                    window.request_redraw();
                }
            }

            Event::WindowEvent {
                event: WindowEvent::RedrawRequested, ..
            } => {
                unsafe {
                    if !elwt.exiting() && !minimized {
                        if let Err(e) = app.render(&window) {
                            error!("Render failed: {:?}", e);
                            panic!("Render failed: {:?}", e);
                        }
                    }

                    frames += 1;
                    let elapsed = last_fps_update.elapsed();

                    if elapsed >= Duration::from_secs(1) {
                        let fps = frames as f64 / elapsed.as_secs_f64();
                        let ms = app.frame_time;

                        window.set_title(&format!(
                            "Vulkan - {:.0} FPS ({:.2} ms)",
                            fps,
                            ms
                        ));

                        frames = 0;
                        last_fps_update = Instant::now();
                    }
                }
            }
            
            Event::WindowEvent {
                event: WindowEvent::Resized(size), ..
            } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            },

            Event::WindowEvent {
                event: WindowEvent::CloseRequested, ..
            } => {
                unsafe {
                    app.device.device_wait_idle().unwrap();
                    app.destroy();
                }

                elwt.exit();
            }

            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event: key_event, .. }, ..
            } => {
                let pressed = key_event.state == winit::event::ElementState::Pressed;
                
                match key_event.physical_key {
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => app.input.forward = pressed,
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) => app.input.back = pressed,
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyA) => app.input.left = pressed,
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyD) => app.input.right = pressed,
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) => app.input.up = pressed,
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft) => app.input.down = pressed,

                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyP) => {
                        if pressed {
                            app.scene.animations[2].play("Chop_Tree RT");
                        }
                    },

                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyO) => {
                        if pressed {
                            match &protogen {
                                Some(protogen) => {
                                    let transform = Mat4::from_translation(vec3(app.camera.position.x, app.camera.position.y, app.camera.position.z)) * Mat4::identity();
                                    app.queue_add_model(protogen, ModelClass::SemiDynamic, transform, None);
                                },
                                None => error!("Could not add model to scene"),
                            }
                        }
                    },

                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyL) => {
                        if pressed {
                            let transform = Mat4::from_translation(vec3(app.camera.position.x, app.camera.position.y, app.camera.position.z)) * Mat4::identity();
                            unsafe {let _ = app.move_semi_dynamic_model(StringOrInt::Int(app.scene.model_info.len() - 1), transform);}
                        }
                    },

                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyI) => {
                        if pressed {
                            app.queue_remove_model(StringOrInt::Int(app.scene.model_info.len() - 1));
                        }
                    },
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyK) => {
                        if pressed {
                            app.queue_remove_model(StringOrInt::Str("Xenon".to_owned()));
                        }
                    },

                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                        if pressed {
                            cursor_grabbed = !cursor_grabbed;
                            app.controls_enabled = cursor_grabbed;

                            window.set_cursor_visible(!cursor_grabbed);
                            if cursor_grabbed {
                                window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                    .or_else(|_e| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
                                    .ok();
                            } else {
                                window.set_cursor_grab(winit::window::CursorGrabMode::None).ok();
                            }
                        }
                    },
                    _ => {}
                }
            }

            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion {delta}, ..
            } => {
                app.input.mouse_dx += delta.0 as f32;
                app.input.mouse_dy += delta.1 as f32;
            }

            _ => {}
        }
    })?;

    Ok(())
}
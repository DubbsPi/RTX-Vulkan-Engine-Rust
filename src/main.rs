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

use cgmath::{vec3, point3, Deg};
use cgmath::SquareMatrix;
use cgmath::InnerSpace;


type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;


const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_ACCELERATION_STRUCTURE_EXTENSION.name,
    vk::KHR_RAY_TRACING_PIPELINE_EXTENSION.name,
    vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION.name,
    vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name,
];

const MAX_FRAMES_IN_FLIGHT: usize = 3;


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

    camera: Camera,
    input: InputState,
    controls_enabled: bool,
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

        data.timestamp_query_pool = device.create_query_pool(&query_pool_info, None)?;

        let properties = instance.get_physical_device_properties(data.physical_device);
        data.timestamp_period = properties.limits.timestamp_period;

        create_swapchain(window, &instance, &device, &mut data)?;

        create_command_pool(&instance, &device, &mut data)?;
        create_storage_image(&instance, &device, &mut data)?;

        create_descriptor_set_layout(&device, &mut data)?; 
        create_rt_pipeline(&device, &mut data)?;

        let (vertices, indices, prim_material_ids, materials) = load_obj(
            "models/FLOWER SET_18.obj", 0.05
        )?;

        info!("Triangles: {}, Vertices: {}", indices.len() / 3, vertices.len());

        create_blas(&instance, &device, &mut data, &vertices, &indices)?;
        create_tlas(&instance, &device, &mut data)?;

        create_scene_buffers(&instance, &device, &mut data, &materials, &prim_material_ids)?;

        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;

        create_shader_binding_table(&instance, &device, &mut data)?;

        create_command_buffers(&device, &mut data)?;

        create_sync_objects(&device, &mut data)?;
        
        let frame = 0;
        let resized = false;
        let last_frame = Instant::now();
        let frame_time = 0.0;

        let camera = Camera::new();
        let input = InputState::default();
        let controls_enabled = true;

        Ok(Self {
            _entry, instance: instance, data, device,
            frame, resized,
            last_frame, frame_time,
            camera, input, controls_enabled
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

        // Blas
        self.device.destroy_acceleration_structure_khr(self.data.blas, None);
        self.device.destroy_buffer(self.data.blas_buffer, None);
        self.device.free_memory(self.data.blas_buffer_memory, None);

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

        self.device.destroy_pipeline(self.data.rt_pipeline, None);
        self.device.destroy_pipeline_layout(self.data.rt_pipeline_layout, None);

        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);

        self.data.uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));
        self.data.uniform_buffers_memory
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));

        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);

        self.data.swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));

        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }}

    unsafe fn render(&mut self, window: &Window) -> Result<()> { unsafe {
        self.device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame]],
            true,
            u64::MAX,
        )?;


        if self.frame_time != 0.0 || self.frame >= MAX_FRAMES_IN_FLIGHT {
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

        
        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::TRANSFER];
        
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
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
        create_rt_pipeline(&self.device, &mut self.data)?;

        create_shader_binding_table(&self.instance, &self.device, &mut self.data)?;
        
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;

        create_command_buffers(&self.device, &mut self.data)?;

        self.data.images_in_flight = vec![vk::Fence::null(); self.data.swapchain_images.len()];

        Ok(())
    }}

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> { unsafe {
        let view = self.camera.view_matrix();

        let mut proj = cgmath::perspective(
            Deg(45.0),
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            0.1,
            100.0,
        );

        proj[1][1] *= -1.0; // Invert Y for Vulkan coordinate space
        proj[0][0] *= -1.0; // Invert X for me

        let view_inverse = view.invert().ok_or_else(|| anyhow!("Failed to invert view matrix"))?;
        let proj_inverse = proj.invert().ok_or_else(|| anyhow!("Failed to invert proj matrix"))?;

        let ubo = CameraUniformBufferObject {view_inverse, proj_inverse};

        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<CameraUniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(
            (&ubo as *const CameraUniformBufferObject).cast::<u8>(),
            memory.cast::<u8>(),
            size_of::<CameraUniformBufferObject>(),
        );

        self.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);
        
        
        Ok(())
    }}

    fn update_camera(&mut self, dt: f32) {
        if self.controls_enabled {
            self.camera.yaw -= self.input.mouse_dx * self.camera.sensitivity;
            self.camera.pitch -= self.input.mouse_dy * self.camera.sensitivity;
            self.camera.pitch = self.camera.pitch.clamp(-1.55, 1.55);

            self.input.mouse_dx = 0.0;
            self.input.mouse_dy = 0.0;

            let forward = self.camera.forward();
            let right = self.camera.right();
            let mut delta = vec3(0.0, 0.0, 0.0);

            if self.input.forward { delta += forward; }
            if self.input.back    { delta -= forward; }
            if self.input.right   { delta -= right; }
            if self.input.left    { delta += right; }
            if self.input.up      { delta.y += 1.0; }
            if self.input.down    { delta.y -= 1.0; }

            if delta.magnitude() > 0.0001 {
                delta = delta.normalize() * self.camera.speed * dt;
                self.camera.position += delta;
            }
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

    // Storage images
    storage_images: Vec<vk::Image>,
    storage_image_memories: Vec<vk::DeviceMemory>,
    storage_image_views: Vec<vk::ImageView>,

    // Geometry
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    // Materials
    materials_buffer: vk::Buffer,
    materials_buffer_memory: vk::DeviceMemory,
    object_descs_buffer: vk::Buffer,
    object_descs_buffer_memory: vk::DeviceMemory,
    material_ids_buffer: vk::Buffer,
    material_ids_buffer_memory: vk::DeviceMemory,

    // Blas
    blas: vk::AccelerationStructureKHR,
    blas_buffer: vk::Buffer,
    blas_buffer_memory: vk::DeviceMemory,

    // Tlas
    tlas: vk::AccelerationStructureKHR,
    tlas_buffer: vk::Buffer,
    tlas_buffer_memory: vk::DeviceMemory,

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
}


#[derive(Debug, Error)]
#[error("Missing {0}")]
pub struct SuitabilityError(pub &'static str);


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
}

impl Vertex {
    const fn new(pos: Vec3) -> Self {
        Self {pos}
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Material {
    albedo: Vec3,
    _pad0: f32,
    metallic: f32,
    roughness: f32,
    _pad1: [f32; 2],
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
    fn new() -> Self {
        Self {
            position: point3(0.0, 0.0, -2.5),
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
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct RtAccelerationStructureInstance {
    transform: vk::TransformMatrixKHR,
    instance_custom_index_and_mask: u32,
    instance_sbt_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}


// Utility functions
fn pack_custom_index_and_mask(custom_index: u32, mask: u8) -> u32 {
    (custom_index & 0x00FF_FFFF) | ((mask as u32) << 24)
}

fn pack_sbt_offset_and_flags(sbt_offset: u32, flags: vk::GeometryInstanceFlagsKHR) -> u32 {
    (sbt_offset & 0x00FF_FFFF) | ((flags.bits() as u32) << 24)
}

fn load_obj(path: &str, scale: f32) -> Result<(Vec<Vertex>, Vec<u32>, Vec<u32>, Vec<Material>)> {
    let (models, materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;

    let converted_materials = convert_materials(&materials.unwrap_or_default());
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut material_ids = Vec::new();

    for model in &models {
        let mesh = &model.mesh;
        let vertex_offset = vertices.len() as u32;

        let triangle_count = mesh.indices.len() / 3;
        
        let mat_id = mesh.material_id.unwrap_or(0) as u32;

        material_ids.extend(std::iter::repeat(mat_id).take(triangle_count));

        vertices.extend(
            mesh.positions
                .chunks(3)
                .map(|p| Vertex::new(vec3(p[0] * scale, p[1] * scale, p[2] * scale))),
        );

        indices.extend(mesh.indices.iter().map(|i| i + vertex_offset));
    }

    Ok((vertices, indices, material_ids, converted_materials))
}

fn convert_materials(tobj_materials: &[tobj::Material]) -> Vec<Material> {
    tobj_materials
        .iter()
        .map(|m| {
            let albedo = m.diffuse.unwrap_or([0.8, 0.8, 0.8]);
            Material {
                albedo: vec3(albedo[0], albedo[1], albedo[2]),
                _pad0: 0.0,
                metallic: m.unknown_param.get("Pm").and_then(|s| s.parse().ok()).unwrap_or(0.0),
                roughness: m.unknown_param.get("Pr").and_then(|s| s.parse().ok()).unwrap_or(0.5),
                _pad1: [0.0, 0.0],
            }
        })
        .collect()
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

    let mut features2 = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut rt_pipeline_features)
        .push_next(&mut accel_struct_features)
        .push_next(&mut bda_features)
        .push_next(&mut scalar_block_layout_features);

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
    
    let base_features = vk::PhysicalDeviceFeatures::builder()
        .shader_int64(true);
    
    let mut features2 = vk::PhysicalDeviceFeatures2::builder()
        .features(base_features)
        .push_next(&mut rt_pipeline_features)
        .push_next(&mut accel_struct_features)
        .push_next(&mut bda_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut scalar_block_layout_features);

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
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
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


unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
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

    for (i, command_buffer) in data.command_buffers.iter().enumerate() {
        let command_buffer = *command_buffer;
        let first_query = (i as u32) * 2;

        let begin_info = vk::CommandBufferBeginInfo::builder();
        device.begin_command_buffer(command_buffer, &begin_info)?;

        device.cmd_reset_query_pool(
            command_buffer,
            data.timestamp_query_pool,
            first_query,
            2,
        );

        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            data.timestamp_query_pool,
            (i as u32) * 2,
        );

        // Bind rt pipeline
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            data.rt_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            data.rt_pipeline_layout,
            0,
            &[data.descriptor_sets[i]],
            &[],
        );

        device.cmd_trace_rays_khr(
            command_buffer,
            &data.sbt_raygen_region,
            &data.sbt_miss_region,
            &data.sbt_hit_region,
            &data.sbt_callable_region,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            1,
        );

        // Setup barriers
        let storage_image_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(data.storage_images[i])
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
            .image(data.swapchain_images[i])
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[storage_image_barrier, swapchain_image_barrier],
        );

        // Use blit due to different image formats
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
                    x: data.swapchain_extent.width as i32,
                    y: data.swapchain_extent.height as i32,
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
                    x: data.swapchain_extent.width as i32,
                    y: data.swapchain_extent.height as i32,
                    z: 1,
                },
            ]);

        device.cmd_blit_image(
            command_buffer,
            data.storage_images[i],
            vk::ImageLayout::GENERAL,
            data.swapchain_images[i],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit_region],
            vk::Filter::NEAREST,
        );

        // Move image to presentable format
        let present_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(data.swapchain_images[i])
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build())
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[present_barrier],
        );

        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            data.timestamp_query_pool,
            (i as u32) * 2 + 1,
        );

        device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}}

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
            .format(vk::Format::R8G8B8A8_UNORM)
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
            .format(vk::Format::R8G8B8A8_UNORM)
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

    // If this buffer was created with SHADER_DEVICE_ADDRESS usage, the
    // memory allocation needs the matching flag or get_buffer_device_address
    // will fail validation even though everything else checks out.
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

unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<CameraUniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}}

unsafe fn create_scene_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    materials: &[Material],
    prim_material_ids: &[u32],
) -> Result<()> { unsafe {
    // Material buffer
    let materials_size = (size_of::<Material>() * materials.len().max(1)) as u64;
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
    let object_descs = vec![ObjectDesc {
        vertex_address: get_buffer_device_address(device, data.vertex_buffer),
        index_address: get_buffer_device_address(device, data.index_buffer),
        material_id: 0,
        _pad: 0,
    }];
    let objects_size = (size_of::<ObjectDesc>() * object_descs.len()) as u64;
    let (objects_buffer, objects_buffer_memory) = create_buffer(
        instance, device, data, objects_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    let mem = device.map_memory(objects_buffer_memory, 0, objects_size, vk::MemoryMapFlags::empty())?;
    memcpy(object_descs.as_ptr().cast::<u8>(), mem.cast::<u8>(), objects_size as usize);
    device.unmap_memory(objects_buffer_memory);

    // Material Ids buffer
    let ids_size = (size_of::<u32>() * prim_material_ids.len()) as u64;
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


unsafe fn create_blas(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    vertices: &[Vertex],
    indices: &[u32],
) -> Result<()> { unsafe {
    let vertex_size = (size_of::<Vertex>() * vertices.len()) as u64;
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance, device, data, vertex_size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let mem = device.map_memory(vertex_buffer_memory, 0, vertex_size, vk::MemoryMapFlags::empty())?;
    memcpy(
        vertices.as_ptr().cast::<u8>(),
        mem.cast::<u8>(),
        size_of::<Vertex>() * vertices.len(),
    );
    device.unmap_memory(vertex_buffer_memory);

    let index_size = (size_of::<u32>() * indices.len()) as u64;
    let (index_buffer, index_buffer_memory) = create_buffer(
        instance, device, data, index_size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let mem = device.map_memory(index_buffer_memory, 0, index_size, vk::MemoryMapFlags::empty())?;
    memcpy(
        indices.as_ptr().cast::<u8>(),
        mem.cast::<u8>(),
        size_of::<u32>() * indices.len(),
    );
    device.unmap_memory(index_buffer_memory);


    let index_addr = get_buffer_device_address(device, index_buffer);
    if index_addr == 0 {
        return Err(anyhow!("BLAS index buffer has a zero device address"));
    }

    let vertex_addr = get_buffer_device_address(device, vertex_buffer);
    if vertex_addr == 0 {
        return Err(anyhow!("BLAS vertex buffer has a zero device address"));
    }
    
    let triangle_count = (indices.len() / 3) as u32;

    let triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_addr })
        .vertex_stride(size_of::<Vertex>() as u64)
        .max_vertex(vertices.len() as u32 - 1)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_addr })
        .build();

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: triangles_data })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .build();

    let geometries = &[geometry];

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

    device.get_acceleration_structure_build_sizes_khr(
        vk::AccelerationStructureBuildTypeKHR::DEVICE,
        &build_info,
        &[triangle_count],
        &mut size_info,
    );


    // Create acceleration structure
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
        .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

    let blas = device.create_acceleration_structure_khr(&create_info, None)?;

    let (scratch_buffer, scratch_buffer_memory) = create_buffer(
        instance, device, data,
        size_info.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let scratch_addr = get_buffer_device_address(device, scratch_buffer);
    
    // Submit build
    build_info = build_info
        .dst_acceleration_structure(blas)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });

    let range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(triangle_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0)
        .build();

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc_info)?[0];

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(cmd, &begin_info)?;

    device.cmd_build_acceleration_structures_khr(
        cmd,
        &[build_info],
        &[&[range_info]],
    );

    device.end_command_buffer(cmd)?;
    
    let cmd_buffers = &[cmd];
    let submit_info = vk::SubmitInfo::builder().command_buffers(cmd_buffers);
    device.queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;
    device.free_command_buffers(data.command_pool, &[cmd]);

    // Clean up and store
    device.destroy_buffer(scratch_buffer, None);
    device.free_memory(scratch_buffer_memory, None);

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    data.blas = blas;
    data.blas_buffer = as_buffer;
    data.blas_buffer_memory = as_buffer_memory;

    Ok(())
}}

unsafe fn create_tlas(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    // Get blas address
    let blas_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
        .acceleration_structure(data.blas);

    let blas_address = device.get_acceleration_structure_device_address_khr(&blas_addr_info);

    let transform = vk::TransformMatrixKHR {
        matrix: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    let blas_instance = RtAccelerationStructureInstance {
        transform,
        instance_custom_index_and_mask: pack_custom_index_and_mask(0, 0xFF),
        instance_sbt_record_offset_and_flags: pack_sbt_offset_and_flags(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
        ),
        acceleration_structure_reference: blas_address,
    };

    let instance_size = size_of::<RtAccelerationStructureInstance>() as u64;

    let (instance_buffer, instance_buffer_memory) = create_buffer(
        instance, device, data, instance_size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let mem = device.map_memory(instance_buffer_memory, 0, instance_size, vk::MemoryMapFlags::empty())?;
    memcpy(
        (&blas_instance as *const RtAccelerationStructureInstance).cast::<u8>(),
        mem.cast::<u8>(),
        size_of::<RtAccelerationStructureInstance>(),
    );
    device.unmap_memory(instance_buffer_memory);


    let instance_buffer_addr = get_buffer_device_address(device, instance_buffer);

    let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {device_address: instance_buffer_addr})
        .build();

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {instances: instances_data})
        .build();

    let geometries = &[geometry];
    let instance_count = 1u32;

    // Get sizes
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);


    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    
    device.get_acceleration_structure_build_sizes_khr(
        vk::AccelerationStructureBuildTypeKHR::DEVICE,
        &build_info,
        &[instance_count],
        &mut size_info,
    );


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

    // Build
    let (scratch_buffer, scratch_buffer_memory) = create_buffer(
        instance, device, data,
        size_info.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    
    let scratch_addr = get_buffer_device_address(device, scratch_buffer);

    let build_info = build_info
        .dst_acceleration_structure(tlas)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });

    let range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(instance_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0)
        .build();

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

    // Cleanup and store
    device.destroy_buffer(scratch_buffer, None);
    device.free_memory(scratch_buffer_memory, None);
    device.destroy_buffer(instance_buffer, None);
    device.free_memory(instance_buffer_memory, None);

    data.tlas = tlas;
    data.tlas_buffer = as_buffer;
    data.tlas_buffer_memory = as_buffer_memory;

    Ok(())
}}


unsafe fn create_rt_pipeline(
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    // Load shaders
    let raygen_bytes = include_bytes!("../shaders/raygen.spv");
    let miss_bytes = include_bytes!("../shaders/miss.spv");
    let shadow_miss_bytes = include_bytes!("../shaders/shadow.spv"); // compiled from shadow.rmiss
    let chit_bytes = include_bytes!("../shaders/closesthit.spv");

    let raygen_module = create_shader_module(device, &raygen_bytes[..])?;
    let miss_module = create_shader_module(device, &miss_bytes[..])?;
    let shadow_miss_module = create_shader_module(device, &shadow_miss_bytes[..])?;
    let chit_module = create_shader_module(device, &chit_bytes[..])?;

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

    let stages = &[raygen_stage, miss_stage, shadow_miss_stage, chit_stage];


    // Setup shader groups
    let raygen_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(0) // raygen_stage
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR);

    let miss_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(1) // miss_stage
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR);

    let shadow_miss_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(2) // shadow_miss_stage
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR);

    let hit_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .type_(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
        .general_shader(vk::SHADER_UNUSED_KHR)
        .closest_hit_shader(3) // chit_stage
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR);

    let groups = &[raygen_group, miss_group, shadow_miss_group, hit_group];

    // Layout pipeline
    let set_layouts = &[data.descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts);

    data.rt_pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Create pipeline
    let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .stages(stages)
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


unsafe fn create_descriptor_set_layout(
    device: &Device,
    data: &mut AppData,
) -> Result<()> { unsafe {
    let rt_stages = vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::MISS_KHR;

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


    let bindings = &[
        as_binding,
        image_binding,
        ubo_binding,
        object_descs_binding,
        materials_binding,
        material_ids_binding,
    ];

    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

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

    let pool_sizes = &[as_size, image_size, ubo_size, object_descs_size, materials_size, storage_buffer_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}}

unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> { unsafe {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        // Build acceleration structure info
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

        // Material write
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


        device.update_descriptor_sets(
            &[as_write, image_write, ubo_write, object_descs_write, materials_write, material_ids_write],
            &[] as &[vk::CopyDescriptorSet],
        );
    }

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
                        app.render(&window).unwrap();
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
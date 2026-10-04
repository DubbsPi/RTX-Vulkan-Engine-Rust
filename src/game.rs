use anyhow::Result;

use cgmath::Deg;
use cgmath::InnerSpace;

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};


use crate::Instance;
use crate::Device;
use crate::App;
use crate::AppData;
use crate::StringOrInt;
use crate::ModelClass;
use crate::Camera;
use crate::ShaderSettings;

use crate::scene::Scene;

use crate::common::Vec3;
use crate::common::Vec3d;
use crate::common::Mat4;

use crate::terrain_gen::*;


pub unsafe fn create_scene(instance: &Instance, device: &Device, data: &mut AppData) -> Result<(Scene, Camera)> {
    data.shader_settings = ShaderSettings {
        max_accumulation: 4096,
        max_bounces: 6,
        max_nee_tests: 6,

        sky_view_samples: 18,
        sky_light_samples: 8,
    };

    
    let mut scene = Scene::new();

    // Create small system
    data.game_data.planets.push( Planet {
        position: Vec3d::new(0.0, 0.0, 0.0),
        radius: 6371000.0,
        atmosphere_radius: 100000.0,
        parent_star: 0,
        planet_color: Vec3::new(0.25, 1.0, 0.25),
        ..Default::default()
    });
    let planet = &data.game_data.planets[0];
    
    // Generate planet surface
    //let scale = 10000.0;

    //let fbm = Fbm::<Perlin>::new(1)
    //    .set_octaves(6)
    //    .set_frequency(0.001)
    //    .set_lacunarity(2.0)
    //    .set_persistence(0.5);
    //let noise_plane = generate_noise_plane(&fbm, &planet, 500, scale);
    //scene.add_model_to_scene(&noise_plane, ModelClass::Rigid, Some("Ground".to_owned()));
    //scene.translate_model(StringOrInt::Str("Ground".to_owned()), Vec3::new(0.0, planet.radius as f32, 0.0));
        
    let origin_y = planet.radius + 50.0;

    let camera = Camera::new(Vec3d::new(0.0, origin_y, 0.0), 50.0);


    let sun_dist = 149600000000.0;
    let sun_dir = Vec3d::new(-0.4, 0.0, 0.6).normalize();
    data.game_data.stars.push(Star {
        position: sun_dir * sun_dist,
        radius: 695700000.0,
        brightness: 1.5e24,
        ..Default::default()
    });
    
    // Add models
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
        let cubes = Scene::load_model_into_memory(
            &mut scene,
            "models/Test_Cubes.glb",
            &instance, &device, data,
        )?;


        scene.add_model_to_scene(&magazine, ModelClass::Rigid, None);
        scene.add_model_to_scene(&magazine, ModelClass::Rigid, None);

        scene.scale_model(StringOrInt::Int(2), Vec3::new(5.0, 5.0, 5.0));
        scene.translate_model(StringOrInt::Int(2), Vec3::new(1.0, camera.position.y as f32 + 10.5, 0.0));

        scene.scale_model(StringOrInt::Int(1), Vec3::new(20.0, 5.0, 10.0));
        let rotation = Mat4::from_angle_z(Deg(90.0));
        scene.transform_model(StringOrInt::Int(1), rotation);
        scene.translate_model(StringOrInt::Int(1), Vec3::new(0.0, camera.position.y as f32 + 8.0, 0.0));

        // Merge magazines
        scene.set_blas_group(StringOrInt::Int(1), Some(0));
        scene.set_blas_group(StringOrInt::Int(2), Some(0));

        scene.add_model_to_scene(&protogen, ModelClass::Deformable, Some("Xenon".to_owned()));
        scene.translate_model(StringOrInt::Str("Xenon".to_owned()), Vec3::new(-4.0, camera.position.y as f32 + 10.0, 0.0));

        scene.add_model_to_scene(&room, ModelClass::Rigid, Some("Room".to_owned()));
        scene.translate_model(StringOrInt::Str("Room".to_owned()), Vec3::new(0.0, camera.position.y as f32 + 5.0, 8.0));
    
        scene.add_model_to_scene(&cubes, ModelClass::Rigid, Some("Cubes".to_owned()));
        scene.scale_model(StringOrInt::Str("Cubes".to_owned()), Vec3::new(1.0, 1.0, 0.5));
        scene.translate_model(StringOrInt::Str("Cubes".to_owned()), Vec3::new(0.0, camera.position.y as f32 + 16.5, 8.0));
    }

    Ok((scene, camera))
}

pub fn render_tick(_app: &mut App, _dt: f32) {
    // Do nothing for now
}


#[derive(smart_default::SmartDefault)]
pub struct Planet {
    #[default(Vec3d::new(0.0, 0.0, 0.0))]
    pub position: Vec3d,
    #[default(1000.0)]
    pub radius: f64,
    #[default(0.0)]
    pub atmosphere_radius: f64,

    #[default(Vec3::new(1.0, 1.0, 1.0))]
    pub planet_color: Vec3,
    #[default(-1)]
    pub parent_star: i32,

    #[default(Vec3::new(5.8e-6, 13.5e-6, 33.1e-6))]
    pub beta_rayleigh: Vec3,
    #[default(21e-6)]
    pub beta_mie: f32,
    #[default(0.76)]
    pub mie_g: f32,
    #[default(8500.0)]
    pub hr: f32,
    #[default(1200.0)]
    pub hm: f32,
}

#[repr(C)]
#[derive(smart_default::SmartDefault, Copy, Clone, Debug)]
pub struct PlanetRenderInfo {
    #[default(Vec3::new(0.0, 0.0, 0.0))]
    pub planet_up: Vec3,
    pub camera_dist: f32,
    pub radius: f32,
    pub atmosphere_radius: f32,

    #[default(Vec3::new(0.0, 0.0, 0.0))]
    pub planet_color: Vec3,
    pub parent_star: i32,

    #[default(Vec3::new(0.0, 0.0, 0.0))]
    pub beta_rayleigh: Vec3,
    pub beta_mie: f32,
    pub mie_g: f32,
    pub hr: f32,
    pub hm: f32,
}


#[derive(smart_default::SmartDefault)]
pub struct Star {
    #[default(Vec3d::new(0.0, 0.0, 0.0))]
    pub position: Vec3d,
    #[default(1000.0)]
    pub radius: f64,
    
    #[default(Vec3::new(1.0, 1.0, 1.0))]
    pub color: Vec3,
    #[default(1000.0)]
    pub brightness: f32,
}

#[repr(C)]
#[derive(smart_default::SmartDefault, Copy, Clone, Debug)]
pub struct StarRenderInfo {
    #[default(Vec3::new(0.0, 0.0, 0.0))]
    pub star_up: Vec3,
    pub camera_dist: f32,
    pub radius: f32,

    #[default(Vec3::new(0.0, 0.0, 0.0))]
    pub color: Vec3,
    pub brightness: f32,
}

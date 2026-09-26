use anyhow::Result;

use cgmath::Deg;
use cgmath::InnerSpace;


use crate::Instance;
use crate::Device;
use crate::App;
use crate::AppData;
use crate::StringOrInt;
use crate::ModelClass;

use crate::scene::Scene;

use crate::common::Vec3;
use crate::common::Vec3d;
use crate::common::Mat4;



pub unsafe fn create_scene(instance: &Instance, device: &Device, data: &mut AppData) -> Result<Scene> {
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
        let cubes = Scene::load_model_into_memory(
            &mut scene,
            "models/Test_Cubes.glb",
            &instance, &device, data,
        )?;


        scene.add_model_to_scene(&magazine, ModelClass::Rigid, None);
        scene.add_model_to_scene(&magazine, ModelClass::Rigid, None);

        scene.scale_model(StringOrInt::Int(1), Vec3::new(5.0, 5.0, 5.0));
        scene.translate_model(StringOrInt::Int(1), Vec3::new(1.0, 6371000.0 * 1.00001 + 10.5, 0.0));

        scene.scale_model(StringOrInt::Int(0), Vec3::new(20.0, 5.0, 10.0));
        let rotation = Mat4::from_angle_z(Deg(90.0));
        scene.transform_model(StringOrInt::Int(0), rotation);
        scene.translate_model(StringOrInt::Int(0), Vec3::new(0.0, 6371000.0 * 1.00001 + 8.0, 0.0));

        // Merge magazines
        scene.set_blas_group(StringOrInt::Int(0), Some(0));
        scene.set_blas_group(StringOrInt::Int(1), Some(0));

        scene.add_model_to_scene(&protogen, ModelClass::Deformable, Some("Xenon".to_owned()));
        scene.translate_model(StringOrInt::Str("Xenon".to_owned()), Vec3::new(-4.0, 6371000.0 * 1.00001 + 10.0, 0.0));

        scene.add_model_to_scene(&room, ModelClass::Rigid, Some("Room".to_owned()));
        scene.translate_model(StringOrInt::Str("Room".to_owned()), Vec3::new(0.0, 6371000.0 * 1.00001 + 5.0, 8.0));
    
        scene.add_model_to_scene(&cubes, ModelClass::Rigid, Some("Cubes".to_owned()));
        scene.translate_model(StringOrInt::Str("Cubes".to_owned()), Vec3::new(0.0, 6371000.0 * 1.00001 + 10.0, -4.0));
        

        data.game_data.planets.push(Planet {
            position: Vec3d::new(0.0, 0.0, 0.0),
            radius: 6371000.0,
            atmosphere_radius: 100000.0,
            parent_star: 0,
            planet_color: Vec3::new(0.25, 1.0, 0.25),
            ..Default::default()
        });

        data.game_data.stars.push(Star {
            position: Vec3d::new(0.0, 0.0, 149600000000.0),
            radius: 695700000.0,
            brightness: 1.5e24,
            ..Default::default()
        });
    }

    Ok(scene)
}

pub fn render_tick(app: &mut App, _dt: f32) {
    let mag = app.camera.position.magnitude();
    
    let planet_radius = 6371000.0;
    let epsilon = 1.00001;
    if mag <= planet_radius * epsilon {
        let normalized = app.camera.position / mag;
        app.camera.position = (planet_radius * epsilon) * normalized;
    }
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

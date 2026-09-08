// Some aspects of code are not used yet here. They will be used for though
#![expect(dead_code)]

use cgmath::{Quaternion, SquareMatrix};


pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub _pad0: f32,
    pub normal: Vec3,
    pub _pad1: f32,
    pub uv: Vec2,

    pub joint_indices: [u16; 4],
    pub joint_weights: [f32; 4],
}

impl Vertex {
    pub const fn new(pos: Vec3, normal: Vec3, uv: Option<Vec2>, ji: Option<[u16; 4]>, jw: Option<[f32; 4]>) -> Self {
        let joint_indices = match ji {
            Some(ji) => ji,
            None => [0; 4],
        };
        let joint_weights = match jw {
            Some(jw) => jw,
            None => [0.0; 4],
        };
        
        match uv {
            Some(a) => Self {pos, _pad0: 0.0, normal, _pad1: 0.0, uv: a, joint_indices, joint_weights},
            None => Self {pos, _pad0: 0.0, normal, _pad1: 0.0, uv: Vec2::new(0.0, 0.0), joint_indices, joint_weights},
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub albedo: Vec3,
    pub albedo_texture_index: i32,

    pub metallic: f32,
    pub roughness: f32,

    pub emission: Vec3,

    pub transmission: f32,
    pub ior: f32,

    pub specular: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,

    pub sheen: f32,
    pub sheen_color: Vec3,
}


#[derive(Clone)]
pub struct Bone {
    pub node_index: usize,
    pub name: String,
    pub children: Vec<usize>,  // Indexes into Skeleton::bones
    pub local_transform: Mat4,
    pub inverse_bind_matrix: Mat4,
}

#[derive(Clone)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    pub root_bones: Vec<usize>,
}


#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub bone_index: usize,
    pub translations: Vec<(f32, Vec3)>,
    pub rotations: Vec<(f32, Quaternion<f32>)>,
    pub scales: Vec<(f32, Vec3)>,
}

impl AnimationChannel {
    fn sample_translation(&self, t: f32) -> Option<Vec3> {
        sample_track(&self.translations, t, |a, b, f| a + (b - a) * f)
    }

    fn sample_rotation(&self, t: f32) -> Option<Quaternion<f32>> {
        sample_track(&self.rotations, t, |a, b, f| a.nlerp(b, f))
    }

    fn sample_scale(&self, t: f32) -> Option<Vec3> {
        sample_track(&self.scales, t, |a, b, f| a + (b - a) * f)
    }
}


fn sample_track<T: Copy>(keys: &[(f32, T)], t: f32, lerp: impl Fn(T, T, f32) -> T) -> Option<T> {
    if keys.is_empty() {
        return None;
    }

    if t <= keys[0].0 {
        return Some(keys[0].1);
    }

    if t >= keys[keys.len() - 1].0 {
        return Some(keys[keys.len() - 1].1);
    }


    let i = keys.partition_point(|&(time, _)| time <= t);

    let (t0, v0) = keys[i - 1];
    let (t1, v1) = keys[i];

    let f = if t1 > t0 {
        (t - t0) / (t1 - t0)
    } else {
        0.0
    };

    Some(lerp(v0, v1, f))
}


#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

impl AnimationClip {
    pub fn sample_local_transforms(&self, skeleton: &Skeleton, t: f32) -> Vec<Mat4> {
        let mut locals: Vec<Mat4> = skeleton.bones.iter().map(|b| b.local_transform).collect();

        for channel in &self.channels {
            let translation = channel.sample_translation(t);
            let rotation = channel.sample_rotation(t);
            let scale = channel.sample_scale(t);

            let t_mat = match translation {
                Some(v) => Mat4::from_translation(v),
                None => Mat4::from_translation(cgmath::vec3(
                    skeleton.bones[channel.bone_index].local_transform.w.x,
                    skeleton.bones[channel.bone_index].local_transform.w.y,
                    skeleton.bones[channel.bone_index].local_transform.w.z,
                )),
            };
            let r_mat = rotation.map(Mat4::from).unwrap_or(cgmath::SquareMatrix::identity());
            let s_mat = match scale {
                Some(v) => Mat4::from_nonuniform_scale(v.x, v.y, v.z),
                None => cgmath::SquareMatrix::identity(),
            };

            locals[channel.bone_index] = t_mat * r_mat * s_mat;
        }

        locals
    }
}


pub struct AnimationPlayer {
    clips: std::collections::HashMap<String, AnimationClip>,
    current: Option<String>,
    time: f32,
    playing: bool,
    pub looping: bool,
    pub speed: f32,
}

impl AnimationPlayer {
    pub fn new(clips: Vec<AnimationClip>) -> Self {
        let clips = clips.into_iter().map(|c| (c.name.clone(), c)).collect();
        Self { clips, current: None, time: 0.0, playing: false, looping: true, speed: 1.0 }
    }

    pub fn play(&mut self, name: &str) {
        if !self.clips.contains_key(name) {
            log::warn!("Animation '{}' not found on this model", name);
            return;
        }

        self.current = Some(name.to_string());
        self.time = 0.0;
        self.playing = true;
    }

    pub fn play_fuzzy(&mut self, query: &str) {
        let query_lower = query.to_lowercase();
        let matched = self.clips.keys()
            .find(|name| name.to_lowercase().contains(&query_lower))
            .cloned();

        match matched {
            Some(name) => self.play(&name),
            None => log::warn!("No animation matching '{}', available: {:?}", query, self.list_clips()),
        }
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.time = 0.0;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn resume(&mut self) {
        if self.current.is_some() {
            self.playing = true;
        }
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn current_clip_name(&self) -> Option<&str> {
        self.current.as_deref()
    }

    pub fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let Some(name) = &self.current else { return };
        let Some(clip) = self.clips.get(name) else { return };

        self.time += dt * self.speed;

        if self.time > clip.duration {
            if self.looping {
                self.time %= clip.duration.max(0.0001);
            } else {
                self.time = clip.duration;
                self.playing = false;
            }
        }
    }

    pub fn sample(&self, skeleton: &Skeleton) -> Vec<Mat4> {
        let locals: Vec<Mat4> = match self.current.as_ref().and_then(|n| self.clips.get(n)) {
            Some(clip) if self.current.is_some() => clip.sample_local_transforms(skeleton, self.time),
            _ => skeleton.bones.iter().map(|b| b.local_transform).collect(),
        };

        let mut world = vec![Mat4::identity(); skeleton.bones.len()];
        for &root in &skeleton.root_bones {
            propagate_world_transform(skeleton, &locals, root, Mat4::identity(), &mut world);
        }

        world
            .iter()
            .zip(skeleton.bones.iter())
            .map(|(w, b)| w * b.inverse_bind_matrix)
            .collect()
    }

    pub fn print_clips(&self, model_label: &str) {
        if self.clips.is_empty() {
            log::info!("[{}] no animations found", model_label);
            return;
        }
        log::info!("[{}] animations:", model_label);
        for (name, clip) in &self.clips {
            log::info!("  - \"{}\" ({:.2}s, {} channels)", name, clip.duration, clip.channels.len());
        }
    }
    
    pub fn list_clips(&self) -> Vec<&str> {
        self.clips.keys().map(|s| s.as_str()).collect()
    }
}


fn propagate_world_transform(
    skeleton: &Skeleton,
    locals: &[Mat4],
    bone_index: usize,
    parent_world: Mat4,
    out: &mut [Mat4],
) {
    let world = parent_world * locals[bone_index];
    out[bone_index] = world;

    for &child in &skeleton.bones[bone_index].children {
        propagate_world_transform(skeleton, locals, child, world, out);
    }
}

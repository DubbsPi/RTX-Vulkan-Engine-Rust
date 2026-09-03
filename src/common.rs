pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
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
            Some(a) => Self {pos, normal, uv: a, joint_indices, joint_weights},
            None => Self {pos, normal, uv: Vec2::new(0.0, 0.0), joint_indices, joint_weights},
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
    pub _pad1: [f32; 2],
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
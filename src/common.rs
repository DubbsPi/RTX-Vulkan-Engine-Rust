pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

impl Vertex {
    pub const fn new(pos: Vec3, normal: Vec3, uv: Option<Vec2>) -> Self {
        match uv {
            Some(a) => Self {pos, normal, uv: a},
            None => Self {pos, normal, uv: Vec2::new(0.0, 0.0)},
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

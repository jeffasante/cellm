//! cellm-kernels: CPU SIMD, Metal compute shaders, and Vulkan.

pub mod cpu;
pub mod cpu_kernels;
pub mod metal;
pub mod vulkan;

pub use cpu::SIMDKernels;
pub use metal::MetalKernels;
pub use vulkan::VulkanKernels;

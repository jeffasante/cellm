// Author: Jeffrey Asante (https://jeffasante.github.io/)
#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalGraph {
    pub device: Device,
    pub queue: CommandQueue,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalGraph {
    pub fn new() -> anyhow::Result<Self> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| anyhow::anyhow!("Metal graph: no device found"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }
}

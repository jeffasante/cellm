// Author: Jeffrey Asante (https://jeffasante.github.io/)
use metal::{Device, MTLResourceOptions};
use memmap2::Mmap;

fn main() {
    let device = Device::system_default().unwrap();
    let f = std::fs::File::open("../../models/README.md").unwrap();
    let mmap = unsafe { Mmap::map(&f).unwrap() };
    let ptr = mmap.as_ptr() as *mut std::ffi::c_void;
    let len = mmap.len() as u64;

    // Align length to page size (mmap does this physically, but MTLCore requires len to be explicitly aligned sometimes)
    println!("Mmap ptr: {:p}, len: {}", ptr, len);

    let buf = device.new_buffer_with_bytes_no_copy(
        ptr,
        len,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    println!("Successfully mapped mmap to Metal buffer: {:?}", buf.length());
}

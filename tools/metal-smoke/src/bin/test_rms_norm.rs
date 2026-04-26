// Author: Jeffrey Asante (https://jeffasante.github.io/)
use cellm_kernels::MetalOps;

fn main() {
    let mut ops = MetalOps::create().unwrap();
    let n = 640;
    let mut x = vec![1.0f32; n];
    let w = vec![0u16; n]; // if w is offset, base is 0.0, we add +1
    let eps = 1e-6;
    let mut out = vec![0.0f32; n];
    
    // CPU reference
    let ss: f32 = x.iter().map(|v| v*v).sum::<f32>() / n as f32 + eps;
    let inv_rms = 1.0 / ss.sqrt();
    let mut cpu_out = vec![0.0f32; n];
    for (i, v) in x.iter().enumerate() {
        cpu_out[i] = v * inv_rms * (0.0 + 1.0);
    }
    
    ops.rms_norm_f16w(&x, &w, eps, true, "test_rms_norm", &mut out).unwrap();
    
    println!("CPU[0..5]: {:?}", &cpu_out[0..5]);
    println!("GPU[0..5]: {:?}", &out[0..5]);
    
    // Test a second dispatch with a different size to catch the buffer allocation bug
    let n2 = 64;
    let mut x2 = vec![2.0f32; n2];
    let w2 = vec![0u16; n2];
    let mut out2 = vec![0.0f32; n2];
    ops.rms_norm_f16w(&x2, &w2, eps, false, "test_rms_norm_2", &mut out2).unwrap();
    
    let ss2: f32 = x2.iter().map(|v| v*v).sum::<f32>() / n2 as f32 + eps;
    let inv_rms2 = 1.0 / ss2.sqrt();
    let mut cpu_out2 = vec![0.0f32; n2];
    for (i, v) in x2.iter().enumerate() {
        cpu_out2[i] = v * inv_rms2 * (0.0);
    }
    
    println!("CPU2[0..5]: {:?}", &cpu_out2[0..5]);
    println!("GPU2[0..5]: {:?}", &out2[0..5]);
}

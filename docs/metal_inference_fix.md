# Fixing Metal Inference for Qwen 2.5

## The Problem

Running Qwen 2.5 on the Metal (GPU) backend produced total garbage. The model either crashed with a `DIAG_STOP` error or output nonsense like repeating Chinese characters. The same model worked perfectly on the CPU backend.

## How It Was Solved

The fix came down to four separate bugs, found one at a time by comparing what the GPU was computing against what the CPU was computing at each stage of the model.

---

### Bug 1: The RMSNorm shader was a placeholder

**What RMSNorm does:** Before each layer of the model, the hidden state (a big list of numbers) gets normalized. Think of it like adjusting the volume on a signal so it stays in a reasonable range.

**What was wrong:** The Metal shader code that was supposed to do this normalization was actually a stub left over from debugging. Instead of computing the real norm, it just copied the first number of the input into every slot of the output. So every element came out identical.

**The fix:** Replaced the stub with a real parallel-reduction RMSNorm. The GPU threads cooperate to sum up the squares of all elements, compute the inverse root-mean-square, then multiply each element by its weight. Standard textbook implementation.

**File:** [metal.rs](file:///Users/jeff/Desktop/cellm/crates/cellm-kernels/src/metal.rs)

---

### Bug 2: A CPU copy was destroying the residual stream

**What the residual stream is:** In a transformer, the model keeps a running total (the "residual") that each layer adds its contribution to. The formula is `x = x + layer_output(normalize(x))`. You normalize a copy, process it, then add the result back to the original. The original must be preserved.

**What was wrong:** In the middle of encoding GPU commands, there was a line of CPU code that copied the normalized values back over the original:

```rust
// LEAK x_norm to x_buf
unsafe { std::ptr::copy_nonoverlapping(
    self.x_norm_buf.contents() as *const f32,
    self.x_buf.contents() as *mut f32, h
); }
```

This was bad for two reasons. First, it ran on the CPU immediately, but the GPU had not actually executed yet (GPU commands are queued up and run later). So it was reading stale data from a previous run, or zeros. Second, even if the timing were right, overwriting x with normalize(x) breaks the residual connection. The model's whole architecture depends on preserving that original x.

**The fix:** Deleted the line. The GPU handles both buffers correctly through the encoded command sequence.

**File:** [qwen.rs](file:///Users/jeff/Desktop/cellm/crates/cellm-model/src/qwen.rs)

---

### Bug 3: Debug code was blocking execution

**What was wrong:** The `step_fused` function had leftover diagnostic code that ran only layer 0 on the GPU, printed some values, then returned an error called `DIAG_STOP`. This meant the model never got past the first layer.

There were also several `println!` statements scattered around both the CPU and GPU paths that dumped internal values to the console on every call.

**The fix:** Removed all the `println!` diagnostics, the early return, and the `DIAG_STOP` error.

**File:** [qwen.rs](file:///Users/jeff/Desktop/cellm/crates/cellm-model/src/qwen.rs)

---

### Bug 4: Bias values were being read as the wrong data type

This was the hardest to find and the most damaging.

**What bias values are:** After a matrix multiply (like projecting the hidden state into query/key/value vectors), the model adds a small per-element offset called the bias. These are small numbers, usually less than 1.0.

**What was wrong:** The bias tensors were stored in the model file as f16 (16-bit floats, 2 bytes each). When the GPU buffers were loaded, all tensors were uploaded as raw bytes. The GPU shader that adds the bias declared its bias parameter as `device const float*` (32-bit floats, 4 bytes each). So when the shader read the bias, it was grabbing 4 bytes at a time from data that was packed as 2-byte values.

This means every "float" the shader read was actually two f16 values smashed together and interpreted as one f32. The result was values like `-12,938,306,000` where the correct answer was `1.95`.

**How it was found:** By running the RMSNorm alone on the GPU and comparing to the CPU, it matched perfectly. Then running RMSNorm + QKV projections, the first few output values matched but others were wildly off. Running the same weights through the CPU-side i8 matmul (without bias) produced correct results. That narrowed it to the bias addition.

Since biases are present (`bq=true`) and stored as f16, but the shader reads them as float32, the data gets garbled. Values that happened to have zero-ish f16 neighbors looked roughly right, but most were catastrophically wrong.

**The fix:** During weight preloading, detect tensors whose names end in `.bias` and convert them from f16 to f32 before uploading to the GPU:

```rust
if name.ends_with(".bias") {
    let f16_data: &[u16] = bytemuck::cast_slice(bytes);
    let f32_data: Vec<f32> = f16_data.iter()
        .map(|&b| f16::from_bits(b).to_f32()).collect();
    let f32_bytes = bytemuck::cast_slice::<f32, u8>(&f32_data);
    gs.preload_weight(name.clone(), f32_bytes);
} else {
    gs.preload_weight(name.clone(), bytes);
}
```

**File:** [qwen.rs](file:///Users/jeff/Desktop/cellm/crates/cellm-model/src/qwen.rs)

---

## The Debugging Approach

The key technique was **incremental isolation**. The model is a pipeline of stages:

```
Input -> RMSNorm -> QKV Projection -> RoPE -> Attention -> O Projection -> MLP -> Output
```

By stopping the GPU after each stage, reading the buffer back to the CPU, and comparing against the known-working CPU path, the exact point of divergence was found:

1. RMSNorm: matched (after fixing bug 1)
2. QKV Projection: diverged at specific elements

From there, the matmul itself was correct (verified by running the CPU i8 matmul with the same GPU-resident weights), so the problem had to be in the bias addition. Checking the data types confirmed the f16/f32 mismatch.

## Result

The Metal backend now produces output identical to the CPU backend:

```
Sycophancy is the behavior of a person who is overly flattering
or flattering others in order to gain favor or respect from them.
```

Decode speed: 64 tokens in 1.25s on Metal.

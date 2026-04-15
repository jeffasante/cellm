import os

def fix_file(path, search, replace):
    if not os.path.exists(path): return
    with open(path, 'r') as f: content = f.read()
    if search in content:
        with open(path, 'w') as f: f.write(content.replace(search, replace))

# Fix metal.rs (restore missing read_f32 calls)
search_metal = """        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_qkv_f16(
                &enc,
                wq,
                wk,
                wv,
                xb,
                &q_buf,
                &k_buf,
                &v_buf,
                rows_q,
                rows_k,
                cols,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

    pub fn logits_mv2_i8("""

replace_metal = """        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_qkv_f16(
                &enc,
                wq,
                wk,
                wv,
                xb,
                &q_buf,
                &k_buf,
                &v_buf,
                rows_q,
                rows_k,
                cols,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(&q_buf, q_out)?;
        read_f32(&k_buf, k_out)?;
        read_f32(&v_buf, v_out)?;
        Ok(())
    }

    pub fn logits_mv2_i8("""

fix_file('crates/cellm-kernels/src/metal.rs', search_metal, replace_metal)

# Fix Qwen attention
search_qwen = "cr.attention_single_token_gqa_from_bases("
replace_qwen = "cr.attention_single_token_gqa_from_bases(&q, &k_bases, &v_bases, cfg.rope_theta, head_dim, None, &mut attn_out)?;"
# Actually I'll use a more precise search for qwen

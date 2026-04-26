// Author: Jeffrey Asante (https://jeffasante.github.io/)
use cellm_cache::{KVCache, PageTable, BlockAllocator, kvcache::KvStorageKind};
use cellm_core::kv_cache::KvCacheLayout;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_layers = 1;
    let n_heads = 14;
    let num_kv_heads = 2;
    let head_dim = 64;
    let seq = 1024;
    let tokens_per_block = 16;
    let total_blocks = 1024;

    let layout = KvCacheLayout {
        total_blocks,
        tokens_per_block,
        num_layers,
        num_kv_heads,
        head_dim,
    };

    // 1. Metal Cache (Optimized)
    let mutter_result = KVCache::new_with_kind(layout.clone(), KvStorageKind::Metal);
    if let Err(e) = mutter_result {
        println!("Skipping Metal benchmark: {}", e);
        return Ok(());
    }
    let mut metal_cache = mutter_result.unwrap();
    let mut page_table = PageTable::new(1, tokens_per_block)?;
    let mut alloc = BlockAllocator::new(total_blocks);
    page_table.append_tokens(&mut alloc, seq)?;

    let mut k = vec![0.0f32; num_kv_heads * head_dim];
    let mut v = vec![0.0f32; num_kv_heads * head_dim];
    let mut bases = Vec::new();

    {
        let mut cv = metal_cache.view_mut();
        for t in 0..seq {
            let block_id = page_table.block_for_token(t)?;
            let off = page_table.offset_in_block(t)?;
            for i in 0..k.len() {
                k[i] = (t as f32 + i as f32) / 1000.0;
                v[i] = (t as f32 - i as f32) / 1000.0;
            }
            cv.write_token(block_id, 0, off, &k, &v)?;
            bases.push(cv.layout.token_base_elem(block_id, 0, off)?);
        }
    }

    let q = vec![0.1f32; n_heads * head_dim];
    let mut out_metal = vec![0.0f32; n_heads * head_dim];

    // Warmup
    for _ in 0..10 {
        metal_cache.view().attention_single_token_gqa_from_bases(
            &bases,
            &q,
            n_heads,
            num_kv_heads,
            head_dim,
            None,          // attn_scale (default)
            None,          // soft_cap (none)
            &mut out_metal,
        )?;
    }

    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        metal_cache.view().attention_single_token_gqa_from_bases(
            &bases,
            &q,
            n_heads,
            num_kv_heads,
            head_dim,
            None,
            None,
            &mut out_metal,
        )?;
    }
    let duration = start.elapsed() / iters as u32;
    println!(
        "Metal Attention (seq={seq}): {:.2}µs",
        duration.as_secs_f64() * 1_000_000.0
    );

    // 2. CPU Cache (Reference)
    let mut cpu_cache = KVCache::new_with_kind(layout, KvStorageKind::Cpu)?;
    {
        let mut cv = cpu_cache.view_mut();
        for t in 0..seq {
            let block_id = page_table.block_for_token(t)?;
            let off = page_table.offset_in_block(t)?;
            for i in 0..k.len() {
                k[i] = (t as f32 + i as f32) / 1000.0;
                v[i] = (t as f32 - i as f32) / 1000.0;
            }
            cv.write_token(block_id, 0, off, &k, &v)?;
        }
    }

    let mut out_cpu = vec![0.0f32; n_heads * head_dim];
    cpu_cache.view().attention_single_token_gqa_from_bases(
        &bases,
        &q,
        n_heads,
        num_kv_heads,
        head_dim,
        None,
        None,
        &mut out_cpu,
    )?;

    let mut max_diff = 0.0f32;
    for i in 0..out_cpu.len() {
        let diff = (out_cpu[i] - out_metal[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("Max divergence vs CPU: {:.6}", max_diff);

    Ok(())
}

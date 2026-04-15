#!/bin/bash
# 1. Gemma attention
sed -i '' '/cr.attention_single_token_gqa_from_bases(/,/attn_out_slice,/ { s/head_dim,/head_dim, None,/; }' crates/cellm-model/src/gemma.rs
# 2. Llama attention
sed -i '' '/cr.attention_single_token_gqa_from_bases(/,/&mut attn_out,/ { s/head_dim,/head_dim, None,/; }' crates/cellm-model/src/llama.rs
# 3. Llama graph encode_attention
sed -i '' 's/head_dim as u32)/head_dim as u32, None)/' crates/cellm-model/src/llama_graph.rs

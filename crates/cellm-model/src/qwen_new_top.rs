    pub fn step_topk(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let mut x = vec![0.0f32; self.cfg.hidden_size];
        self.embed_token(token, &mut x)?;
        let logits = self.step_inner(&x, pos, page_table, kv_cache, true)?;
        Ok(self.top_k_logits(&logits, top_k))
    }

    pub fn prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let mut x = vec![0.0f32; hidden];
            self.embed_token(tok, &mut x)?;
            self.step_inner(&x, pos, page_table, kv_cache, false)?;
        }
        Ok(())
    }

    fn step_inner(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        return_logits: bool,
    ) -> Result<Vec<f32>, CoreError> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / n_heads;
        let rotary_dim = ((head_dim as f32) * self.partial_rotary_factor) as usize;

        // Lazy session state init.
        let session_id = page_table.session_id();
        let sess = self.sessions
            .entry(session_id)
            .or_insert_with(|| QwenSessionState {
                linear: vec![],
                graph_state: None,
            });

        if sess.graph_state.is_none() && self.metal_ops.is_some() && self.metal_strict {
            let mut gs = QwenGraphState::new(
                hidden,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.vocab_size,
                cfg.intermediate_size,
                self.metal_ops.clone().unwrap(),
            )?;
            for (name, bytes) in self.file.all_tensors() {
                gs.preload_weight(name.clone(), bytes);
            }
            sess.graph_state = Some(gs);
        }

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("qwen step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "qwen step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table block_for_token failed: {e}"))
        })? as u32;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table offset_in_block failed: {e}"))
        })?;

        // x = embedding(token)
        let mut x = x0.to_vec();

        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; n_kv_heads * head_dim];
        let mut v = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; hidden];
        let mut attn_proj = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];
        let mut gather_bases = Vec::new();

        let prefix = if has_tensor_file(&self.file, "language_model.model.embed_tokens.weight") {
            "language_model."
        } else {
            ""
        };

        if let Some(graph_state) = &mut sess.graph_state {
            let res = graph_state.step_fused(
                &x,
                &cfg,
                prefix,
                kv_cache,
                page_table,
                pos,
                token_off,
                block_id,
                self.partial_rotary_factor,
                self.rmsnorm_weight_is_offset,
                return_logits,
            )?;
            return Ok(res.unwrap_or_else(Vec::new));
        }

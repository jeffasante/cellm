# cellm Inference Graph

This document visualizes the computation graph implemented in `crates/cellm-model/src/llama_graph.rs`. The graph is optimized for **Metal** execution on Apple Silicon, fusing layer operations into a single command buffer dispatch.

```mermaid
graph TD
    subgraph Input ["Step 1: Input & Embeddings"]
        Tokens["Token Sequence (IDs)"] --> Embeds["Embedding Lookup (F16 weights)"]
        Embeds --> X["X Buffer (Activation Space)"]
    end

    subgraph LayerLoop ["Step 2: Transformer Block (Repeated N times)"]
        X --> RMS1["RMSNorm (Input Layernorm)"]
        
        subgraph Attention ["Self-Attention Stack"]
            RMS1 --> QKV_Proj["Q, K, V Projections (MV F16)"]
            QKV_Proj --> RoPE["Rotary Positional Embeddings (RoPE)"]
            RoPE --> KV_Write["Write to Paged KV Cache"]
            KV_Write --> SDPA["Scaled Dot-Product Attention"]
            SDPA --> O_Proj["O Projection (MV F16)"]
        end

        O_Proj --> Res1["Residual Add 1 (X + O_Proj)"]
        Res1 --> RMS2["RMSNorm (Post-Attention Layernorm)"]

        subgraph FeedForward ["MLP Stack"]
            RMS2 --> GateUp["Gate & Up Projections (MV F16)"]
            GateUp --> SiLU["SiLU Masking & Element-wise Mul"]
            SiLU --> Down_Proj["Down Projection (MV F16)"]
        end

        Down_Proj --> Res2["Residual Add 2 (X + MLP_Out)"]
        Res2 -- "Loop for Layer N+1" --> X
    end

    subgraph Output ["Step 3: Final Output Projection"]
        Res2 --> FinalNorm["Final RMSNorm"]
        FinalNorm --> LMHead["LM Head Projection (Logits)"]
        LMHead --> Logits["Softmax & Sampling"]
    end

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style LayerLoop fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Output fill:#f9f,stroke:#333,stroke-width:2px
    style KV_Write fill:#fff9c4,stroke:#fbc02d
    style SDPA fill:#fff9c4,stroke:#fbc02d
```

### Execution Strategy: `step_fused`
- **CPU Preparation**: Maps the `PageTable` to a buffer of block indices/offsets once per sequence.
- **Metal Command Buffer**: All layers are encoded into a single `MTLCommandBuffer`.
- **Zero-Copy Activations**: Intermediate buffers (Q, K, V, MLP) are pre-allocated in `LlamaGraphState` and reused across layers to minimize memory pressure.
- **Synchronization**: `cb.wait_until_completed()` is called exactly once after the entire model pass, ensuring maximum GPU occupancy.

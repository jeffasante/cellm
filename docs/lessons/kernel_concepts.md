# Lesson: High-Performance GPU Kernels

This lesson breaks down six critical concepts for understanding how modern LLM inference engines (like **cellm**) achieve high performance on mobile GPUs.

---

## 1. Flash Attention
Standard attention computes a full $N \times N$ score matrix in global memory (HBM). Flash Attention uses **tiling** and **online-softmax** to keep memory usage linear in sequence length.

```mermaid
graph LR
    subgraph Naive ["Naive Attention"]
        M1["Full Seq x Seq Matrix (HBM)"]
        M1 --> S1["O(N²) Memory"]
    end

    subgraph Flash ["Flash Attention"]
        T1["Tile 1"] -- "Sum/Max" --> Acc["Running Stats"]
        T2["Tile 2"] -- "Update" --> Acc
        Acc --> S2["O(N) Memory"]
    end

    style Naive fill:#fee2e2,stroke:#ef4444
    style Flash fill:#dcfce7,stroke:#22c55e
```

---

## 2. Tiled Execution
To minimize slow trips to device memory (HBM), we load blocks of data into **threadgroup shared memory** (SRAM). Computation happens in fast registers before results are written back.

```mermaid
graph TD
    HBM[Device Memory / HBM] -- "Load Tile" --> SRAM[Shared Memory / SRAM]
    SRAM -- "Registers" --> ALU[GPU Core / ALU]
    ALU -- "Accumulate" --> SRAM
    SRAM -- "Store Result" --> HBM

    style SRAM fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
```

---

## 3. GQA Mapping
Grouped Query Attention (GQA) fans Query heads down to shared KV heads. If we have 8 Q heads and 4 KV heads ($G=2$), the implementation is a simple integer division.

**Formula**: `kv_head = q_head / G`

```mermaid
graph LR
    Q0[Q0] --> G0((Group 0))
    Q1[Q1] --> G0
    Q2[Q2] --> G1((Group 1))
    Q3[Q3] --> G1
    
    G0 --> KV0[KV Head 0]
    G1 --> KV1[KV Head 1]

    style KV0 fill:#f9f,stroke:#333
```

---

## 4. Grid Shape (3D Dispatch)
The GPU dispatches a 3D grid of threadgroups. This allows the hardware to schedule work freely across all available compute units.

*   **X Axis**: Q-Tiles (Sequence blocks)
*   **Y Axis**: Attention Heads
*   **Z Axis**: Batch Items (Independent sequences)

```mermaid
graph TD
    subgraph Grid ["GPU Dispatch Grid"]
        Z1["Batch Index"] 
        Z1 --> Y1["Head Index"]
        Y1 --> X1["Tile Index (Q-Block)"]
    end

    X1 -- "Kernel Call" --> Work["Threadgroup Execution"]
```

---

## 5. Causal Masking
In autoregressive decoding, tokens cannot "look ahead." We apply a lower-triangular mask by setting scores above the diagonal to $-\infty$. During softmax, $e^{-\infty}$ becomes 0.

```mermaid
graph TD
    subgraph Matrix ["Causal Score Matrix"]
        M1["S[0,0]"]  --> M2["-inf"]
        M3["S[1,0]"]  --> M4["S[1,1]"]
    end
    
    Softmax["Softmax Layer"]
    Matrix --> Softmax
    Softmax --> Out["[0.4, 0.0]\n[0.2, 0.8]"]
```

---

## 6. Variable-Length Sequences
Batches often contain sequences of different lengths. We use a `kv_lens` array to inform the kernel where the "real" data ends. The tile loader pads the rest with zeros so the GPU can process the batch in one go.

```mermaid
graph LR
    B1["Seq 1 (len: 12)"] --> P1["[Data] [000]"]
    B2["Seq 2 (len: 15)"] --> P2["[Data]"]
    
    KVLens["kv_lens[batch_idx]"] -- "Bounds Check" --> Kernel["Unified Kernel Dispatch"]

    style KVLens fill:#e1f5fe,stroke:#01579b
```

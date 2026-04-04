# cellm Project Architecture

This document provides a high-level overview of the **cellm** engine architecture, showing how the various crates and components interact to provide multi-session LLM inference on mobile devices.

## System Overview

The engine follows a layered architecture designed for extreme memory efficiency and predictable performance on memory-constrained devices.

```mermaid
graph TD
    subgraph PublicAPI ["Public API (crates/cellm-sdk)"]
        Engine["Engine (Global Manager)"]
        Session["EngineSession (Per User)"]
    end

    subgraph Scheduling ["Scheduling (crates/cellm-scheduler)"]
        RR["RoundRobinScheduler"]
        TP["ThermalPolicy"]
    end

    subgraph Memory ["Memory & Cache (crates/cellm-cache)"]
        KVC["KVCache (Shared)"]
        BA["BlockAllocator"]
        PT["PageTable (Per Session)"]
    end

    subgraph Inference ["Inference Logic (crates/cellm-model)"]
        Runner["Runner (Llama / Gemma / Qwen)"]
        Graph["LlamaGraphState (Metal Fused)"]
        Format["CellmFile (Mmap weight access)"]
    end

    subgraph Kernels ["Compute Kernels (crates/cellm-kernels)"]
        Metal["Metal Kernels (MSL)"]
        CPU["CPU Kernels (SIMD)"]
    end

    %% Relationships
    Engine -- "owns" --> KVC
    Engine -- "owns" --> Runner
    Engine -- "manages" --> Session
    Engine -- "schedules with" --> RR

    Session -- "owns" --> PT
    PT -- "references" --> BA

    Runner -- "uses" --> Graph
    Runner -- "loads from" --> Format

    Graph -- "dispatches" --> Metal
    Graph -- "falls back to" --> CPU
    
    Graph -- "reads/writes" --> KVC
```

### Key Components

#### 1. The `Engine` (`cellm-sdk`)
The primary entry point for mobile integrations (iOS/Android). It manages the global state:
- Shared **KVCache** for all active sessions.
- The active **Model Runner**.
- The **Scheduler** for interleaved decoding.

#### 2. Paged KV Cache (`cellm-cache`)
Inspired by vLLM but optimized for mobile:
- **BlockAllocator**: Manages a fixed pool of memory blocks (pages) to prevent fragmentation.
- **PageTable**: Maps logical token positions to physical blocks in the cache. This allows sessions to grow dynamically without contiguous memory allocations.

#### 3. Fused Inference Graph (`cellm-model`)
Instead of issuing many small GPU commands, **cellm** uses a "Graph" approach:
- Compiles the entire model pass into a single Metal command buffer.
- Minimizes CPU-to-GPU synchronization by waiting only at the end of the full forward pass.

#### 4. Thermal Policy (`cellm-scheduler`)
Monitors device health and adjusts the scheduler frequency.
- Can pause background sessions if the device reaches critical temperature.
- Prioritizes completion of active user prompts.

## Data Flow (Single Token Decode)

```mermaid
sequenceDiagram
    participant App as Mobile App
    participant E as Engine
    participant S as Scheduler
    participant R as Runner
    participant M as Metal Graph

    App->>E: step_decode()
    E->>S: Get next scheduled session
    S-->>E: Session ID 42
    E->>R: step_topk(last_token, session_42_page_table)
    R->>M: Dispatch fused LlamaGraph pass
    M-->>R: Logits
    R-->>E: Top-K candidates
    E->>E: Sample (Temperature/Penalty)
    E-->>App: (SessionID, NextTokenID)
```

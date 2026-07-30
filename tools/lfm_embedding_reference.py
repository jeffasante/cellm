#!/usr/bin/env python3
"""Emit reference embeddings for LFM2.5-Embedding-350M as JSON.

Used as the ground truth for the Rust encoder parity test. Runs the HF
Lfm2BidirectionalModel in float32 and applies CLS pooling + L2 normalize,
matching the SentenceTransformer config shipped with the model.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital and most populous city of France.",
    "document: The mitochondrion is the powerhouse of the cell.",
    "query: how do I reset my password?",
]


def main(model_dir: Path, out_path: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32
    ).eval()

    records = []
    for text in TEXTS:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**enc).last_hidden_state  # [1, seq, 1024]
        cls = out[0, 0]
        vec = torch.nn.functional.normalize(cls, p=2, dim=-1)
        records.append(
            {
                "text": text,
                "tokens": enc["input_ids"][0].tolist(),
                "embedding": [round(float(v), 6) for v in vec.tolist()],
            }
        )
        print(f"{text[:48]!r:52} tokens={len(records[-1]['tokens'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=1))
    print(f"wrote {out_path}")

    # Sanity: the matching query/document pair should out-score the distractor.
    import itertools

    for a, b in itertools.combinations(range(len(records)), 2):
        va = torch.tensor(records[a]["embedding"])
        vb = torch.tensor(records[b]["embedding"])
        print(f"cos({a},{b}) = {float(va @ vb):.4f}")


if __name__ == "__main__":
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "models/hf/LFM2.5-Embedding-350M"
    )
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "tests/data/lfm_embedding_reference.json"
    )
    main(model_dir, out_path)

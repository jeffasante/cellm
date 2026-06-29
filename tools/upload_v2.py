#!/usr/bin/env python3
"""Upload LFM2.5-230M-int4-v2.cellm to Hugging Face with a long timeout."""

import os

from huggingface_hub import HfApi

api = HfApi()

local_path = "/Users/jeff/Desktop/cellm/models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4-v2.cellm"
repo_id = "jeffasante/cellm-models"
path_in_repo = "LFM2.5-230M/LFM2.5-230M-int4-v2.cellm"

print(f"Uploading {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)...")
print(f"  -> {repo_id}/{path_in_repo}")

api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model",
)

print("Upload complete!")

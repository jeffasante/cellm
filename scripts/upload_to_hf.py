from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "jeffasante/cellm-models"
base_dir = "models/to-huggingface"

missing_models = [
    "lfm2.5-350m-v1",
    "qwen3-0.6b-v1",
    "qwen3.5-0.8b-v1",
    "smollm2-360m-q1-v1",
    "smolvlm-256m-instruct-f16-full"
]

for model in missing_models:
    model_path = os.path.join(base_dir, model)
    if os.path.exists(model_path):
        print(f"Uploading {model}...")
        try:
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                path_in_repo=model,
                repo_type="model"
            )
            print(f"Successfully uploaded {model}")
        except Exception as e:
            print(f"Error uploading {model}: {e}")
    else:
        print(f"Error: {model_path} does not exist")

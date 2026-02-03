import torch
import time

device='cuda:2'
def main():
    l =[]
    for i in range(100):
        print('Iteration', i, 'Memory allocated:', torch.cuda.memory_allocated(device))
        time.sleep(3)
        l.append(torch.randn(100, 100).to(device))

if __name__ == '__main__':
    main()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_path = "/home/zulfiyausmonova/models/llama-3.1-8b-instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb,
)

inputs = tokenizer("Explain what semantic drift is in one sentence.", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(out[0], skip_special_tokens=True))








import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Choose a GPU like your supervisor's example
DEVICE_STR = "cuda:2"
DEVICE = torch.device(DEVICE_STR)

# Optional but strongly recommended:
# If you want to make the process see ONLY GPU 2 as "cuda:0", uncomment this line
# (must be set before torch/transformers touches CUDA)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_PATH = "/home/zulfiyausmonova/models/llama-3.1-8b-instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Explicit placement:
    # - If you keep CUDA_VISIBLE_DEVICES unset, use device_map={"": DEVICE_STR}
    # - If you set CUDA_VISIBLE_DEVICES="2", then use device_map="cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map={"": DEVICE_STR},  # put the whole model on cuda:2
    )

    prompt = "Explain what semantic drift is in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs explicitly to the same device as the model
    # (for quantized models, model.device may not always be reliable; using DEVICE_STR is safe)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=40)

    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Optional sanity prints (remove later)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Visible GPUs:", torch.cuda.device_count())
        print("Running on:", DEVICE_STR, torch.cuda.get_device_name(DEVICE))
    main()





# #!/bin/bash
# #SBATCH --job-name=download_big_models
# #SBATCH --partition=a100_only
# #SBATCH --gres=gpu:0
# #SBATCH --cpus-per-task=6
# #SBATCH --mem=64G
# #SBATCH --time=12:00:00
# #SBATCH --output=/home/zulfiyausmonova/projects/llm_chain/logs/%x_%j.out
# #SBATCH --error=/home/zulfiyausmonova/projects/llm_chain/logs/%x_%j.err

# set -euo pipefail

# PROJECT_DIR="/home/zulfiyausmonova/projects/llm_chain"

# # CHOOSE ONE (recommended: /mnt/disk1 for local ext4)
# # BASE="/mnt/disk1/zulfiyausmonova"
# BASE="/mnt/h200_raid5/zulfiyausmonova"

# MODEL_DIR="$BASE/models"
# HF_CACHE_DIR="$BASE/hf_cache"

# mkdir -p "$MODEL_DIR" "$HF_CACHE_DIR"

# # Make huggingface cache use the big disk too
# export HF_HOME="$HF_CACHE_DIR"
# export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
# export HF_HUB_CACHE="$HF_CACHE_DIR/hub"

# cd "$PROJECT_DIR"
# source "$PROJECT_DIR/venv/bin/activate"

# python - <<'PY'
# import os
# from huggingface_hub import snapshot_download

# token = os.environ.get("HF_TOKEN")
# if not token:
#     raise RuntimeError('HF_TOKEN is not set. Submit like: HF_TOKEN="..." sbatch download_big_models.sbatch')

# base_dir = os.environ["MODEL_DIR"]

# models = {
#     "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b-instruct",
#     "google/gemma-3-27b-it": "gemma-3-27b-it",
# }

# for repo_id, folder in models.items():
#     print(f"\n=== Downloading {repo_id} ===", flush=True)
#     snapshot_download(
#         repo_id=repo_id,
#         local_dir=f"{base_dir}/{folder}",
#         token=token,
#     )
#     print(f"=== Finished {repo_id} ===", flush=True)
# PY

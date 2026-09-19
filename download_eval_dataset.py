from huggingface_hub import snapshot_download

repo_id = "Tensor-Link/cascade-eval-pool"

local_download = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="./cascade-eval-pool",
)

print(f"Downloaded to: {local_download}")
from huggingface_hub import snapshot_download

repo_id = "tonybilling/gen-64a0412cd332"

local_download = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir="./generators/tonybilling/gen-64a0412cd332",
)

print(f"Downloaded to: {local_download}")
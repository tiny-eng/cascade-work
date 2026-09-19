from hippius_hub import snapshot_download

repo_id = "cascade/ckpt-r15703366466408532495-challenger-toto2-4m-u238"

local_download = snapshot_download(
    repo_id=repo_id,
    revision="main",
    # allow_patterns=[
    #     "optimizer.safetensors",
    #     "weights.safetensors",
    #     "weights_stable.safetensors",
    #     "model.py",
    #     "forecast_wrapper.py",
    #     "config.json",
    # ],
    local_dir="./models/ckpt-r15703366466408532495-challenger-toto2-4m-u238",
    max_workers=8,
)

print("Downloaded to:", local_download)
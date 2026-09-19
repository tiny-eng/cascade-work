from hippius_hub import snapshot_download

repo_id = "cascade/genesis-baseline"

local_download = snapshot_download(
    repo_id=repo_id,
    revision="main",
    local_dir="./generators/cascade/genesis-baseline",
    max_workers=8,
)

print("Downloaded to:", local_download)
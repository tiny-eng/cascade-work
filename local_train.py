"""Train Toto2 locally from a saved Cascade heat or final dataset.

The dataset manifest selects the stage and records the exact budget. This
script reconstructs the matching subnet contract and calls the same
``Toto2Trainer.train`` implementation used by Cascade training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-dir", type=Path, required=True,
						help="Directory containing manifest.json and shard-*.npy.")
	parser.add_argument("--output-dir", type=Path, required=True,
						help="Directory receiving the trained Toto2 checkpoint.")
	parser.add_argument("--chain-toml", type=Path, default=None,
						help="Chain config; defaults to the repository's chain.toml.")
	parser.add_argument("--repo-root", type=Path, default=None,
						help="Cascade repository root; defaults to the parent of cascade-work.")
	parser.add_argument("--device", default=None,
						help="Torch device, for example cuda or cpu; default follows Toto2Trainer.")
	parser.add_argument("--dtype", default="float32",
						help="Torch model dtype passed to Toto2Trainer (default: float32).")
	parser.add_argument("--warm-start-dir", type=Path, default=None,
						help="Downloaded checkpoint directory containing weights.safetensors.")
	parser.add_argument("--allow-mismatch", action="store_true",
						help="Allow manifest/config differences; not recommended for exact runs.")
	parser.add_argument("--dry-run", action="store_true",
						help="Validate manifest, contract, and shards without training.")
	return parser


def _absolute(path: Path) -> Path:
	return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _final_contract(cfg, arch_preset: str):
	if arch_preset == cfg.training.primary_size.arch_preset:
		return cfg.training.primary_size
	for size in cfg.training.extra_sizes:
		if size.arch_preset == arch_preset:
			return cfg.training.for_size(size)
	available = [cfg.training.primary_size.arch_preset]
	available.extend(size.arch_preset for size in cfg.training.extra_sizes)
	raise ValueError(
		f"manifest arch_preset {arch_preset!r} is not configured; "
		f"available: {', '.join(available)}"
	)


def _contract_for_manifest(cfg, manifest: dict):
	stage = manifest.get("stage")
	if stage == "heat":
		return cfg.screen_contract().for_hours(
			cfg.round.heat_train_hours,
			guard_factor=cfg.round.heat_guard_factor,
			guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
		)
	if stage == "final":
		return _final_contract(cfg, str(manifest["arch_preset"]))
	raise ValueError(f"manifest stage must be 'heat' or 'final', got {stage!r}")


def _check_manifest(cfg, manifest: dict, contract, args: argparse.Namespace) -> None:
	from cascade.trainer.contract import RoundSeeds

	seeds = RoundSeeds.derive(int(manifest["base_seed"]), contract)
	expected = {
		"generation_seed": seeds.generation_seed,
		"training_seed": seeds.training_seed,
		"token_budget": contract.train_tokens,
		"budget_denomination": getattr(contract, "budget_denomination", "points"),
		"max_train_seconds": contract.max_train_seconds,
	}
	manifest_chain = str(manifest.get("chain_toml", ""))
	requested_chain = str(args.chain_toml)
	# Dataset manifests may be created on Windows and consumed from WSL. Compare
	# the config filename after normalizing both slash styles; the contract fields
	# below remain the authoritative cross-platform compatibility check.
	manifest_name = manifest_chain.replace("\\", "/").rsplit("/", 1)[-1]
	requested_name = requested_chain.replace("\\", "/").rsplit("/", 1)[-1]
	if manifest_name != requested_name and not args.allow_mismatch:
		raise ValueError(
			f"manifest chain config is {manifest['chain_toml']!r}, "
			f"but --chain-toml is {args.chain_toml}; pass --allow-mismatch to override"
		)
	for key, value in expected.items():
		if manifest.get(key) != value and not args.allow_mismatch:
			raise ValueError(
				f"manifest {key}={manifest.get(key)!r} does not match config {value!r}; "
				"use the matching chain config or --allow-mismatch"
			)


def _dataset_stream(dataset_dir: Path, manifest: dict) -> Iterator[np.ndarray]:
	for shard_name in manifest["shards"]:
		path = dataset_dir / shard_name
		if not path.is_file():
			raise FileNotFoundError(f"dataset shard does not exist: {path}")
		records = np.load(path, allow_pickle=True)
		for values in records:
			yield values


def _resolve_warm_start(path: Path | None) -> Path | None:
	if path is None:
		return None
	path = _absolute(path)
	if path.is_file() and path.name == "weights.safetensors":
		path = path.parent
	if not path.is_dir():
		raise FileNotFoundError(f"warm-start checkpoint directory does not exist: {path}")
	weights = path / "weights.safetensors"
	stable = path / "weights_stable.safetensors"
	if not weights.is_file() and not stable.is_file():
		raise FileNotFoundError(
			f"warm-start checkpoint has no weights.safetensors or "
			f"weights_stable.safetensors: {path}"
		)
	return path


def train(args: argparse.Namespace) -> dict[str, object]:
	root = (args.repo_root or _repo_root()).resolve()
	if str(root) not in sys.path:
		sys.path.insert(0, str(root))

	from cascade.shared.config import load_chain_config
	from cascade.trainer.toto2_trainer import Toto2Trainer

	dataset_dir = _absolute(args.dataset_dir)
	manifest_path = dataset_dir / "manifest.json"
	if not manifest_path.is_file():
		raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	chain_toml = _absolute(args.chain_toml) if args.chain_toml else root / "chain.toml"
	cfg = load_chain_config(chain_toml)
	contract = _contract_for_manifest(cfg, manifest)
	_check_manifest(cfg, manifest, contract, argparse.Namespace(
		chain_toml=chain_toml, allow_mismatch=args.allow_mismatch,
	))
	warm_start_dir = _resolve_warm_start(args.warm_start_dir)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	if args.dry_run:
		for _ in _dataset_stream(dataset_dir, manifest):
			pass
		return {
			"stage": manifest["stage"],
			"arch_preset": contract.arch_preset,
			"token_budget": contract.train_tokens,
			"dataset_digest": manifest.get("corpus_digest"),
			"status": "validated",
			"warm_start_dir": str(warm_start_dir) if warm_start_dir else None,
		}

	trainer = Toto2Trainer(device=args.device, dtype=args.dtype, deterministic=True)
	result = trainer.train(
		_dataset_stream(dataset_dir, manifest),
		contract,
		training_seed=int(manifest["training_seed"]),
		token_budget=contract.train_tokens,
		out_dir=args.output_dir,
		warm_start_dir=warm_start_dir,
	)
	return {
		"stage": manifest["stage"],
		"arch_preset": contract.arch_preset,
		"dataset_digest": manifest.get("corpus_digest"),
		"checkpoint_dir": str(result.local_dir),
		"warm_start_dir": str(warm_start_dir) if warm_start_dir else None,
		"train_seconds": result.train_seconds,
		"metrics": result.metrics,
	}


def main() -> int:
	args = _parser().parse_args()
	try:
		print(json.dumps(train(args), indent=2, sort_keys=True))
	except Exception as exc:  # noqa: BLE001 - CLI reports a concise failure
		print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

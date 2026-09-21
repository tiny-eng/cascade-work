"""Construct a local finalist-training dataset with cascade's final contract.

The generator loading, validation, seeding, stream budgeting, and digesting
are all delegated to the subnet implementation. Unlike the heat constructor,
this uses the configured full ``[training]`` budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--generator-dir", type=Path, required=True,
						help="Local finalist generator repo containing generator.py.")
	parser.add_argument("--base-seed", type=int, required=True,
						help="Round base seed from the epoch-boundary block hash.")
	parser.add_argument("--chain-toml", type=Path, default=None,
						help="Chain config; defaults to the repository's chain.toml.")
	parser.add_argument("--output-dir", type=Path, required=True,
						help="Directory receiving shard-*.npy and manifest.json.")
	parser.add_argument("--repo-root", type=Path, default=None,
						help="Cascade repository root; defaults to the parent of cascade-work.")
	parser.add_argument("--arch-preset", default=None,
						help="Configured final size; defaults to the primary training size.")
	parser.add_argument("--shard-points", type=int, default=10_000_000,
						help="Approximate series-points per shard (default: 10,000,000).")
	parser.add_argument("--overwrite", action="store_true",
						help="Remove existing shard files and manifest before rebuilding.")
	parser.add_argument("--progress", action="store_true",
						help="Show dataset-generation progress and ETA.")
	parser.add_argument("--in-process", action="store_true",
						help="Run the generator in-process instead of cascade's sandbox.")
	return parser


def _absolute(path: Path) -> Path:
	return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _final_contract(cfg, arch_preset: str | None):
	if not arch_preset:
		return cfg.training.primary_size
	if arch_preset == cfg.training.primary_size.arch_preset:
		return cfg.training.primary_size
	for size in cfg.training.extra_sizes:
		if size.arch_preset == arch_preset:
			return cfg.training.for_size(size)
	raised = [cfg.training.primary_size.arch_preset]
	raised.extend(size.arch_preset for size in cfg.training.extra_sizes)
	raised = ", ".join(raised)
	raise ValueError(f"unknown --arch-preset {arch_preset!r}; available: {raised}")


def construct_dataset(args: argparse.Namespace) -> dict[str, object]:
	root = (args.repo_root or _repo_root()).resolve()
	if str(root) not in sys.path:
		sys.path.insert(0, str(root))

	from cascade.shared.config import load_chain_config
	from cascade.trainer.contract import RoundSeeds
	from cascade.trainer.stream import open_round_stream

	chain_toml = args.chain_toml or root / "chain.toml"
	generator_dir = _absolute(args.generator_dir)
	output_dir = _absolute(args.output_dir)
	if not generator_dir.is_dir():
		raise FileNotFoundError(f"generator directory does not exist: {generator_dir}")
	if args.shard_points <= 0:
		raise ValueError("--shard-points must be positive")

	cfg = load_chain_config(chain_toml)
	contract = _final_contract(cfg, args.arch_preset)
	seeds = RoundSeeds.derive(args.base_seed, contract)
	token_budget = contract.train_tokens
	output_dir.mkdir(parents=True, exist_ok=True)
	if any(output_dir.glob("shard-*.npy")) or (output_dir / "manifest.json").exists():
		if not args.overwrite:
			raise FileExistsError(
				f"output directory already contains a dataset: {output_dir}; "
				"pass --overwrite to rebuild it"
			)
		for path in output_dir.glob("shard-*.npy"):
			path.unlink()
		manifest_path = output_dir / "manifest.json"
		if manifest_path.exists():
			manifest_path.unlink()

	shard: list[np.ndarray] = []
	shard_points = 0
	shard_index = 0
	shard_paths: list[str] = []
	n_series = 0
	total_points = 0
	started = time.monotonic()
	last_progress = started
	denomination = getattr(contract, "budget_denomination", "points")

	def flush() -> None:
		nonlocal shard, shard_points, shard_index
		if not shard:
			return
		path = output_dir / f"shard-{shard_index:06d}.npy"
		records = np.empty(len(shard), dtype=object)
		for index, values in enumerate(shard):
			records[index] = values
		np.save(path, records, allow_pickle=True)
		shard_paths.append(path.name)
		shard_index += 1
		shard = []
		shard_points = 0

	with open_round_stream(
		contract.corpus_mode,
		generator_dir,
		seeds.generation_seed,
		cfg.generator,
		token_budget=token_budget,
		use_sandbox=not args.in_process,
		blocked=cfg.static_guard.blocked,
		max_wall_seconds=contract.max_train_seconds,
		seed_mix=int(getattr(contract, "gen_seed_mix", 1) or 1),
		budget_denomination=denomination,
	) as stream:
		for item in stream.series():
			values = item["values"] if isinstance(item, dict) else item
			values = np.asarray(values)
			shard.append(values.copy())
			points = (int(values.shape[-1]) if denomination == "series_points"
					  else int(values.size))
			shard_points += points
			total_points += points
			n_series += 1
			if args.progress and time.monotonic() - last_progress >= 1.0:
				elapsed = max(0.001, time.monotonic() - started)
				fraction = min(1.0, total_points / max(1, token_budget))
				throughput = total_points / elapsed
				remaining = max(0, token_budget - total_points)
				eta = remaining / throughput if throughput > 0 else 0.0
				width = 30
				filled = int(width * fraction)
				bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
				sys.stderr.write(
					f"\rgenerating [{bar}] {fraction * 100:6.2f}% "
					f"points={total_points:,}/{token_budget:,} "
					f"series={n_series:,} rate={throughput:,.0f}/s "
					f"ETA={eta / 60:.1f}m"
				)
				sys.stderr.flush()
				last_progress = time.monotonic()
			if shard_points >= args.shard_points:
				flush()
		flush()
		if args.progress:
			sys.stderr.write("\n")
			sys.stderr.flush()
		manifest = {
			"format": 1,
			"stage": "final",
			"chain_toml": str(Path(chain_toml).resolve()),
			"generator_dir": str(generator_dir),
			"arch_preset": contract.arch_preset,
			"base_seed": int(args.base_seed),
			"generation_seed": int(seeds.generation_seed),
			"training_seed": int(seeds.training_seed),
			"corpus_mode": contract.corpus_mode,
			"budget_denomination": denomination,
			"target_train_hours": float(contract.target_train_hours),
			"token_budget": int(token_budget),
			"shard_points_target": int(args.shard_points),
			"max_train_seconds": int(contract.max_train_seconds),
			"n_series": int(n_series),
			"total_points": int(total_points),
			"stream_total_points": int(stream.total_points),
			"corpus_digest": stream.digest,
			"shards": shard_paths,
			"sandboxed": not args.in_process,
		}

	(output_dir / "manifest.json").write_text(
		json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	return manifest


def main() -> int:
	args = _parser().parse_args()
	try:
		manifest = construct_dataset(args)
	except Exception as exc:  # noqa: BLE001 - CLI reports a concise failure
		print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
		return 1
	print(json.dumps(manifest, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

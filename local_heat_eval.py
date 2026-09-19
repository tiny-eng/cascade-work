"""Rank local challenger checkpoints using Cascade's heat-screen logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _root() -> Path:
	return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--challenger", type=Path, action="append", required=True,
				   help="Challenger checkpoint directory; repeat for multiple models.")
	p.add_argument("--eval-pool", type=Path, required=True,
				   help="Local eval-pool directory containing .npy/.npz series.")
	p.add_argument("--chain-toml", type=Path, default=None)
	p.add_argument("--base-seed", type=int, required=True,
				   help="Round seed, normally the epoch-boundary block-derived seed.")
	p.add_argument("--block", type=int, default=None,
				   help="Epoch block for block-gated window composition; optional.")
	p.add_argument("--device", default="cpu")
	p.add_argument("--output-json", type=Path, default=None,
				   help="Also save the evaluation result as JSON at this path.")
	p.add_argument("--trust-checkpoint-code", action="store_true",
				   help="Skip checkpoint guard; use only for trusted local checkpoints.")
	return p


def main() -> int:
	args = _parser().parse_args()
	try:
		root = _root()
		if str(root) not in sys.path:
			sys.path.insert(0, str(root))
		from cascade.eval.scoring import global_geomean
		from cascade.eval.heat import screen_diagnostics, tied_set
		from cascade.shared.config import load_chain_config
		from cascade.validator.evaluator import evaluate_checkpoint
		from cascade.validator.pool import window_source_from_dir

		chain_toml = args.chain_toml or root / "chain.toml"
		cfg = load_chain_config(chain_toml)
		pool = window_source_from_dir(
			args.eval_pool.resolve(), cfg, label=str(args.eval_pool.resolve())
		)
		windows = pool.windows_for_round(
			args.base_seed, cfg.eval.n_windows, block=args.block
		)
		contract = cfg.screen_contract()
		rows = []
		scores_by_checkpoint = {}
		for checkpoint in args.challenger:
			checkpoint = checkpoint.resolve()
			scores = evaluate_checkpoint(
				checkpoint, windows, num_samples=cfg.eval.num_samples,
				device=args.device, contract=contract,
				trust_checkpoint_code=args.trust_checkpoint_code,
			)
			rows.append({
				"checkpoint": str(checkpoint),
				"geomean": global_geomean(scores),
				"n_windows": len(scores),
				"scores": [
					{"series_id": s.series_id, "source": s.source,
					 "channel": s.channel, "domain": s.domain,
					 "mase": s.mase,
					 "qloss_per_q": s.qloss_per_q.tolist(),
					 "abs_target": s.abs_target,
					 "quantile_levels": list(s.quantile_levels)}
					for s in scores
				],
			})
			scores_by_checkpoint[str(checkpoint)] = scores
		rows.sort(key=lambda row: row["geomean"])
		ranked_keys = [row["checkpoint"] for row in rows]
		selection_diagnostics = None
		if cfg.round.max_finalists > 1 and ranked_keys:
			selection_diagnostics = screen_diagnostics(
				[(key, scores_by_checkpoint[key]) for key in ranked_keys],
				seed=args.base_seed,
				B=cfg.scoring.bootstrap_B,
				alpha=cfg.scoring.bootstrap_alpha,
			)
			selected_keys = tied_set(
				selection_diagnostics,
				ranked_keys,
				cap=cfg.round.finalist_cap,
			) if cfg.round.max_finalists > 1 else ranked_keys[:cfg.round.finalists]
		selection_diagnostics_json = None
		if selection_diagnostics is not None:
			selection_diagnostics_json = {
				"p_best": selection_diagnostics.p_best,
				"leader_lcb": selection_diagnostics.leader_lcb,
				"leader_key": selection_diagnostics.leader_key,
				"runner_up_key": selection_diagnostics.runner_up_key,
				"n_windows": selection_diagnostics.n_windows,
				"n_clusters": selection_diagnostics.n_clusters,
				"lcb_vs": selection_diagnostics.lcb_vs,
			}
		result = {
			"stage": "heat",
			"chain_toml": str(Path(chain_toml).resolve()),
			"eval_pool": str(args.eval_pool.resolve()),
			"base_seed": args.base_seed,
			"block": args.block,
			"n_windows": len(windows),
			"num_samples": cfg.eval.num_samples,
			"configured_finalists": cfg.round.finalists,
			"finalist_cap": cfg.round.finalist_cap,
			"selected_finalists": selected_keys,
			"selection_diagnostics": selection_diagnostics_json,
			"ranking": rows,
		}
		text = json.dumps(result, indent=2, sort_keys=True) + "\n"
		if args.output_json is not None:
			output_json = args.output_json.resolve()
			output_json.parent.mkdir(parents=True, exist_ok=True)
			output_json.write_text(text, encoding="utf-8")
		print(text, end="")
		return 0
	except Exception as exc:  # noqa: BLE001 - concise CLI failure
		print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())

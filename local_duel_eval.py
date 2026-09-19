"""Compare a local challenger checkpoint against a local king checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def _root() -> Path:
	return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--challenger", type=Path, required=True,
				   help="Finalist challenger checkpoint directory.")
	p.add_argument("--king", type=Path, required=True,
				   help="Current king checkpoint directory.")
	p.add_argument("--eval-pool", type=Path, required=True,
				   help="Local eval-pool directory containing .npy/.npz series.")
	p.add_argument("--chain-toml", type=Path, default=None)
	p.add_argument("--base-seed", type=int, required=True,
				   help="Round seed, normally the epoch-boundary block-derived seed.")
	p.add_argument("--block", type=int, default=None,
				   help="Epoch block for block-gated scoring and window composition.")
	p.add_argument("--king-tenure-rounds", type=int, default=0,
				   help="King tenure used by the configured margin schedule.")
	p.add_argument("--arch-preset", default=None,
				   help="Final contract size; defaults to the primary training size.")
	p.add_argument("--device", default="cpu")
	p.add_argument("--output-json", type=Path, default=None,
				   help="Also save the evaluation result as JSON at this path.")
	p.add_argument("--trust-checkpoint-code", action="store_true",
				   help="Skip checkpoint guard; use only for trusted local checkpoints.")
	return p


def _contract(cfg, arch_preset: str | None):
	if not arch_preset:
		return cfg.training.primary_size
	return cfg.training.contract_for(arch_preset)


def _dethrone_state(result) -> str:
	if result.inconclusive:
		return "INCONCLUSIVE"
	return "DETHRONE" if result.challenger_wins_round else "HELD"


def main() -> int:
	args = _parser().parse_args()
	try:
		root = _root()
		if str(root) not in sys.path:
			sys.path.insert(0, str(root))
		from cascade.eval.koth import evaluate_round
		from cascade.shared.config import load_chain_config
		from cascade.validator.evaluator import evaluate_checkpoint
		from cascade.validator.pool import window_source_from_dir

		chain_toml = args.chain_toml or root / "chain.toml"
		cfg = load_chain_config(chain_toml)
		contract = _contract(cfg, args.arch_preset)
		pool = window_source_from_dir(
			args.eval_pool.resolve(), cfg, label=str(args.eval_pool.resolve())
		)
		windows = pool.windows_for_round(
			args.base_seed, cfg.eval.n_windows, block=args.block
		)
		common = {
			"windows": windows,
			"num_samples": cfg.eval.num_samples,
			"device": args.device,
			"contract": contract,
			"trust_checkpoint_code": args.trust_checkpoint_code,
		}
		king_scores = evaluate_checkpoint(args.king.resolve(), **common)
		challenger_scores = evaluate_checkpoint(args.challenger.resolve(), **common)
		params = cfg.koth_params(block=args.block)
		result = evaluate_round(
			king_scores,
			challenger_scores,
			params,
			seed=args.base_seed,
			king_tenure_rounds=args.king_tenure_rounds,
		)
		output = {
			"stage": "duel",
			"chain_toml": str(Path(chain_toml).resolve()),
			"eval_pool": str(args.eval_pool.resolve()),
			"king": str(args.king.resolve()),
			"challenger": str(args.challenger.resolve()),
			"base_seed": args.base_seed,
			"block": args.block,
			"king_tenure_rounds": args.king_tenure_rounds,
			"arch_preset": contract.arch_preset,
			"n_windows": len(windows),
			"num_samples": cfg.eval.num_samples,
			"params": dataclasses.asdict(params),
			"dethrone_state": _dethrone_state(result),
			"result": dataclasses.asdict(result),
		}
		text = json.dumps(output, indent=2, sort_keys=True) + "\n"
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

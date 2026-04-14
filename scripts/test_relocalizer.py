"""
Smoke test for ProteinRelocalizer using multiple source->target pairs.

The script tries candidate shifts:
- Cytoplasm -> Extracellular
- Cytoplasm -> Membrane
- Cytoplasm -> Nucleus

For each pair, it picks a protein in a configurable source-probability band and
low target probability, then runs a short probe optimization to estimate movement.
It chooses the pair with the clearest target-probability gain and runs the full
demo optimization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _QuietTqdm:
    """Disable progress bar noise for smoke tests."""

    def __init__(self, iterable, *args, **kwargs):
        self._it = iterable

    def __iter__(self):
        return iter(self._it)

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


import tqdm as _tqdm_mod  # noqa: E402

_tqdm_mod.tqdm = _QuietTqdm  # noqa: E402

from src.design.relocalizer import ProteinRelocalizer  # noqa: E402


def _pick_test_protein_for_pair(
    df: pd.DataFrame,
    relocalizer: ProteinRelocalizer,
    *,
    source: str,
    target: str,
    min_seq_len: int,
    max_seq_len: int,
    source_prob_min: float,
    source_prob_max: float,
    target_prob_max: float,
    max_scan: int,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    From CSV rows labeled by `source`, pick a sequence in [min_seq_len, max_seq_len]
    whose classifier probabilities satisfy:
      source_prob_min <= P(source) <= source_prob_max
      P(target) < target_prob_max

    Rows are ordered **longest first** (within the length band).
    """
    required = {"ACC", "Sequence", source, target}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    pool = df[df[source] == 1].copy()
    if pool.empty:
        raise ValueError(f"No rows with {source} label == 1 in CSV.")

    pool["seq_len"] = pool["Sequence"].astype(str).str.len()
    pool = pool[(pool["seq_len"] >= min_seq_len) & (pool["seq_len"] <= max_seq_len)]
    pool = pool.sort_values("seq_len", ascending=False)

    if pool.empty:
        raise ValueError(
            f"No Cytoplasm-positive rows with {min_seq_len} <= length <= {max_seq_len}."
        )

    tried = 0
    for _, row in pool.iterrows():
        if tried >= max_scan:
            break
        tried += 1
        seq = str(row["Sequence"]).upper().strip()
        if len(seq) < min_seq_len or len(seq) > max_seq_len:
            continue
        scored = relocalizer.score_variant(seq)
        probs = scored["localization_probs"]
        p_src = float(probs.get(source, 0.0))
        p_tgt = float(probs.get(target, 0.0))
        if source_prob_min <= p_src <= source_prob_max and p_tgt < target_prob_max:
            return str(row["ACC"]), seq, scored

    raise RuntimeError(
        f"No sequence found after scoring {tried} {source}-labeled proteins "
        f"(len in [{min_seq_len}, {max_seq_len}], need {source_prob_min}<=P({source})<={source_prob_max} "
        f"and P({target})<{target_prob_max}). "
        "Try increasing --max-scan or relaxing probability/length bounds."
    )


def _print_trajectory(traj: list, source: str, target: str) -> None:
    print("\n--- Optimization trajectory ---")
    if not traj:
        print("(empty)")
        return
    for step in traj:
        it = step.get("iteration", "?")
        p_tgt = step.get("best_target_prob")
        p_src = step.get("best_source_prob")
        nmut = step.get("num_mutations")
        neval = step.get("num_candidates_evaluated")
        err = step.get("error")
        parts = [f"iter={it}"]
        if p_tgt is not None:
            parts.append(f"P({target})={p_tgt:.4f}")
        if p_src is not None:
            parts.append(f"P({source})={p_src:.4f}")
        if nmut is not None:
            parts.append(f"muts={nmut}")
        if neval is not None:
            parts.append(f"evaluated={neval}")
        if err:
            parts.append(f"error={err}")
        print("  " + " | ".join(parts))


def _print_top_candidate(
    results: Dict[str, Any],
    source: str,
    target: str,
) -> None:
    tops = results.get("top_candidates") or []
    print("\n--- Top candidate ---")
    if not tops:
        print("(none)")
        return

    best = tops[0]
    orig = results["original_scores"]["localization_probs"]
    newp = best["localization_probs"]

    print(f"Composite score: {best['composite_score']:.4f}")
    print(f"Plausibility:    {best['plausibility_score']:.4f}")
    print("")
    print("Probability changes (original -> candidate):")
    for loc in (source, target):
        o = float(orig.get(loc, 0.0))
        n = float(newp.get(loc, 0.0))
        print(f"  P({loc}): {o:.4f} -> {n:.4f}  (delta {n - o:+.4f})")

    muts = best.get("mutations") or []
    print("")
    print(f"Mutations vs original ({len(muts)}):")
    if muts:
        shown = muts[:40]
        line = ", ".join(f"{p}{a}>{b}" for p, a, b in shown)
        if len(muts) > 40:
            line += ", ..."
        print(f"  {line}")
    else:
        print("  (none; same as original)")


def _print_all_location_changes(results: Dict[str, Any]) -> None:
    print("\n--- Full localization profile shift (all labels) ---")
    tops = results.get("top_candidates") or []
    if not tops:
        print("(no top candidate)")
        return
    orig = results.get("original_scores", {}).get("localization_probs", {})
    best = tops[0].get("localization_probs", {})
    labels = sorted(set(orig.keys()) | set(best.keys()))
    for label in labels:
        o = float(orig.get(label, 0.0))
        n = float(best.get(label, 0.0))
        print(f"  {label:24s} {o:.4f} -> {n:.4f}  (delta {n - o:+.4f})")


def _movement_score(results: Dict[str, Any], source: str, target: str) -> float:
    """Positive score means shift toward target and away from source."""
    orig = results.get("original_scores", {}).get("localization_probs", {})
    tops = results.get("top_candidates") or []
    if not tops:
        return float("-inf")
    best = tops[0].get("localization_probs", {})
    delta_t = float(best.get(target, 0.0)) - float(orig.get(target, 0.0))
    delta_s = float(orig.get(source, 0.0)) - float(best.get(source, 0.0))
    return delta_t + 0.5 * delta_s


def main() -> None:
    try:
        import captum  # noqa: F401
    except ImportError:
        print(
            "ERROR: captum is required for relocalize() (integrated gradients). "
            "Install with: pip install captum",
            file=sys.stderr,
        )
        sys.exit(2)

    p = argparse.ArgumentParser(description="Smoke test ProteinRelocalizer (lightweight, fast).")
    p.add_argument(
        "--classifier-path",
        type=Path,
        default=ROOT / "models" / "best_model.pt",
        help="Trained classifier checkpoint.",
    )
    p.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "data" / "processed" / "deeploc_multilabel.csv",
        help="Multilabel CSV with ACC, Sequence, Cytoplasm, ...",
    )
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument(
        "--min-seq-len",
        type=int,
        default=150,
        help="Minimum sequence length for the test protein (default: 150).",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=350,
        help="Maximum sequence length for the test protein (default: 350).",
    )
    p.add_argument(
        "--source-prob-min",
        type=float,
        default=0.5,
        help="Minimum P(source) for selected test protein (default: 0.5).",
    )
    p.add_argument(
        "--source-prob-max",
        type=float,
        default=0.9,
        help="Maximum P(source) for selected test protein (default: 0.9).",
    )
    p.add_argument(
        "--target-prob-max",
        type=float,
        default=0.25,
        help="Require P(target) below this for selected protein (default: 0.25).",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=500,
        help="Max CSV rows (after length filter) to score when searching for a match.",
    )
    p.add_argument(
        "--probe-iterations",
        type=int,
        default=3,
        help="Short probe iterations for pair selection (default: 3).",
    )
    p.add_argument(
        "--probe-candidates",
        type=int,
        default=10,
        help="Short probe candidates/iter for pair selection (default: 10).",
    )
    args = p.parse_args()

    classifier_path = args.classifier_path
    if not classifier_path.is_absolute():
        classifier_path = (ROOT / classifier_path).resolve()
    csv_path = args.csv_path
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()

    if not classifier_path.is_file():
        print(f"ERROR: missing classifier: {classifier_path}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.is_file():
        print(f"ERROR: missing dataset CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading ProteinRelocalizer.from_lightweight() (t33 encoder + t12 MLM)...", flush=True)
    device = args.device
    relocalizer = ProteinRelocalizer.from_lightweight(
        classifier_path=classifier_path,
        device=device,
    )

    if args.min_seq_len > args.max_seq_len:
        print("ERROR: --min-seq-len must be <= --max-seq-len", file=sys.stderr)
        sys.exit(1)
    if args.source_prob_min > args.source_prob_max:
        print("ERROR: --source-prob-min must be <= --source-prob-max", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    pair_candidates: List[Tuple[str, str]] = [
        ("Cytoplasm", "Extracellular"),
        ("Cytoplasm", "Membrane"),
        ("Cytoplasm", "Nucleus"),
    ]
    pair_trials: List[Dict[str, Any]] = []
    print("\nTrying source->target pairs to find a clear demo shift...", flush=True)
    for source, target in pair_candidates:
        print(f"\n[Pair probe] {source} -> {target}", flush=True)
        try:
            acc, sequence, initial = _pick_test_protein_for_pair(
                df,
                relocalizer,
                source=source,
                target=target,
                min_seq_len=args.min_seq_len,
                max_seq_len=args.max_seq_len,
                source_prob_min=args.source_prob_min,
                source_prob_max=args.source_prob_max,
                target_prob_max=args.target_prob_max,
                max_scan=args.max_scan,
            )
            probe = relocalizer.relocalize(
                sequence,
                source_location=source,
                target_location=target,
                n_iterations=args.probe_iterations,
                candidates_per_iteration=args.probe_candidates,
            )
            move = _movement_score(probe, source, target)
            print(
                f"  ACC={acc} len={len(sequence)} "
                f"P({source})={initial['localization_probs'].get(source, 0):.4f} "
                f"P({target})={initial['localization_probs'].get(target, 0):.4f} "
                f"probe_movement={move:+.4f}",
                flush=True,
            )
            pair_trials.append(
                {
                    "source": source,
                    "target": target,
                    "acc": acc,
                    "sequence": sequence,
                    "initial": initial,
                    "probe_results": probe,
                    "movement": move,
                }
            )
        except Exception as ex:
            print(f"  skipped: {ex}", flush=True)

    if not pair_trials:
        raise RuntimeError("No valid pair/test-protein found for the configured constraints.")

    pair_trials.sort(key=lambda x: float(x["movement"]), reverse=True)
    chosen = pair_trials[0]
    source = str(chosen["source"])
    target = str(chosen["target"])
    acc = str(chosen["acc"])
    sequence = str(chosen["sequence"])
    initial = chosen["initial"]
    print(
        f"\nSelected demo pair: {source} -> {target} "
        f"(probe movement {float(chosen['movement']):+.4f})",
        flush=True,
    )
    print(
        f"Selected ACC={acc}, length={len(sequence)} "
        f"P({source})={initial['localization_probs'].get(source, 0):.4f} "
        f"P({target})={initial['localization_probs'].get(target, 0):.4f}",
        flush=True,
    )

    print("\nRunning full relocalize (10 iterations, 20 candidates/iter)...", flush=True)
    results = relocalizer.relocalize(
        sequence,
        source_location=source,
        target_location=target,
        n_iterations=10,
        candidates_per_iteration=20,
    )

    _print_trajectory(results.get("optimization_trajectory") or [], source, target)
    _print_top_candidate(results, source, target)
    _print_all_location_changes(results)

    print("\n--- Summary ---")
    print(relocalizer.get_summary(results))
    print(f"\nTotal time: {results.get('total_time_seconds', 0):.2f}s  "
          f"variants scored: {results.get('total_variants_evaluated', 0)}")


if __name__ == "__main__":
    main()

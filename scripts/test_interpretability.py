"""
Smoke test for residue-level interpretability.

From project root:
    .\\venv\\Scripts\\python.exe scripts\\test_interpretability.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.interpretability import ProteinInterpreter  # noqa: E402


def _select_short_membrane_sequence(csv_path: Path) -> Tuple[str, str]:
    df = pd.read_csv(csv_path)
    required = {"ACC", "Sequence", "Membrane"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    mem = df[df["Membrane"] == 1].copy()
    if mem.empty:
        raise ValueError("No membrane-positive sample found in dataset.")
    mem["seq_len"] = mem["Sequence"].astype(str).str.len()

    # Prefer short sequences to keep ESM+IG runtime manageable.
    short = mem[mem["seq_len"] <= 160]
    pool = short if not short.empty else mem
    row = pool.sort_values("seq_len", ascending=True).iloc[0]
    return str(row["ACC"]), str(row["Sequence"])


def _top_k(
    residue_scores: Sequence[Tuple[int, str, float]],
    k: int = 10,
    by_abs: bool = False,
) -> List[Tuple[int, str, float]]:
    if by_abs:
        ranked = sorted(residue_scores, key=lambda x: abs(float(x[2])), reverse=True)
    else:
        ranked = sorted(residue_scores, key=lambda x: float(x[2]), reverse=True)
    return ranked[:k]


def main() -> None:
    t0 = time.perf_counter()

    model_path = ROOT / "models" / "best_model.pt"
    csv_path = ROOT / "data" / "processed" / "deeploc_multilabel.csv"
    plot_path = ROOT / "plots" / "test_attribution.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {csv_path}")

    acc, seq = _select_short_membrane_sequence(csv_path)
    print(f"Selected sample ACC={acc}, length={len(seq)}")

    interpreter = ProteinInterpreter(
        classifier_path=model_path,
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    print("\n1) Attention scores")
    attention_out = interpreter.get_attention_scores(seq)
    top_attn = _top_k(attention_out["residue_scores"], k=10, by_abs=False)
    for pos, aa, score in top_attn:
        print(f"  pos={pos:4d} aa={aa} attention={score:.6f}")

    print("\n2) Integrated gradients for target='Membrane'")
    ig_out = interpreter.get_integrated_gradients(seq, target_location="Membrane")
    top_ig = _top_k(ig_out["residue_scores"], k=10, by_abs=True)
    for pos, aa, score in top_ig:
        print(f"  pos={pos:4d} aa={aa} attribution={score:.6f}")

    print("\n3) Hot regions")
    hot_regions = interpreter.identify_hot_regions(ig_out["residue_scores"], window_size=10, top_percentile=90)
    if not hot_regions:
        print("  No hot regions found.")
    for i, r in enumerate(hot_regions, start=1):
        print(
            f"  region {i}: start={r['start']}, end={r['end']}, "
            f"avg_score={r['avg_score']:.6f}, subseq={r['subsequence'][:30]}..."
        )

    print("\n4) Known signal checks")
    signal_checks = interpreter.validate_against_known_signals(seq, hot_regions)
    for name, payload in signal_checks.items():
        print(
            f"  {name}: detected={payload['detected']}, "
            f"region={payload['region']}, overlap_with_attribution={payload['overlap_with_attribution']}"
        )

    print("\n5) Attribution visualization")
    interpreter.visualize_attribution(
        sequence=seq,
        residue_scores=ig_out["residue_scores"],
        hot_regions=hot_regions,
        output_path=plot_path,
    )
    print(f"  Saved plot: {plot_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()


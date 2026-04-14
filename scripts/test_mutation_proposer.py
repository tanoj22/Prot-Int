"""
Smoke-test src/design/mutation_proposer.py.

From project root:
    .\\venv\\Scripts\\python.exe scripts\\test_mutation_proposer.py

Uses a short sequence and facebook/esm2_t12_35M_UR50D by default for speed
(override with ESM_MODEL env var).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.design.mutation_proposer import MutationProposer  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    esm = os.environ.get("ESM_MODEL", "facebook/esm2_t12_35M_UR50D")
    # Short sequence; membrane-relevant N-term for smart-mutation heuristics
    seq = "MKFLKFSLALTLLSGLIAAAFA"  # 22 aa

    t0 = time.perf_counter()
    proposer = MutationProposer(esm_model_name=esm, device="cpu")

    print("\n--- propose_single_mutations (positions 3, 10, 18) ---")
    single = proposer.propose_single_mutations(seq, positions=[3, 10, 18], top_k=5)
    for block in single:
        print(block)

    print("\n--- propose_smart_mutations (Membrane -> Mitochondrion) ---")
    # Fake attribution: high at 5 and 12
    att = {i: 0.1 for i in range(1, len(seq) + 1)}
    att[5] = 1.0
    att[12] = 0.9
    smart = proposer.propose_smart_mutations(
        seq,
        attribution_scores=att,
        current_location="Membrane",
        target_location="Mitochondrion",
        n_positions=4,
        top_k=3,
    )
    for block in smart:
        print(block)

    print("\n--- generate_variants ---")
    variants = proposer.generate_variants(
        seq,
        mutation_proposals=smart,
        max_simultaneous_mutations=3,
        max_variants=20,
    )
    for v in variants[:8]:
        print(v)

    print("\n--- score_sequence_plausibility ---")
    pll = proposer.score_sequence_plausibility(seq, subsample_step=5)
    print(f"PLL (subsampled): {pll:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.2f}s")
    print("OK - mutation proposer smoke test finished.")


if __name__ == "__main__":
    main()

"""
Smoke tests for VariantEffectPredictor.

Test cases:
- scan: original single-mutation workflow + scan_single_mutations demo
- nls: NLS-targeted disruption in a high-nucleus protein (single + multi mutation)
- membrane: TM-targeted disruption in a high-membrane protein
- combined: apply top 3 loss-direction mutations from scan simultaneously
- all: run everything
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.variant_effect import VariantEffectPredictor  # noqa: E402

HYDROPHOBIC_TM = set("LVIAF")


def _predict_map(predictor: VariantEffectPredictor, seq: str) -> Dict[str, float]:
    emb = predictor.embed_sequence(seq)
    probs = predictor._predict_proba_from_embeddings(emb[None, :])[0]
    return predictor._dict_from_probs(probs)


def _pick_protein_with_high_label(
    df: pd.DataFrame,
    predictor: VariantEffectPredictor,
    *,
    label: str,
    min_prob: float,
    max_scan: int,
    min_len: int = 120,
    max_len: int = 700,
) -> Tuple[str, str, Dict[str, float]]:
    required = {"ACC", "Sequence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    pool = df.copy()
    if label in pool.columns:
        pool = pool[pool[label] == 1].copy()
    pool["seq_len"] = pool["Sequence"].astype(str).str.len()
    pool = pool[(pool["seq_len"] >= min_len) & (pool["seq_len"] <= max_len)].sort_values("seq_len", ascending=True)
    tried = 0
    for _, row in pool.iterrows():
        if tried >= max_scan:
            break
        tried += 1
        seq = str(row["Sequence"]).upper().strip()
        pmap = _predict_map(predictor, seq)
        if float(pmap.get(label, 0.0)) > float(min_prob):
            return str(row["ACC"]), seq, pmap
    raise RuntimeError(f"No protein found with P({label}) > {min_prob} in first {tried} candidates.")


def _top_residues(residue_scores: Sequence[Tuple[int, str, float]], n: int = 20) -> List[Tuple[int, str, float]]:
    return sorted(residue_scores, key=lambda x: abs(float(x[2])), reverse=True)[:n]


def _print_effect(name: str, predictor: VariantEffectPredictor, effect: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    print(predictor.format_report(effect))
    print("\n--- Raw JSON ---")
    print(json.dumps(effect, indent=2, default=str))


def _run_scan_case(
    predictor: VariantEffectPredictor,
    df: pd.DataFrame,
    max_scan: int,
) -> Dict[str, Any]:
    acc, seq, pmap = _pick_protein_with_high_label(
        df,
        predictor,
        label="Nucleus",
        min_prob=0.6,
        max_scan=max_scan,
        min_len=120,
        max_len=500,
    )
    print(f"\n[scan] Selected ACC={acc}, length={len(seq)}, P(Nucleus)={pmap.get('Nucleus', 0):.4f}")

    ig = predictor.interpreter.get_integrated_gradients(seq, target_location="Nucleus")
    ranked = _top_residues(ig["residue_scores"], n=40)
    single_mut: Optional[Tuple[int, str, str]] = None
    for pos, aa, _ in ranked:
        aa_u = str(aa).upper()
        if aa_u == "K":
            single_mut = (int(pos), "K", "A")
            break
        if aa_u == "R":
            single_mut = (int(pos), "R", "W")
            break
    if single_mut is None:
        raise RuntimeError("No K/R among top important Nucleus residues for scan case.")

    print(f"[scan] Chosen mutation from important residue: {single_mut[0]}{single_mut[1]}>{single_mut[2]}")
    effect = predictor.predict_variant_effect(seq, [single_mut])
    _print_effect("TEST CASE 1 - Existing scan workflow", predictor, effect)

    print("\n[scan] Running scan_single_mutations on positions 1-50 (step=5) ...")
    scan = predictor.scan_single_mutations(
        seq,
        region_start=1,
        region_end=min(50, len(seq)),
        step=5,
        top_k=20,
    )
    print(f"[scan] Scored variants: {scan['total_variants_scored']} in {scan['time_seconds']:.2f}s")
    print("[scan] Top 10 impactful mutations:")
    for row in (scan["top_mutations"] or [])[:10]:
        print(
            f"  {row['position']}{row['original_aa']}>{row['mutant_aa']} | "
            f"max_delta={row['max_delta']:+.4f} | "
            f"{row['most_affected_location']} ({row['direction']})"
        )
    return {"acc": acc, "sequence": seq, "scan": scan}


def _run_nls_case(
    predictor: VariantEffectPredictor,
    df: pd.DataFrame,
    max_scan: int,
) -> Dict[str, Any]:
    acc, seq, pmap = _pick_protein_with_high_label(
        df,
        predictor,
        label="Nucleus",
        min_prob=0.7,
        max_scan=max_scan,
        min_len=120,
        max_len=700,
    )
    print(f"\n[nls] Selected ACC={acc}, length={len(seq)}, P(Nucleus)={pmap.get('Nucleus', 0):.4f}")

    ig = predictor.interpreter.get_integrated_gradients(seq, target_location="Nucleus")
    top20 = _top_residues(ig["residue_scores"], n=20)
    kr_top = [(int(p), str(a).upper(), float(s)) for p, a, s in top20 if str(a).upper() in {"K", "R"}]
    if not kr_top:
        raise RuntimeError("No K/R in top 20 important Nucleus residues.")

    anchor_pos, anchor_aa, _ = kr_top[0]
    single_mut = (anchor_pos, anchor_aa, "A")

    nearby: List[Tuple[int, str, str]] = [single_mut]
    for p, a, _s in kr_top[1:]:
        if abs(p - anchor_pos) <= 8:
            nearby.append((p, a, "A"))
        if len(nearby) >= 3:
            break
    if len(nearby) < 2:
        for i, aa in enumerate(seq, start=1):
            if aa in {"K", "R"} and abs(i - anchor_pos) <= 12 and i != anchor_pos:
                nearby.append((i, aa, "A"))
            if len(nearby) >= 3:
                break

    print(f"[nls] Single mutation: {single_mut[0]}{single_mut[1]}>{single_mut[2]}")
    print(f"[nls] Multi mutation set: {', '.join(f'{p}{o}>{m}' for p, o, m in nearby)}")

    eff_single = predictor.predict_variant_effect(seq, [single_mut])
    eff_multi = predictor.predict_variant_effect(seq, nearby[:3])

    _print_effect("TEST CASE 2A - NLS disruption (single)", predictor, eff_single)
    _print_effect("TEST CASE 2B - NLS disruption (multi nearby)", predictor, eff_multi)

    print("\n[nls] Comparison summary:")
    print(
        f"  Single delta Nucleus: {eff_single['deltas'].get('Nucleus', 0.0):+.4f} | "
        f"Multi delta Nucleus: {eff_multi['deltas'].get('Nucleus', 0.0):+.4f}"
    )
    print(
        f"  Single risk: {eff_single['mislocalization_risk']} | "
        f"Multi risk: {eff_multi['mislocalization_risk']}"
    )
    return {"acc": acc, "sequence": seq, "single": eff_single, "multi": eff_multi}


def _run_membrane_case(
    predictor: VariantEffectPredictor,
    df: pd.DataFrame,
    max_scan: int,
) -> Dict[str, Any]:
    acc, seq, pmap = _pick_protein_with_high_label(
        df,
        predictor,
        label="Membrane",
        min_prob=0.7,
        max_scan=max_scan,
        min_len=120,
        max_len=900,
    )
    print(f"\n[membrane] Selected ACC={acc}, length={len(seq)}, P(Membrane)={pmap.get('Membrane', 0):.4f}")

    ig = predictor.interpreter.get_integrated_gradients(seq, target_location="Membrane")
    top30 = _top_residues(ig["residue_scores"], n=30)

    chosen: Optional[Tuple[int, str, str]] = None
    for p, aa, _ in top30:
        a = str(aa).upper()
        if a not in HYDROPHOBIC_TM:
            continue
        if a == "V":
            chosen = (int(p), "V", "K")  # requested example
        else:
            chosen = (int(p), a, "D")  # requested style (e.g., L->D)
        break
    if chosen is None:
        raise RuntimeError("No hydrophobic residue (L/V/I/A/F) in top membrane-important positions.")

    print(f"[membrane] Mutation: {chosen[0]}{chosen[1]}>{chosen[2]}")
    effect = predictor.predict_variant_effect(seq, [chosen])
    _print_effect("TEST CASE 3 - Transmembrane disruption", predictor, effect)
    return {"acc": acc, "sequence": seq, "effect": effect}


def _run_combined_case(
    predictor: VariantEffectPredictor,
    accession: str,
    sequence: str,
    scan: Mapping[str, Any],
) -> Dict[str, Any]:
    top = list(scan.get("top_mutations") or [])
    losses = [x for x in top if str(x.get("direction", "")) == "loss"]
    if len(losses) < 3:
        raise RuntimeError("Need at least 3 loss-direction mutations in scan results for combined test.")
    sel = losses[:3]
    muts = [(int(x["position"]), str(x["original_aa"]), str(x["mutant_aa"])) for x in sel]
    print(f"\n[combined] Using scan-case protein ACC={accession}, length={len(sequence)}")
    print("[combined] Using top 3 loss-direction mutations:")
    for m in muts:
        print(f"  - {m[0]}{m[1]}>{m[2]}")

    combined = predictor.predict_variant_effect(sequence, muts)
    _print_effect("TEST CASE 4 - Combined top-loss mutations", predictor, combined)

    print("\n[combined] Individual vs combined most-affected deltas:")
    for row in sel:
        print(
            f"  {row['position']}{row['original_aa']}>{row['mutant_aa']}: "
            f"{row['most_affected_location']} {float(row['max_delta']):+.4f}"
        )
    print(
        f"  Combined: {combined['most_affected_location']} "
        f"{float(combined['max_delta']):+.4f}"
    )
    return {"acc": accession, "sequence": sequence, "combined": combined, "mutations": muts}


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke tests for VariantEffectPredictor.")
    p.add_argument("--classifier-path", type=Path, default=ROOT / "models" / "best_model.pt")
    p.add_argument("--csv-path", type=Path, default=ROOT / "data" / "processed" / "deeploc_multilabel.csv")
    p.add_argument("--device", default="cuda", help="Device (default: cuda).")
    p.add_argument("--max-scan", type=int, default=400)
    p.add_argument(
        "--test-case",
        choices=["all", "scan", "nls", "membrane", "combined"],
        default="all",
        help="Which test workflow to run (default: all).",
    )
    args = p.parse_args()

    classifier_path = args.classifier_path if args.classifier_path.is_absolute() else (ROOT / args.classifier_path).resolve()
    csv_path = args.csv_path if args.csv_path.is_absolute() else (ROOT / args.csv_path).resolve()
    if not classifier_path.is_file():
        raise FileNotFoundError(f"Missing classifier: {classifier_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {csv_path}")

    predictor = VariantEffectPredictor(classifier_path=classifier_path, device=args.device)
    df = pd.read_csv(csv_path)

    want_all = args.test_case == "all"
    scan_bundle: Optional[Dict[str, Any]] = None

    if want_all or args.test_case == "scan":
        scan_bundle = _run_scan_case(predictor, df, max_scan=args.max_scan)

    if want_all or args.test_case == "nls":
        _run_nls_case(predictor, df, max_scan=args.max_scan)

    if want_all or args.test_case == "membrane":
        _run_membrane_case(predictor, df, max_scan=args.max_scan)

    if want_all or args.test_case == "combined":
        if scan_bundle is None:
            scan_bundle = _run_scan_case(predictor, df, max_scan=args.max_scan)
        _run_combined_case(
            predictor,
            accession=str(scan_bundle["acc"]),
            sequence=str(scan_bundle["sequence"]),
            scan=scan_bundle["scan"],
        )


if __name__ == "__main__":
    main()


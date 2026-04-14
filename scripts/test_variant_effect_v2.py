"""
Attention-targeted smoke tests for residue-level variant effects.

Test A (Nucleus):
1) find a protein with moderate P(Nucleus) in [0.5, 0.8]
2) pick highest-attention K/R residue
3) mutate it to A

Test B (Membrane short proteins):
1) find a short protein (50-150 aa) with P(Membrane) > 0.5
2) pick highest-attention hydrophobic residue (L/V/I/F)
3) mutate it to D

Also keeps the N-terminal wipeout test for Test A (mutate K/R in positions 1-30 to A).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.residue_classifier import FALLBACK_LABEL_NAMES, ResidueLocalizationClassifier  # noqa: E402
from src.utils.device import resolve_torch_device  # noqa: E402

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1024


def _load_sequences_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"ACC", "Sequence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("CSV is empty.")
    df = df.copy()
    df["ACC"] = df["ACC"].astype(str)
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df = df[df["Sequence"].str.len() > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No non-empty sequences in CSV.")
    return df


def _load_models(
    classifier_path: Path,
    device_req: str | None,
) -> Tuple[torch.device, Any, Any, ResidueLocalizationClassifier, List[str]]:
    device = resolve_torch_device(device_req)
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm = AutoModel.from_pretrained(
        ESM_MODEL_NAME,
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )
    esm.eval().to(device)

    ckpt = torch.load(classifier_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported residue checkpoint format.")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    embedding_dim = int(ckpt.get("embedding_dim", 1280))
    num_labels = int(ckpt.get("num_labels", 11))
    label_names = list(ckpt.get("label_names") or FALLBACK_LABEL_NAMES[:num_labels])
    if len(label_names) != num_labels:
        raise ValueError("Checkpoint label_names length does not match num_labels.")
    model = ResidueLocalizationClassifier(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        label_names=label_names,
        dropout=float(ckpt.get("dropout", 0.3)),
        num_heads=int(ckpt.get("num_heads", 4)),
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return device, tokenizer, esm, model, label_names


@torch.inference_mode()
def _embed_residue_sequence(
    sequence: str,
    tokenizer: Any,
    esm: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = sequence.upper().strip()
    toks = tokenizer(
        [seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = esm(**toks, return_dict=True)
    hidden = out.last_hidden_state
    attn = toks["attention_mask"]
    valid_len = int(attn[0].sum().item())
    if valid_len < 3:
        raise ValueError("Tokenized sequence too short for residue embedding extraction.")
    core = hidden[0, 1 : valid_len - 1, :].float()  # strip special tokens -> (L, 1280)
    mask = torch.ones(core.shape[0], dtype=torch.bool, device=device)
    return core.unsqueeze(0), mask.unsqueeze(0)


@torch.inference_mode()
def _predict_with_attention(
    sequence: str,
    tokenizer: Any,
    esm: Any,
    classifier: ResidueLocalizationClassifier,
    label_names: Sequence[str],
    device: torch.device,
) -> Tuple[Dict[str, float], List[float]]:
    x, mask = _embed_residue_sequence(sequence, tokenizer, esm, device)
    logits, attn = classifier.get_attention_weights(x, mask=mask)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    pred = {str(label_names[i]): float(probs[i]) for i in range(len(label_names))}
    attn_vec = attn[0].detach().cpu().numpy().tolist()
    return pred, [float(x) for x in attn_vec]


def _apply_mutations(sequence: str, mutations: Sequence[Tuple[int, str, str]]) -> str:
    seq = list(sequence.upper().strip())
    n = len(seq)
    for pos, orig, mut in mutations:
        p = int(pos)
        if p < 1 or p > n:
            raise ValueError(f"Mutation position {p} out of range for length {n}")
        o = str(orig).upper()
        m = str(mut).upper()
        if seq[p - 1] != o:
            raise ValueError(f"Original AA mismatch at {p}: sequence has {seq[p-1]!r}, mutation expects {o!r}")
        seq[p - 1] = m
    return "".join(seq)


def _risk_from_delta(abs_delta: float) -> str:
    if abs_delta > 0.3:
        return "high"
    if abs_delta >= 0.15:
        return "medium"
    if abs_delta >= 0.05:
        return "low"
    return "none"


def _best_attention_index(sequence: str, attention: Sequence[float], allowed: set[str]) -> Optional[int]:
    best_idx: Optional[int] = None
    best_score = -1.0
    n = min(len(sequence), len(attention))
    for i in range(n):
        aa = sequence[i]
        if aa not in allowed:
            continue
        score = float(attention[i])
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _top_attention_indices(
    sequence: str,
    attention: Sequence[float],
    *,
    k: int,
    allowed: Optional[set[str]] = None,
    exclude_target_aa: Optional[str] = None,
) -> List[int]:
    ranked: List[Tuple[float, int]] = []
    n = min(len(sequence), len(attention))
    for i in range(n):
        aa = sequence[i]
        if allowed is not None and aa not in allowed:
            continue
        if exclude_target_aa is not None and aa == exclude_target_aa:
            continue
        ranked.append((float(attention[i]), i))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return [idx for _score, idx in ranked[: max(1, int(k))]]


def _find_case(
    df: pd.DataFrame,
    tokenizer: Any,
    esm: Any,
    classifier: ResidueLocalizationClassifier,
    label_names: Sequence[str],
    device: torch.device,
    target_label: str,
    prob_min: float,
    prob_max: float,
    allowed_residues: set[str],
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
    max_scan: int = 500,
) -> Tuple[str, str, Dict[str, float], List[float], int, str]:
    scanned = 0
    for _, row in df.iterrows():
        if scanned >= max_scan:
            break
        acc = str(row["ACC"])
        seq = str(row["Sequence"]).upper().strip()
        if length_min is not None and len(seq) < int(length_min):
            continue
        if length_max is not None and len(seq) > int(length_max):
            continue
        scanned += 1
        pred, attn = _predict_with_attention(seq, tokenizer, esm, classifier, label_names, device)
        p = float(pred.get(target_label, 0.0))
        if p < prob_min or p > prob_max:
            continue
        idx = _best_attention_index(seq, attn, allowed_residues)
        if idx is None:
            continue
        return acc, seq, pred, attn, idx, seq[idx]
    raise RuntimeError(
        f"Could not find case for label={target_label!r}, prob in [{prob_min}, {prob_max}], "
        f"allowed residues={sorted(allowed_residues)} after scanning {scanned} candidate proteins."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-targeted residue variant smoke tests.")
    parser.add_argument("--classifier-path", type=Path, default=ROOT / "models" / "best_residue_model.pt")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "data" / "processed" / "deeploc_multilabel.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scan", type=int, default=500, help="Max proteins to scan while searching test cases.")
    args = parser.parse_args()

    classifier_path = args.classifier_path if args.classifier_path.is_absolute() else (ROOT / args.classifier_path).resolve()
    csv_path = args.csv_path if args.csv_path.is_absolute() else (ROOT / args.csv_path).resolve()
    if not classifier_path.is_file():
        raise FileNotFoundError(f"Missing classifier: {classifier_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {csv_path}")

    device, tokenizer, esm, classifier, label_names = _load_models(
        classifier_path=classifier_path,
        device_req=args.device,
    )

    df = _load_sequences_from_csv(csv_path)

    # ---------------- Test A: moderate-confidence Nucleus ----------------
    print("\n=== TEST A: Moderate-confidence Nucleus (0.5-0.8), mutate top-attention K/R -> A ===")
    acc_n, seq_n, pred_n, attn_n, idx_n, aa_n = _find_case(
        df,
        tokenizer,
        esm,
        classifier,
        label_names,
        device,
        target_label="Nucleus",
        prob_min=0.5,
        prob_max=0.8,
        allowed_residues={"K", "R"},
        max_scan=max(1, int(args.max_scan)),
    )
    print(f"Selected ACC={acc_n} | length={len(seq_n)} | P(Nucleus)={pred_n.get('Nucleus', 0.0):.4f}")
    print(f"Top-attention basic residue: index {idx_n} (position {idx_n + 1}) = {aa_n}, attn={attn_n[idx_n]:.6f}")

    mut_n = [(idx_n + 1, aa_n, "A")]
    seq_n_mut = _apply_mutations(seq_n, mut_n)
    pred_n_mut, _ = _predict_with_attention(seq_n_mut, tokenizer, esm, classifier, label_names, device)
    print("\nOriginal vs mutant predictions (all locations) [Nucleus targeted]:")
    deltas_nuc: Dict[str, float] = {}
    for label in sorted(pred_n.keys()):
        p0 = float(pred_n[label])
        p1 = float(pred_n_mut[label])
        d = p1 - p0
        deltas_nuc[label] = d
        print(f"  {label:24s} {p0:.4f} -> {p1:.4f}  (delta {d:+.4f})")

    most_aff_nuc = max(deltas_nuc, key=lambda k: abs(deltas_nuc[k]))
    max_abs_delta_nuc = abs(float(deltas_nuc[most_aff_nuc]))
    print("\nClinical summary [Nucleus targeted]:")
    print(
        f"Single mutation {idx_n + 1}{aa_n}>A: most affected location={most_aff_nuc}, "
        f"delta={deltas_nuc[most_aff_nuc]:+.4f}"
    )
    print(f"Mislocalization risk: {_risk_from_delta(max_abs_delta_nuc)}")
    print(
        f"Key question (single R/K->A): max |delta| = {max_abs_delta_nuc:.4f} "
        f"=> {'YES' if max_abs_delta_nuc > 0.05 else 'NO'} (threshold > 0.05)"
    )

    # Additional stress test: mutate ALL K/R in N-terminal positions 1-30 to A.
    n_term_end = min(30, len(seq_n))
    n_term_mutations: List[Tuple[int, str, str]] = []
    for idx in range(0, n_term_end):  # idx is 0-based
        aa = seq_n[idx]
        if aa in {"K", "R"}:
            n_term_mutations.append((idx + 1, aa, "A"))  # predictor expects 1-based positions

    print("\n--- N-terminal basic-signal wipeout test (positions 1-30) ---")
    if not n_term_mutations:
        print("No K/R residues found in positions 1-30; skipping combined N-terminal mutation test.")
        return

    print(f"Found {len(n_term_mutations)} K/R residues in positions 1-30; mutating all to A.")
    print("Mutations:")
    print("  " + ", ".join(f"{p}{o}>A" for p, o, _ in n_term_mutations))

    nterm_mutant_sequence = _apply_mutations(seq_n, n_term_mutations)
    predm_nterm, _ = _predict_with_attention(nterm_mutant_sequence, tokenizer, esm, classifier, label_names, device)
    print("\nOriginal vs mutant predictions (all locations) [N-term combined mutation]:")
    deltas_nterm: Dict[str, float] = {}
    for label in sorted(pred_n.keys()):
        p0 = float(pred_n[label])
        p1 = float(predm_nterm[label])
        d = p1 - p0
        deltas_nterm[label] = d
        print(f"  {label:24s} {p0:.4f} -> {p1:.4f}  (delta {d:+.4f})")
    most_aff_n = max(deltas_nterm, key=lambda k: abs(deltas_nterm[k]))
    print("\nClinical summary [N-term combined mutation]:")
    print(
        f"N-term wipeout: most affected location={most_aff_n}, "
        f"delta={deltas_nterm[most_aff_n]:+.4f}"
    )
    print(
        f"\nMislocalization risk [N-term combined mutation]: "
        f"{_risk_from_delta(abs(float(deltas_nterm[most_aff_n])))}"
    )

    # ---------------- Test B: short membrane proteins ----------------
    print("\n=== TEST B: Short Membrane protein (50-150 aa), mutate top-attention L/V/I/F -> D ===")
    acc_m, seq_m, pred_m, attn_m, idx_m, aa_m = _find_case(
        df,
        tokenizer,
        esm,
        classifier,
        label_names,
        device,
        target_label="Membrane",
        prob_min=0.5,
        prob_max=1.0,
        allowed_residues={"L", "V", "I", "F"},
        length_min=50,
        length_max=150,
        max_scan=max(1, int(args.max_scan)),
    )
    print(f"Selected ACC={acc_m} | length={len(seq_m)} | P(Membrane)={pred_m.get('Membrane', 0.0):.4f}")
    print(
        f"Top-attention hydrophobic residue: index {idx_m} (position {idx_m + 1}) = {aa_m}, "
        f"attn={attn_m[idx_m]:.6f}"
    )

    mut_m = [(idx_m + 1, aa_m, "D")]
    seq_m_mut = _apply_mutations(seq_m, mut_m)
    pred_m_mut, _ = _predict_with_attention(seq_m_mut, tokenizer, esm, classifier, label_names, device)
    print("\nOriginal vs mutant predictions (all locations) [Membrane targeted]:")
    deltas_mem: Dict[str, float] = {}
    for label in sorted(pred_m.keys()):
        p0 = float(pred_m[label])
        p1 = float(pred_m_mut[label])
        d = p1 - p0
        deltas_mem[label] = d
        print(f"  {label:24s} {p0:.4f} -> {p1:.4f}  (delta {d:+.4f})")
    most_aff_mem = max(deltas_mem, key=lambda k: abs(deltas_mem[k]))
    print("\nClinical summary [Membrane targeted]:")
    print(
        f"Single mutation {idx_m + 1}{aa_m}>D: most affected location={most_aff_mem}, "
        f"delta={deltas_mem[most_aff_mem]:+.4f}"
    )
    print(f"Mislocalization risk: {_risk_from_delta(abs(float(deltas_mem[most_aff_mem])))}")

    # ---------------- Test C: same protein as Test B (ACC=Q6QNY1), top-3 hydrophobic -> D ----------------
    print("\n=== TEST C: ACC=Q6QNY1, top-3 attention hydrophobic (L/V/I/F/W/M) -> D ===")
    row_q = df[df["ACC"] == "Q6QNY1"]
    if len(row_q) == 0:
        print("ACC Q6QNY1 not found in CSV; skipping Test C and Test D.")
        return
    seq_q = str(row_q.iloc[0]["Sequence"]).upper().strip()
    print(f"Selected ACC=Q6QNY1 | length={len(seq_q)}")
    if len(seq_q) != 142:
        print("Warning: expected length 142 for Q6QNY1, got " + str(len(seq_q)))

    pred_q, attn_q = _predict_with_attention(seq_q, tokenizer, esm, classifier, label_names, device)
    top3_h = _top_attention_indices(
        seq_q,
        attn_q,
        k=3,
        allowed={"L", "V", "I", "F", "W", "M"},
        exclude_target_aa="D",
    )
    if len(top3_h) < 3:
        print("Could not find 3 hydrophobic residues for Test C; skipping.")
    else:
        mut_c = [(i + 1, seq_q[i], "D") for i in top3_h]
        print("Mutations:")
        print("  " + ", ".join(f"{p}{o}>D (attn={attn_q[p-1]:.6f})" for p, o, _ in mut_c))
        seq_q_c = _apply_mutations(seq_q, mut_c)
        pred_q_c, _ = _predict_with_attention(seq_q_c, tokenizer, esm, classifier, label_names, device)
        print("\nOriginal vs mutant predictions (all locations) [Test C]:")
        deltas_c: Dict[str, float] = {}
        for label in sorted(pred_q.keys()):
            p0 = float(pred_q[label])
            p1 = float(pred_q_c[label])
            d = p1 - p0
            deltas_c[label] = d
            print(f"  {label:24s} {p0:.4f} -> {p1:.4f}  (delta {d:+.4f})")
        most_aff_c = max(deltas_c, key=lambda k: abs(deltas_c[k]))
        max_abs_c = abs(float(deltas_c[most_aff_c]))
        print(
            f"\nClinical summary [Test C]: most affected location={most_aff_c}, "
            f"delta={deltas_c[most_aff_c]:+.4f}"
        )
        print(f"Mislocalization risk [Test C]: {_risk_from_delta(max_abs_c)}")
        print(f"Key check [Test C]: max |delta| = {max_abs_c:.4f} => {'YES' if max_abs_c > 0.05 else 'NO'}")

    # ---------------- Test D: same protein, top-5 attention residues (any AA) -> A ----------------
    print("\n=== TEST D: ACC=Q6QNY1, top-5 attention residues (any AA) -> A ===")
    top5_any = _top_attention_indices(
        seq_q,
        attn_q,
        k=5,
        allowed=None,
        exclude_target_aa="A",
    )
    if len(top5_any) < 5:
        print("Could not find 5 mutable residues for Test D; skipping.")
        return
    mut_d = [(i + 1, seq_q[i], "A") for i in top5_any]
    print("Mutations:")
    print("  " + ", ".join(f"{p}{o}>A (attn={attn_q[p-1]:.6f})" for p, o, _ in mut_d))
    seq_q_d = _apply_mutations(seq_q, mut_d)
    pred_q_d, _ = _predict_with_attention(seq_q_d, tokenizer, esm, classifier, label_names, device)
    print("\nOriginal vs mutant predictions (all locations) [Test D]:")
    deltas_d: Dict[str, float] = {}
    for label in sorted(pred_q.keys()):
        p0 = float(pred_q[label])
        p1 = float(pred_q_d[label])
        d = p1 - p0
        deltas_d[label] = d
        print(f"  {label:24s} {p0:.4f} -> {p1:.4f}  (delta {d:+.4f})")
    most_aff_d = max(deltas_d, key=lambda k: abs(deltas_d[k]))
    max_abs_d = abs(float(deltas_d[most_aff_d]))
    print(
        f"\nClinical summary [Test D]: most affected location={most_aff_d}, "
        f"delta={deltas_d[most_aff_d]:+.4f}"
    )
    print(f"Mislocalization risk [Test D]: {_risk_from_delta(max_abs_d)}")
    print(f"Key check [Test D]: max |delta| = {max_abs_d:.4f} => {'YES' if max_abs_d > 0.05 else 'NO'}")


if __name__ == "__main__":
    main()


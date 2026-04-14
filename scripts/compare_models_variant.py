"""
Controlled side-by-side variant-effect comparison:
mean-pooled classifier vs residue-attention classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.variant_effect import VariantEffectPredictor  # noqa: E402
from src.models.residue_classifier import FALLBACK_LABEL_NAMES, ResidueLocalizationClassifier  # noqa: E402
from src.utils.device import resolve_torch_device  # noqa: E402

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1024

TESTS: Dict[str, List[Tuple[int, str, str]]] = {
    "Test B": [(45, "F", "D")],
    "Test C": [(45, "F", "D"), (9, "L", "D"), (8, "V", "D")],
    "Test D": [(142, "R", "A"), (141, "K", "A"), (45, "F", "A"), (41, "C", "A"), (9, "L", "A")],
}


class ResidueVariantEffectPredictor:
    """Simple residue-level analog of VariantEffectPredictor.predict_variant_effect."""

    def __init__(
        self,
        classifier_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.classifier_path = Path(classifier_path).expanduser().resolve()
        if not self.classifier_path.is_file():
            raise FileNotFoundError(f"Missing residue classifier checkpoint: {self.classifier_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.esm_model = AutoModel.from_pretrained(
            ESM_MODEL_NAME,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )
        self.esm_model.eval().to(self.device)

        ckpt = torch.load(self.classifier_path, map_location="cpu")
        if not isinstance(ckpt, Mapping):
            raise ValueError("Unsupported residue checkpoint format.")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        embedding_dim = int(ckpt.get("embedding_dim", 1280))
        num_labels = int(ckpt.get("num_labels", 11))
        label_names = list(ckpt.get("label_names") or FALLBACK_LABEL_NAMES[:num_labels])
        if len(label_names) != num_labels:
            raise ValueError("Residue checkpoint label_names length mismatch.")
        self.label_names = label_names

        self.classifier = ResidueLocalizationClassifier(
            embedding_dim=embedding_dim,
            num_labels=num_labels,
            label_names=label_names,
            dropout=float(ckpt.get("dropout", 0.3)),
            num_heads=int(ckpt.get("num_heads", 4)),
        )
        self.classifier.load_state_dict(state, strict=True)
        self.classifier.eval().to(self.device)

    @torch.inference_mode()
    def _predict_proba(self, sequence: str) -> Dict[str, float]:
        seq = sequence.upper().strip()
        toks = self.tokenizer(
            [seq],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.esm_model(**toks, return_dict=True)
        hidden = out.last_hidden_state
        attn_mask = toks["attention_mask"]
        valid_len = int(attn_mask[0].sum().item())
        if valid_len < 3:
            raise ValueError("Tokenized sequence too short.")
        core = hidden[0, 1 : valid_len - 1, :].float()
        x = core.unsqueeze(0)
        mask = torch.ones((1, core.shape[0]), dtype=torch.bool, device=self.device)
        logits = self.classifier(x, mask=mask)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
        return {self.label_names[i]: float(probs[i]) for i in range(len(self.label_names))}

    @staticmethod
    def _apply_mutations(sequence: str, mutations: Sequence[Tuple[int, str, str]]) -> str:
        seq = list(sequence.upper().strip())
        n = len(seq)
        for pos, orig, mut in mutations:
            if pos < 1 or pos > n:
                raise ValueError(f"Mutation position {pos} out of range for length {n}")
            if seq[pos - 1] != orig:
                raise ValueError(
                    f"Original AA mismatch at {pos}: expected {seq[pos - 1]!r}, got mutation original {orig!r}"
                )
            seq[pos - 1] = mut
        return "".join(seq)

    def predict_variant_effect(
        self,
        original_sequence: str,
        mutations: Sequence[Tuple[int, str, str]],
    ) -> Dict[str, Any]:
        p0 = self._predict_proba(original_sequence)
        seqm = self._apply_mutations(original_sequence, mutations)
        pm = self._predict_proba(seqm)
        deltas = {k: float(pm[k] - p0[k]) for k in self.label_names}
        return {
            "original_predictions": p0,
            "mutant_predictions": pm,
            "deltas": deltas,
            "mutant_sequence": seqm,
        }


def _load_q6qny1_sequence(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    req = {"ACC", "Sequence"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(req)} columns.")
    rows = df[df["ACC"].astype(str) == "Q6QNY1"]
    if len(rows) == 0:
        raise ValueError("ACC Q6QNY1 not found in CSV.")
    seq = str(rows.iloc[0]["Sequence"]).upper().strip()
    if len(seq) != 142:
        print(f"Warning: expected length 142 for Q6QNY1, got {len(seq)}")
    return seq


def _print_test_table(
    test_name: str,
    mean_deltas: Dict[str, float],
    residue_deltas: Dict[str, float],
) -> None:
    print(f"\n=== {test_name} ===")
    print("Location                  | Mean-pooled delta | Residue delta    | Improvement")
    print("--------------------------+-------------------+------------------+------------")
    for loc in sorted(mean_deltas.keys()):
        m = float(mean_deltas[loc])
        r = float(residue_deltas.get(loc, 0.0))
        denom = abs(m)
        if denom < 1e-9:
            imp = "inf" if abs(r) > 0 else "1.0x"
        else:
            imp = f"{(abs(r) / denom):.1f}x"
        print(f"{loc:26s} | {m:+.4f}            | {r:+.4f}           | {imp}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled comparison: mean-pooled vs residue variant-effect sensitivity.")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "data" / "processed" / "deeploc_multilabel.csv")
    parser.add_argument("--mean-model", type=Path, default=ROOT / "models" / "best_model.pt")
    parser.add_argument("--residue-model", type=Path, default=ROOT / "models" / "best_residue_model.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    csv_path = args.csv_path if args.csv_path.is_absolute() else (ROOT / args.csv_path).resolve()
    mean_model = args.mean_model if args.mean_model.is_absolute() else (ROOT / args.mean_model).resolve()
    residue_model = args.residue_model if args.residue_model.is_absolute() else (ROOT / args.residue_model).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not mean_model.is_file():
        raise FileNotFoundError(f"Missing mean-pooled checkpoint: {mean_model}")
    if not residue_model.is_file():
        raise FileNotFoundError(f"Missing residue checkpoint: {residue_model}")

    sequence = _load_q6qny1_sequence(csv_path)
    print(f"Protein: ACC=Q6QNY1 | length={len(sequence)}")

    mean_predictor = VariantEffectPredictor(classifier_path=mean_model, device=args.device)
    residue_predictor = ResidueVariantEffectPredictor(classifier_path=residue_model, device=args.device)

    all_abs_mean: List[float] = []
    all_abs_res: List[float] = []

    for test_name, muts in TESTS.items():
        print("Mutations: " + ", ".join(f"{p}{o}>{m}" for p, o, m in muts))
        out_mean = mean_predictor.predict_variant_effect(sequence, muts)
        out_res = residue_predictor.predict_variant_effect(sequence, muts)
        d_mean = {k: float(v) for k, v in (out_mean.get("deltas") or {}).items()}
        d_res = {k: float(v) for k, v in (out_res.get("deltas") or {}).items()}
        _print_test_table(test_name, d_mean, d_res)
        all_abs_mean.extend(abs(x) for x in d_mean.values())
        all_abs_res.extend(abs(x) for x in d_res.values())

    avg_abs_mean = sum(all_abs_mean) / max(1, len(all_abs_mean))
    avg_abs_res = sum(all_abs_res) / max(1, len(all_abs_res))
    if avg_abs_mean < 1e-12:
        overall_ratio_txt = "inf"
    else:
        overall_ratio_txt = f"{(avg_abs_res / avg_abs_mean):.2f}x"

    print("\n=== Summary ===")
    print(f"Average |delta| (mean-pooled): {avg_abs_mean:.6f}")
    print(f"Average |delta| (residue):     {avg_abs_res:.6f}")
    print(f"Average sensitivity improvement across all locations/tests: {overall_ratio_txt}")


if __name__ == "__main__":
    main()


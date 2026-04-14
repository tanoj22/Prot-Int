"""
Variant effect prediction for subcellular localization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.design.utils import check_localization_signals, compare_signals, validate_sequence
from src.models.classifier import ProteinLocalizationClassifier, load_model
from src.models.interpretability import ProteinInterpreter
from src.utils.device import resolve_torch_device

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def _mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class VariantEffectPredictor:
    def __init__(
        self,
        classifier_path: str | Path = "models/best_model.pt",
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.classifier_path = Path(classifier_path).expanduser().resolve()
        if not self.classifier_path.is_file():
            raise FileNotFoundError(f"Missing classifier checkpoint: {self.classifier_path}")

        ckpt = torch.load(self.classifier_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError("Unsupported classifier checkpoint format")
        embedding_dim = int(ckpt.get("embedding_dim", 1280))

        self.classifier: ProteinLocalizationClassifier = load_model(
            self.classifier_path,
            embedding_dim=embedding_dim,
            num_labels=None,
            device=self.device,
        )
        self.label_names = list(self.classifier.label_names)
        self.label_to_idx = {n: i for i, n in enumerate(self.label_names)}

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm_model: PreTrainedModel = AutoModel.from_pretrained(
            esm_model_name,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )
        self.esm_model.to(self.device).eval()
        self.esm_model_name = esm_model_name

        self.interpreter = ProteinInterpreter(
            classifier_path=self.classifier_path,
            esm_model_name=esm_model_name,
            device=self.device,
        )

    def _tokenize_batch(self, sequences: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _predict_proba_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    def embed_sequence(self, sequence: str) -> np.ndarray:
        seq = sequence.upper().strip()
        if not seq:
            raise ValueError("Empty sequence")
        toks = self._tokenize_batch([seq])
        with torch.no_grad():
            out = self.esm_model(**toks, return_dict=True)
            pooled = _mean_pool_last_hidden(out.last_hidden_state, toks["attention_mask"])
        return pooled.detach().cpu().numpy().astype(np.float32).squeeze(0)

    def _embed_sequences_batched(
        self,
        sequences: Sequence[str],
        batch_size: int = 16,
        show_progress: bool = False,
        progress_desc: str = "Embedding variants",
    ) -> np.ndarray:
        all_out: List[np.ndarray] = []
        it = range(0, len(sequences), batch_size)
        if show_progress:
            it = tqdm(it, desc=progress_desc, unit="batch")
        for i in it:
            batch = [str(s).upper().strip() for s in sequences[i : i + batch_size]]
            toks = self._tokenize_batch(batch)
            with torch.no_grad():
                out = self.esm_model(**toks, return_dict=True)
                pooled = _mean_pool_last_hidden(out.last_hidden_state, toks["attention_mask"])
            all_out.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not all_out:
            return np.zeros((0, 1280), dtype=np.float32)
        return np.vstack(all_out)

    def _apply_mutations(self, sequence: str, mutations: Sequence[Tuple[int, str, str]]) -> str:
        seq = list(sequence.upper().strip())
        n = len(seq)
        for pos, orig, mut in mutations:
            p = int(pos)
            if p < 1 or p > n:
                raise ValueError(f"Mutation position {p} out of range for length {n}")
            o = str(orig).upper().strip()
            m = str(mut).upper().strip()
            if len(o) != 1 or len(m) != 1:
                raise ValueError(f"Mutation at {p} must be single-letter AA, got ({orig!r}, {mut!r})")
            if o not in AA20:
                raise ValueError(f"Invalid original amino acid at {p}: {o!r}")
            if m not in AA20:
                raise ValueError(f"Invalid mutant amino acid at {p}: {m!r}")
            if seq[p - 1] != o:
                raise ValueError(
                    f"Original AA mismatch at position {p}: expected {seq[p - 1]!r}, got mutation original {o!r}"
                )
            seq[p - 1] = m
        return "".join(seq)

    @staticmethod
    def _risk_from_delta(abs_delta: float) -> str:
        if abs_delta > 0.3:
            return "high"
        if abs_delta >= 0.15:
            return "medium"
        if abs_delta >= 0.05:
            return "low"
        return "none"

    def _dict_from_probs(self, probs: np.ndarray) -> Dict[str, float]:
        return {self.label_names[i]: float(probs[i]) for i in range(len(self.label_names))}

    def predict_variant_effect(
        self,
        original_sequence: str,
        mutations: Sequence[Tuple[int, str, str]],
    ) -> Dict[str, Any]:
        ok, msg = validate_sequence(original_sequence)
        if not ok:
            raise ValueError(msg)

        seq0 = original_sequence.upper().strip()
        seqm = self._apply_mutations(seq0, mutations)

        emb0 = self.embed_sequence(seq0)
        embm = self.embed_sequence(seqm)
        p0 = self._predict_proba_from_embeddings(emb0[None, :])[0]
        pm = self._predict_proba_from_embeddings(embm[None, :])[0]
        pred0 = self._dict_from_probs(p0)
        predm = self._dict_from_probs(pm)

        deltas = {name: float(predm[name] - pred0[name]) for name in self.label_names}
        most_affected = max(self.label_names, key=lambda n: abs(deltas[n]))
        max_delta = float(deltas[most_affected])
        direction = "gain" if max_delta >= 0 else "loss"
        abs_delta = abs(max_delta)
        risk = self._risk_from_delta(abs_delta)

        ig0 = self.interpreter.get_integrated_gradients(seq0, most_affected)
        hot0 = self.interpreter.identify_hot_regions(ig0["residue_scores"], window_size=10, top_percentile=90)
        igm = self.interpreter.get_integrated_gradients(seqm, most_affected)
        hotm = self.interpreter.identify_hot_regions(igm["residue_scores"], window_size=10, top_percentile=90)

        sig0 = check_localization_signals(seq0)
        sigm = check_localization_signals(seqm)
        sig_cmp = compare_signals(seq0, seqm)
        disrupted = list(sig_cmp["removed"])
        gained = list(sig_cmp["added"])

        mut_txt = ", ".join(f"{p}{o}>{m}" for p, o, m in mutations)
        top_gain = max(self.label_names, key=lambda n: deltas[n])
        top_loss = min(self.label_names, key=lambda n: deltas[n])
        summary = (
            f"Mutation(s) {mut_txt} most strongly affect {most_affected} ({direction}, delta={max_delta:+.3f}). "
            f"P({most_affected}) changes {pred0[most_affected]:.2f} -> {predm[most_affected]:.2f}. "
            f"Largest gain: {top_gain} ({deltas[top_gain]:+.3f}), largest loss: {top_loss} ({deltas[top_loss]:+.3f})."
        )
        if disrupted:
            summary += f" Disrupted signal(s): {', '.join(disrupted)}."
        if gained:
            summary += f" Gained signal(s): {', '.join(gained)}."

        return {
            "original_sequence": seq0,
            "mutant_sequence": seqm,
            "mutations": [(int(p), str(o).upper(), str(m).upper()) for p, o, m in mutations],
            "original_predictions": pred0,
            "mutant_predictions": predm,
            "deltas": deltas,
            "most_affected_location": most_affected,
            "max_delta": max_delta,
            "direction": direction,
            "signals_original": sig0,
            "signals_mutant": sigm,
            "signals_disrupted": disrupted,
            "signals_gained": gained,
            "interpretation_original": {
                "residue_scores": ig0["residue_scores"],
                "hot_regions": hot0,
            },
            "interpretation_mutant": {
                "residue_scores": igm["residue_scores"],
                "hot_regions": hotm,
            },
            "clinical_summary": summary,
            "mislocalization_risk": risk,
            "validation_message": msg,
        }

    def scan_single_mutations(
        self,
        sequence: str,
        region_start: Optional[int] = None,
        region_end: Optional[int] = None,
        step: int = 1,
        top_k: int = 20,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        ok, msg = validate_sequence(sequence)
        if not ok:
            raise ValueError(msg)
        seq = sequence.upper().strip()
        n = len(seq)

        rs = int(region_start) if region_start is not None else 1
        re = int(region_end) if region_end is not None else n
        if rs < 1 or re > n or rs > re:
            raise ValueError(f"Invalid region [{rs}, {re}] for sequence length {n}")
        step = max(1, int(step))
        top_k = max(1, int(top_k))

        t0 = time.perf_counter()
        base_emb = self.embed_sequence(seq)
        base_probs = self._predict_proba_from_embeddings(base_emb[None, :])[0]
        base_map = self._dict_from_probs(base_probs)

        variants: List[Tuple[int, str, str, str]] = []
        positions = list(range(rs, re + 1, step))
        for pos in positions:
            orig = seq[pos - 1]
            for aa in sorted(AA20):
                if aa == orig:
                    continue
                mut_seq = seq[: pos - 1] + aa + seq[pos:]
                variants.append((pos, orig, aa, mut_seq))

        all_mut_seqs = [v[3] for v in variants]
        embs = self._embed_sequences_batched(
            all_mut_seqs,
            batch_size=batch_size,
            show_progress=True,
            progress_desc="Scanning single mutations",
        )
        probs = self._predict_proba_from_embeddings(embs)

        rows: List[Dict[str, Any]] = []
        per_pos_max: Dict[int, float] = {p: 0.0 for p in positions}
        per_pos_loc: Dict[int, str] = {p: "none" for p in positions}
        for i, (pos, orig, aa, _seqm) in enumerate(variants):
            p = probs[i]
            mut_map = self._dict_from_probs(p)
            deltas = {name: float(mut_map[name] - base_map[name]) for name in self.label_names}
            loc = max(self.label_names, key=lambda n_: abs(deltas[n_]))
            delta = float(deltas[loc])
            absd = abs(delta)
            if absd > per_pos_max[pos]:
                per_pos_max[pos] = absd
                per_pos_loc[pos] = loc
            rows.append(
                {
                    "position": pos,
                    "original_aa": orig,
                    "mutant_aa": aa,
                    "max_delta": delta,
                    "most_affected_location": loc,
                    "direction": "gain" if delta >= 0 else "loss",
                }
            )

        rows.sort(key=lambda x: abs(float(x["max_delta"])), reverse=True)
        elapsed = time.perf_counter() - t0
        return {
            "sequence_length": n,
            "region_scanned": (rs, re),
            "total_variants_scored": len(rows),
            "time_seconds": float(elapsed),
            "top_mutations": rows[:top_k],
            "heatmap_data": {
                "positions": positions,
                "max_delta_per_position": [float(per_pos_max[p]) for p in positions],
                "most_affected_per_position": [per_pos_loc[p] for p in positions],
            },
        }

    def format_report(self, effect_result: Mapping[str, Any]) -> str:
        lines: List[str] = ["# Variant Effect Analysis", ""]
        muts = effect_result.get("mutations") or []
        mut_txt = ", ".join(f"{p}{o}>{m}" for p, o, m in muts) if muts else "-"
        lines.append(f"- Mutations: **{mut_txt}**")
        lines.append(f"- Mislocalization risk: **{effect_result.get('mislocalization_risk', 'none')}**")
        lines.append("")
        lines.append("## Prediction comparison")
        lines.append("")
        lines.append("| Location | Original | Mutant | Delta |")
        lines.append("|----------|----------|--------|-------|")
        p0 = effect_result.get("original_predictions", {})
        pm = effect_result.get("mutant_predictions", {})
        dd = effect_result.get("deltas", {})
        for k in sorted(p0.keys()):
            lines.append(f"| {k} | {float(p0.get(k, 0.0)):.4f} | {float(pm.get(k, 0.0)):.4f} | {float(dd.get(k, 0.0)):+.4f} |")
        lines.append("")
        lines.append("## Signal disruption analysis")
        lines.append("")
        lines.append(f"- Disrupted: {', '.join(effect_result.get('signals_disrupted', [])) or '-'}")
        lines.append(f"- Gained: {', '.join(effect_result.get('signals_gained', [])) or '-'}")
        lines.append("")
        lines.append("## Clinical summary")
        lines.append("")
        lines.append(effect_result.get("clinical_summary", ""))
        lines.append("")
        lines.append("## Interpretability findings")
        lines.append("")
        io_ = effect_result.get("interpretation_original", {})
        im_ = effect_result.get("interpretation_mutant", {})
        lines.append(f"- Original hot regions: {len(io_.get('hot_regions', []))}")
        lines.append(f"- Mutant hot regions: {len(im_.get('hot_regions', []))}")
        return "\n".join(lines)


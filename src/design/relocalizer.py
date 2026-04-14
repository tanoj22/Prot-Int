"""
Iterative protein relocalization design engine (classifier + ESM + mutations + interpretability).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.design.mutation_proposer import MutationProposer
from src.models.classifier import ProteinLocalizationClassifier, load_model
from src.models.interpretability import ProteinInterpreter
from src.utils.device import resolve_torch_device

logger = logging.getLogger(__name__)


def _hamming_mutations(original: str, variant: str) -> List[Tuple[int, str, str]]:
    """1-based positions where sequences differ."""
    o = original.upper()
    v = variant.upper()
    n = min(len(o), len(v))
    out: List[Tuple[int, str, str]] = []
    for i in range(n):
        if o[i] != v[i]:
            out.append((i + 1, o[i], v[i]))
    return out


def _mutations_preserve(
    mutations: Sequence[Tuple[int, str, str]],
    preserve_regions: Optional[Sequence[Tuple[int, int]]],
) -> bool:
    """Return True if variant is valid (no mutation touches a preserved region)."""
    if not preserve_regions:
        return True
    for pos, _a, _b in mutations:
        for s, e in preserve_regions:
            if int(s) <= int(pos) <= int(e):
                return False
    return True


def _attribution_to_dict(residue_scores: Sequence[Tuple[int, str, float]]) -> Dict[int, float]:
    return {int(p): abs(float(s)) for p, _a, s in residue_scores}


class ProteinRelocalizer:
    """
    Design engine: interpretability-guided mutations toward a target compartment.

    Embeddings reuse :class:`ProteinInterpreter`'s ESM encoder (no third full ESM copy).
    The masked LM in :class:`MutationProposer` remains a separate model (MLM head required).
    """

    def __init__(
        self,
        classifier_path: str | Path | None = None,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        *,
        mutation_esm_model_name: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        interpreter: Optional[ProteinInterpreter] = None,
        proposer: Optional[MutationProposer] = None,
    ) -> None:
        """
        Pass ``interpreter`` and/or ``proposer`` to reuse instances across runs (saves load time).

        ``mutation_esm_model_name`` selects the MaskedLM for mutations/PLL; defaults to
        ``esm_model_name`` when not using a separate smaller MLM.
        """
        if interpreter is not None:
            self.device = resolve_torch_device(device) if device is not None else interpreter.device
        else:
            self.device = resolve_torch_device(device)

        if interpreter is not None:
            self.classifier_path = interpreter.classifier_path
            if classifier_path is not None:
                cp = Path(classifier_path).expanduser().resolve()
                if cp != self.classifier_path:
                    logger.warning(
                        "classifier_path %s does not match interpreter (%s); using interpreter path",
                        cp,
                        self.classifier_path,
                    )
            self.classifier = interpreter.classifier
            self.label_names = list(interpreter.label_names)
            self.interpreter = interpreter
            self.esm_model_name = getattr(interpreter, "esm_model_name", esm_model_name)
            logger.info("Using injected ProteinInterpreter (encoder=%s)", self.esm_model_name)
        else:
            if classifier_path is None:
                raise ValueError("classifier_path is required when interpreter is not provided")
            self.classifier_path = Path(classifier_path).expanduser().resolve()
            if not self.classifier_path.is_file():
                raise FileNotFoundError(f"Missing classifier: {self.classifier_path}")

            ckpt = torch.load(self.classifier_path, map_location="cpu")
            if not isinstance(ckpt, dict):
                raise ValueError("Unsupported checkpoint format.")
            embedding_dim = int(ckpt.get("embedding_dim", 1280))

            logger.info("Loading classifier from %s", self.classifier_path)
            self.classifier = load_model(
                self.classifier_path,
                embedding_dim=embedding_dim,
                num_labels=None,
                device=self.device,
            )
            self.label_names = list(self.classifier.label_names)

            logger.info("Loading ProteinInterpreter (%s)", esm_model_name)
            self.interpreter = ProteinInterpreter(
                classifier_path=self.classifier_path,
                esm_model_name=esm_model_name,
                device=self.device,
            )
            self.esm_model_name = esm_model_name

        mlm_name = mutation_esm_model_name if mutation_esm_model_name is not None else esm_model_name
        self.mutation_esm_model_name = mlm_name

        if proposer is not None:
            self.proposer = proposer
            logger.info("Using injected MutationProposer (MLM=%s)", getattr(proposer, "esm_model_name", mlm_name))
        else:
            logger.info("Loading MutationProposer (MLM=%s)", mlm_name)
            self.proposer = MutationProposer(esm_model_name=mlm_name, device=self.device)

    @classmethod
    def from_lightweight(
        cls,
        classifier_path: str | Path,
        device: Optional[str | torch.device] = None,
        *,
        interpreter_model: str = "facebook/esm2_t33_650M_UR50D",
        mlm_model: str = "facebook/esm2_t12_35M_UR50D",
    ) -> ProteinRelocalizer:
        """
        Two ESM stacks total: t33 encoder (interpreter + embeddings) + t12 MaskedLM (mutations/PLL).

        Fits better in ~8GB VRAM than loading 650M three times.
        """
        dev = resolve_torch_device(device)
        interpreter = ProteinInterpreter(
            classifier_path=classifier_path,
            esm_model_name=interpreter_model,
            device=dev,
        )
        proposer = MutationProposer(esm_model_name=mlm_model, device=dev)
        return cls(
            classifier_path=classifier_path,
            esm_model_name=interpreter_model,
            mutation_esm_model_name=mlm_model,
            device=dev,
            interpreter=interpreter,
            proposer=proposer,
        )

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Mean-pooled ESM embedding via the shared interpreter encoder (matches training-style pooling)."""
        return self.interpreter.mean_pool_embedding(sequence)

    def score_variant(self, sequence: str) -> Dict[str, Any]:
        """Classifier multilabel probs + ESM pseudo-log-likelihood plausibility."""
        seq = sequence.upper().strip()
        emb = torch.from_numpy(self.embed_sequence(seq)).unsqueeze(0).to(self.device)
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(emb)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
        loc_probs = {self.label_names[i]: float(probs[i]) for i in range(len(self.label_names))}
        plaus = float(self.proposer.score_sequence_plausibility(seq, subsample_step=5))
        return {"localization_probs": loc_probs, "plausibility_score": plaus}

    def _composite_score(
        self,
        scored: Mapping[str, Any],
        original_target_prob: float,
        original_source_prob: float,
        source_location: str,
        target_location: str,
        pll_term: float = 0.0,
    ) -> float:
        probs = scored["localization_probs"]
        tgt = float(probs.get(target_location, 0.0))
        src = float(probs.get(source_location, 0.0))
        return (
            (tgt - original_target_prob) * 2.0
            + (original_source_prob - src) * 1.0
            + float(pll_term) * 0.1
        )

    def relocalize(
        self,
        sequence: str,
        source_location: str,
        target_location: str,
        n_iterations: int = 10,
        candidates_per_iteration: int = 20,
        max_total_mutations: Optional[int] = None,
        preserve_regions: Optional[Sequence[Tuple[int, int]]] = None,
        n_positions_smart: int = 6,
        top_k_mutations: int = 3,
    ) -> Dict[str, Any]:
        """
        Iterative relocalization: IG-guided mutations, filtered, scored, ranked.

        ``preserve_regions`` uses **1-based inclusive** (start, end) residue indices.
        """
        t0 = time.perf_counter()
        seq0 = sequence.upper().strip()
        if source_location not in self.label_names or target_location not in self.label_names:
            raise ValueError(f"Unknown location(s). Use one of: {self.label_names}")

        n = len(seq0)
        if max_total_mutations is None:
            max_total_mutations = max(1, int(np.ceil(0.1 * n)))

        original_scores = self.score_variant(seq0)
        orig_src = float(original_scores["localization_probs"].get(source_location, 0.0))
        orig_tgt = float(original_scores["localization_probs"].get(target_location, 0.0))

        if orig_src <= 0.3:
            logger.warning(
                "This protein may not actually be in %s (source probability=%.3f <= 0.3)",
                source_location,
                orig_src,
            )

        seed_sequences: List[str] = [seq0]
        all_seen: Dict[str, Dict[str, Any]] = {}
        trajectory: List[Dict[str, Any]] = []
        total_evaluated = 0

        def record(seq: str, scored: Dict[str, Any], comp: float) -> None:
            muts = _hamming_mutations(seq0, seq)
            if seq not in all_seen or comp > all_seen[seq]["composite_score"]:
                all_seen[seq] = {
                    "sequence": seq,
                    "mutations": muts,
                    "num_mutations": len(muts),
                    "localization_probs": dict(scored["localization_probs"]),
                    "plausibility_score": float(scored["plausibility_score"]),
                    "composite_score": float(comp),
                }

        # seed original
        record(
            seq0,
            original_scores,
            self._composite_score(
                original_scores,
                orig_tgt,
                orig_src,
                source_location,
                target_location,
                pll_term=0.0,
            ),
        )

        pbar = tqdm(range(1, n_iterations + 1), desc="Relocalize", unit="iter")
        for it in pbar:
            pooled_valid: List[Dict[str, Any]] = []
            pooled_seen: set[str] = set()
            seed_errors: List[str] = []

            for base in list(seed_sequences):
                try:
                    ig_src = self.interpreter.get_integrated_gradients(base, source_location)
                    att_src = _attribution_to_dict(ig_src["residue_scores"])
                except Exception as ex:
                    logger.exception("Iteration %d seed: integrated gradients (source) failed: %s", it, ex)
                    seed_errors.append(f"{base[:12]}... source_ig: {ex}")
                    continue

                try:
                    ig_tgt = self.interpreter.get_integrated_gradients(base, target_location)
                    att_tgt = _attribution_to_dict(ig_tgt["residue_scores"])
                except Exception as ex:
                    logger.warning("Iteration %d seed: integrated gradients (target) failed: %s; continuing without secondary", it, ex)
                    att_tgt = None

                try:
                    proposals = self.proposer.propose_smart_mutations(
                        base,
                        attribution_scores=att_src,
                        current_location=source_location,
                        target_location=target_location,
                        n_positions=n_positions_smart,
                        top_k=top_k_mutations,
                        attribution_secondary=att_tgt,
                    )
                except Exception as ex:
                    logger.exception("Iteration %d seed: propose_smart_mutations failed: %s", it, ex)
                    seed_errors.append(f"{base[:12]}... propose: {ex}")
                    continue

                if not proposals:
                    continue

                try:
                    raw_variants = self.proposer.generate_variants(
                        base,
                        mutation_proposals=proposals,
                        max_simultaneous_mutations=3,
                        max_variants=max(80, candidates_per_iteration * 6),
                    )
                except Exception as ex:
                    logger.exception("Iteration %d seed: generate_variants failed: %s", it, ex)
                    seed_errors.append(f"{base[:12]}... gen: {ex}")
                    continue

                for v in raw_variants:
                    sseq = str(v["sequence"])
                    if sseq in pooled_seen:
                        continue
                    muts = list(v["mutations"])
                    total_from_orig = len(_hamming_mutations(seq0, sseq))
                    if total_from_orig > max_total_mutations:
                        continue
                    if not _mutations_preserve(muts, preserve_regions):
                        continue
                    pooled_seen.add(sseq)
                    pooled_valid.append(v)

            if not pooled_valid:
                best_now = max(all_seen.values(), key=lambda x: float(x["composite_score"]))
                trajectory.append(
                    {
                        "iteration": it,
                        "best_target_prob": float(best_now["localization_probs"].get(target_location, 0.0)),
                        "best_source_prob": float(best_now["localization_probs"].get(source_location, 0.0)),
                        "num_mutations": len(best_now["mutations"]),
                        "num_candidates_evaluated": 0,
                        "num_seeds": len(seed_sequences),
                        "error": "; ".join(seed_errors[:3]) if seed_errors else "no valid variants",
                    }
                )
                continue

            scored_batch_raw: List[Tuple[str, Dict[str, Any], float]] = []
            for v in pooled_valid:
                if len(scored_batch_raw) >= candidates_per_iteration * 6:
                    break
                try:
                    s = self.score_variant(v["sequence"])
                    total_evaluated += 1
                    pll = float(s["plausibility_score"])
                    scored_batch_raw.append((v["sequence"], s, pll))
                except Exception as ex:
                    logger.warning("Iteration %d: score_variant failed for a variant: %s", it, ex)

            if not scored_batch_raw:
                best_now = max(all_seen.values(), key=lambda x: float(x["composite_score"]))
                trajectory.append(
                    {
                        "iteration": it,
                        "best_target_prob": float(best_now["localization_probs"].get(target_location, 0.0)),
                        "best_source_prob": float(best_now["localization_probs"].get(source_location, 0.0)),
                        "num_mutations": len(best_now["mutations"]),
                        "num_candidates_evaluated": 0,
                        "num_seeds": len(seed_sequences),
                        "error": "all variant scoring failed",
                    }
                )
                continue

            # FIX 1: normalize PLL within iteration batch to keep plausibility as tie-breaker.
            pll_vals = [x[2] for x in scored_batch_raw]
            pll_min = min(pll_vals)
            pll_max = max(pll_vals)
            pll_span = (pll_max - pll_min) + 1e-8

            scored_batch: List[Tuple[float, str, Dict[str, Any]]] = []
            for sseq, sdict, pll in scored_batch_raw:
                pll_norm = (pll - pll_min) / pll_span
                comp = self._composite_score(
                    sdict,
                    orig_tgt,
                    orig_src,
                    source_location,
                    target_location,
                    pll_term=pll_norm,
                )
                scored_batch.append((comp, sseq, sdict))

            scored_batch.sort(key=lambda x: x[0], reverse=True)
            top_iter = scored_batch[: candidates_per_iteration]

            for comp, sseq, sdict in top_iter:
                record(sseq, sdict, comp)

            if top_iter:
                # FIX 3: keep multiple seeds (top 3 unique) to reduce local optima trapping.
                next_seeds: List[str] = []
                seen_seed: set[str] = set()
                for _comp, sseq, _sdict in top_iter:
                    if sseq in seen_seed:
                        continue
                    next_seeds.append(sseq)
                    seen_seed.add(sseq)
                    if len(next_seeds) >= 3:
                        break
                if not next_seeds:
                    next_seeds = list(seed_sequences)
                seed_sequences = next_seeds
                best_comp, next_seq, _sdict = top_iter[0]
                pbar.set_postfix(
                    comp=f"{best_comp:.3f}",
                    muts=len(_hamming_mutations(seq0, next_seq)),
                )
                trajectory.append(
                    {
                        "iteration": it,
                        "best_target_prob": float(top_iter[0][2]["localization_probs"].get(target_location, 0.0)),
                        "best_source_prob": float(top_iter[0][2]["localization_probs"].get(source_location, 0.0)),
                        "num_mutations": len(_hamming_mutations(seq0, next_seq)),
                        "num_candidates_evaluated": len(scored_batch),
                        "num_seeds": len(seed_sequences),
                    }
                )
            else:
                best_now = max(all_seen.values(), key=lambda x: float(x["composite_score"]))
                trajectory.append(
                    {
                        "iteration": it,
                        "best_target_prob": float(best_now["localization_probs"].get(target_location, 0.0)),
                        "best_source_prob": float(best_now["localization_probs"].get(source_location, 0.0)),
                        "num_mutations": len(best_now["mutations"]),
                        "num_candidates_evaluated": 0,
                        "num_seeds": len(seed_sequences),
                    }
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranked = sorted(all_seen.values(), key=lambda x: x["composite_score"], reverse=True)
        top5 = []
        for rank, item in enumerate(ranked[:5], start=1):
            top5.append({
                "rank": rank,
                "sequence": item["sequence"],
                "mutations": item["mutations"],
                "num_mutations": item["num_mutations"],
                "localization_probs": item["localization_probs"],
                "plausibility_score": item["plausibility_score"],
                "composite_score": item["composite_score"],
            })

        elapsed = time.perf_counter() - t0
        return {
            "original_sequence": seq0,
            "original_scores": original_scores,
            "source_location": source_location,
            "target_location": target_location,
            "top_candidates": top5,
            "optimization_trajectory": trajectory,
            "total_variants_evaluated": total_evaluated,
            "total_time_seconds": float(elapsed),
        }

    def get_summary(self, results: Mapping[str, Any]) -> str:
        """Human-readable summary of a relocalization run."""
        lines = []
        src = results["source_location"]
        tgt = results["target_location"]
        orig = results["original_sequence"]
        os_ = results["original_scores"]
        o_probs = os_["localization_probs"]
        lines.append(f"Relocalization design: {src} -> {tgt}")
        lines.append(f"Original length: {len(orig)}")
        lines.append(f"Original P({src})={o_probs.get(src, 0):.4f}  P({tgt})={o_probs.get(tgt, 0):.4f}")
        lines.append(f"Variants scored: {results.get('total_variants_evaluated', 0)}  time: {results.get('total_time_seconds', 0):.2f}s")

        tops = results.get("top_candidates") or []
        if not tops:
            lines.append("No candidates recorded.")
            return "\n".join(lines)

        best = tops[0]
        bp = best["localization_probs"]
        lines.append("")
        lines.append(f"Best candidate (rank 1): composite={best['composite_score']:.4f}")
        lines.append(f"  P({src})={bp.get(src, 0):.4f}  P({tgt})={bp.get(tgt, 0):.4f}")
        lines.append(f"  Plausibility (PLL term): {best['plausibility_score']:.4f}")
        muts = best.get("mutations") or []
        if muts:
            mut_str = ", ".join(f"{p}{o}>{n}" for p, o, n in muts[:30])
            if len(muts) > 30:
                mut_str += ", ..."
            lines.append(f"  Mutations vs original ({len(muts)}): {mut_str}")
        else:
            lines.append("  No mutations (same as original).")

        best_seq = best["sequence"]
        try:
            ig = self.interpreter.get_integrated_gradients(best_seq, tgt)
            hot = self.interpreter.identify_hot_regions(ig["residue_scores"], window_size=10, top_percentile=90)
            sig = self.interpreter.validate_against_known_signals(best_seq, hot)
            lines.append("")
            lines.append("Known-signal check (best sequence):")
            for name, payload in sig.items():
                if payload.get("detected"):
                    lines.append(
                        f"  - {name}: region={payload.get('region')}, "
                        f"overlap_with_attribution={payload.get('overlap_with_attribution')}"
                    )
        except Exception as ex:
            lines.append(f"(Could not run signal validation: {ex})")

        return "\n".join(lines)

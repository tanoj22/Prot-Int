"""
ESM-2 masked language model for proposing biologically plausible mutations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerBase

from src.utils.device import resolve_torch_device

logger = logging.getLogger(__name__)

# Standard amino acids (single-letter)
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# Batch size for masked forward passes (tune for GPU memory)
DEFAULT_MLM_BATCH = 8

# Smaller MLM for development / limited VRAM (mutation proposals + PLL)
DEFAULT_SMALL_MLM = "facebook/esm2_t12_35M_UR50D"


def _normalize_attribution(
    scores: Union[Mapping[int, float], Sequence[Tuple[int, float]]],
) -> Dict[int, float]:
    if isinstance(scores, Mapping):
        return {int(k): float(v) for k, v in scores.items()}
    return {int(p): float(s) for p, s in scores}


class MutationProposer:
    """
    Propose mutations using ESM-2 as a masked language model.

    Positions are **1-based** (first residue = 1), consistent with biology conventions.
    """

    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[str | torch.device] = None,
        mlm_batch_size: int = DEFAULT_MLM_BATCH,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.mlm_batch_size = max(1, int(mlm_batch_size))
        self.esm_model_name = esm_model_name
        logger.info("Loading tokenizer: %s", esm_model_name)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(esm_model_name)
        logger.info("Loading MaskedLM model: %s", esm_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            esm_model_name,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device).eval()

        self._mask_token = self.tokenizer.mask_token
        self._mask_id = int(self.tokenizer.mask_token_id)
        self._aa_token_ids: List[int] = []
        self._id_to_aa: Dict[int, str] = {}
        for aa in sorted(AA_ALPHABET):
            tid = self.tokenizer.convert_tokens_to_ids(aa)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                self._aa_token_ids.append(tid)
                self._id_to_aa[tid] = aa
        if not self._aa_token_ids:
            # Fallback: single-token AAs may be stored differently
            for tid in range(len(self.tokenizer)):
                t = self.tokenizer.convert_ids_to_tokens(tid)
                if isinstance(t, str) and len(t) == 1 and t in AA_ALPHABET:
                    self._aa_token_ids.append(tid)
                    self._id_to_aa[tid] = t
        self._aa_tensor = torch.tensor(self._aa_token_ids, dtype=torch.long, device=self.device)

    def _clear_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sequence_with_mask(self, sequence: str, pos_1based: int) -> str:
        if not 1 <= pos_1based <= len(sequence):
            raise ValueError(f"position {pos_1based} out of range for sequence length {len(sequence)}")
        ch = sequence[pos_1based - 1]
        if ch.upper() not in AA_ALPHABET:
            raise ValueError(f"Non-standard residue at position {pos_1based}: {ch!r}")
        seq_list = list(sequence.upper())
        seq_list[pos_1based - 1] = self._mask_token or "<mask>"
        return "".join(seq_list)

    def _find_mask_index(self, input_ids: torch.Tensor, row: int) -> int:
        """Token index of mask in row ``row``."""
        row_ids = input_ids[row]
        matches = (row_ids == self._mask_id).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise RuntimeError("Mask token not found in tokenized sequence.")
        return int(matches[0, 0].item())

    def propose_single_mutations(
        self,
        sequence: str,
        positions: Sequence[int],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        For each 1-based position, return top-k alternative amino acids (excluding wild-type)
        with masked LM probabilities at the masked site.
        """
        seq = sequence.upper().strip()
        if not seq:
            raise ValueError("Empty sequence")
        top_k = max(1, int(top_k))
        results: List[Dict[str, Any]] = []

        pos_list = list(positions)
        logger.info("propose_single_mutations: %d positions, batch_size=%d", len(pos_list), self.mlm_batch_size)

        for start in range(0, len(pos_list), self.mlm_batch_size):
            batch_pos = pos_list[start : start + self.mlm_batch_size]
            masked_texts = [self._sequence_with_mask(seq, p) for p in batch_pos]
            enc = self.tokenizer(
                masked_texts,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits  # [batch, seq, vocab]

            for bi, pos in enumerate(batch_pos):
                mi = self._find_mask_index(enc["input_ids"], bi)
                pos_logits = logits[bi, mi]
                # Restrict to standard amino acids
                aa_logits = pos_logits[self._aa_tensor]
                probs = F.softmax(aa_logits, dim=-1)
                orig = seq[pos - 1]
                orig_id = self.tokenizer.convert_tokens_to_ids(orig)
                mut_list: List[Dict[str, float]] = []
                scores_cpu = probs.detach().cpu()
                ids_cpu = self._aa_tensor.detach().cpu()
                pairs = [(float(scores_cpu[j]), int(ids_cpu[j])) for j in range(len(ids_cpu))]
                pairs.sort(key=lambda x: x[0], reverse=True)
                for prob, tid in pairs:
                    aa = self._id_to_aa.get(tid)
                    if aa is None or aa == orig:
                        continue
                    mut_list.append({"residue": aa, "score": prob})
                    if len(mut_list) >= top_k:
                        break

                results.append(
                    {
                        "position": int(pos),
                        "original": orig,
                        "mutations": mut_list,
                    }
                )
            self._clear_cuda()

        logger.info("propose_single_mutations: produced %d site proposals", len(results))
        return results

    def _target_region_bonus(self, position: int, target_location: str) -> float:
        """Heuristic 0..1 bonus for positions relevant to acquiring ``target_location``."""
        t = target_location.strip()
        # N-terminal transit / signal-like regions
        if t == "Mitochondrion":
            return 1.0 if 1 <= position <= 70 else 0.0
        if t in ("Extracellular", "Membrane", "Cell membrane", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus"):
            return 1.0 if 1 <= position <= 35 else 0.15
        if t == "Peroxisome":
            return 1.0 if 1 <= position <= 40 else 0.1
        if t == "Nucleus":
            # NLS can be internal; slight bias to charged clusters not modeled here
            return 0.25
        if t in ("Cytoplasm", "Plastid"):
            return 0.2
        return 0.2

    def _get_target_bias(self, target_location: str, seq_length: int) -> Tuple[List[int], Dict[str, float]]:
        """
        Target-aware mutation bias metadata.

        Returns:
          - preferred_positions: 1-based positions to prioritize
          - residue_bonuses: multiplicative factors for candidate residues
        """
        t = target_location.strip()
        n = int(seq_length)
        preferred: List[int] = []
        residue_bonuses: Dict[str, float] = {}

        if t == "Mitochondrion":
            preferred = list(range(1, min(40, n) + 1))
            residue_bonuses = {aa: 2.0 for aa in ("R", "K", "S", "L", "A")}
        elif t == "Nucleus":
            # NLS-like bias: K/R-rich windows anywhere in sequence.
            residue_bonuses = {"K": 2.0, "R": 2.0}
            preferred_set: Set[int] = set()
            for i in range(1, max(1, n - 4 + 2)):  # windows of 4 residues
                win = [i + j for j in range(4) if i + j <= n]
                if len(win) < 4:
                    continue
                # Prefer windows where at least 3 sites can become K/R.
                mutable = sum(1 for p in win if p >= 1 and p <= n)
                if mutable >= 3:
                    preferred_set.update(win)
            preferred = sorted(preferred_set)
        elif t in ("Extracellular", "Endoplasmic reticulum"):
            preferred = list(range(1, min(30, n) + 1))
            residue_bonuses = {aa: 1.6 for aa in ("A", "V", "I", "L", "M", "F", "W")}
        elif t in ("Membrane", "Cell membrane"):
            # Bias into one contiguous putative TM region.
            win = min(22, n)
            if win >= 18:
                start = max(1, (n - win) // 2 + 1)
                preferred = list(range(start, start + win))
            else:
                preferred = list(range(1, n + 1))
            residue_bonuses = {aa: 1.8 for aa in ("A", "V", "I", "L", "M", "F", "W")}
        else:
            preferred = []
            residue_bonuses = {}

        return preferred, residue_bonuses

    def propose_smart_mutations(
        self,
        sequence: str,
        attribution_scores: Union[Mapping[int, float], Sequence[Tuple[int, float]]],
        current_location: str,
        target_location: str,
        n_positions: int = 5,
        top_k: int = 3,
        attribution_secondary: Optional[Union[Mapping[int, float], Sequence[Tuple[int, float]]]] = None,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select positions using a weighted mix of:
        (a) high attribution for ``current_location`` (disrupt current signal),
        (b) bonus in organelle-relevant regions for ``target_location``,
        (c) penalty when both primary and secondary attributions are high (conserved / dual-use sites).

        ``attribution_secondary`` is optional (e.g. IG scores for ``target_location``). If omitted,
        conservation penalty uses only primary scores at a lower weight.
        """
        seq = sequence.upper().strip()
        n = len(seq)
        n_positions = max(1, min(int(n_positions), n))
        top_k = max(1, int(top_k))

        w_attr, w_region, w_cons = weights if weights is not None else (0.55, 0.30, 0.15)
        att = _normalize_attribution(attribution_scores)
        att2 = _normalize_attribution(attribution_secondary) if attribution_secondary is not None else {}
        preferred_positions, residue_bonuses = self._get_target_bias(target_location, n)
        preferred_set = set(preferred_positions)

        # Normalize primary attribution to [0, 1]
        vals = [float(att.get(i, 0.0)) for i in range(1, n + 1)]
        vmin, vmax = min(vals), max(vals)
        span = max(vmax - vmin, 1e-8)

        def norm_primary(p: int) -> float:
            return (float(att.get(p, 0.0)) - vmin) / span

        def conservation_penalty(p: int) -> float:
            a1 = float(att.get(p, 0.0))
            if att2:
                a2 = float(att2.get(p, 0.0))
                return min(a1, a2) / max(max(a1, a2), 1e-8)
            return 0.0

        scored: List[Tuple[float, int, str]] = []
        for p in range(1, n + 1):
            if seq[p - 1] not in AA_ALPHABET:
                continue
            attr_n = norm_primary(p)
            reg = self._target_region_bonus(p, target_location)
            pref = 1.0 if p in preferred_set else 0.0
            cons = conservation_penalty(p)
            # Higher score = better candidate to mutate toward target
            score = (w_attr * attr_n) + (w_region * reg) + (0.25 * pref) - (w_cons * cons)
            reason_parts = [
                f"attr_norm={attr_n:.3f} for current='{current_location}'",
                f"target_region_bonus={reg:.3f} for '{target_location}'",
                f"preferred_pos={pref:.1f}",
            ]
            if att2:
                reason_parts.append(f"conservation_penalty={cons:.3f} (min primary/secondary)")
            else:
                reason_parts.append("no secondary attribution; conservation term ~0")
            scored.append((score, p, "; ".join(reason_parts)))

        scored.sort(key=lambda x: x[0], reverse=True)
        if target_location.strip() == "Mitochondrion" and preferred_positions:
            min_pref = max(1, int(np.ceil(0.5 * n_positions)))
            pref_scored = [x for x in scored if x[1] in preferred_set]
            nonpref_scored = [x for x in scored if x[1] not in preferred_set]
            chosen = pref_scored[:min_pref] + nonpref_scored[: max(0, n_positions - min_pref)]
            if len(chosen) < n_positions:
                used = {x[1] for x in chosen}
                chosen.extend([x for x in scored if x[1] not in used][: n_positions - len(chosen)])
        else:
            chosen = scored[:n_positions]

        positions_only = [p for _, p, _ in chosen]

        logger.info(
            "propose_smart_mutations: selected positions %s (current=%s -> target=%s)",
            positions_only,
            current_location,
            target_location,
        )

        proposals = self.propose_single_mutations(seq, positions_only, top_k=top_k)
        pos_to_reason = {p: r for _, p, r in chosen}
        pos_to_sel_score = {pp: s for s, pp, _ in chosen}
        out: List[Dict[str, Any]] = []
        for prop in proposals:
            p = int(prop["position"])
            boosted: List[Dict[str, float]] = []
            for m in prop.get("mutations", []):
                aa = str(m["residue"]).upper()
                base_sc = float(m.get("score", 0.0))
                mult = float(residue_bonuses.get(aa, 1.0))
                boosted.append({"residue": aa, "score": base_sc * mult})
            boosted.sort(key=lambda x: float(x["score"]), reverse=True)
            boosted = boosted[:top_k]
            out.append(
                {
                    **prop,
                    "mutations": boosted,
                    "reasoning": pos_to_reason.get(p, ""),
                    "selection_score": float(pos_to_sel_score.get(p, 0.0)),
                }
            )
        return out

    def generate_variants(
        self,
        sequence: str,
        mutation_proposals: Sequence[Mapping[str, Any]],
        max_simultaneous_mutations: int = 3,
        max_variants: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Build single, double, and (optionally) triple mutants from proposed substitutions.
        ``mutation_proposals`` entries should include ``position``, ``original``, ``mutations`` (list of dicts with residue, score).
        """
        seq = sequence.upper().strip()
        max_simultaneous_mutations = max(1, min(int(max_simultaneous_mutations), 3))
        max_variants = max(1, int(max_variants))

        # Flatten: (position, orig, new_aa, score)
        singles: List[Tuple[int, str, str, float]] = []
        for block in mutation_proposals:
            pos = int(block["position"])
            orig = str(block["original"]).upper()
            for m in block.get("mutations", []):
                aa = str(m["residue"]).upper()
                sc = float(m.get("score", 0.0))
                if aa != orig:
                    singles.append((pos, orig, aa, sc))

        singles.sort(key=lambda x: x[3], reverse=True)
        logger.info("generate_variants: %d single-mutation options before combinatorics", len(singles))

        variants: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        def add_variant(muts: List[Tuple[int, str, str]], combined: float) -> None:
            if len(variants) >= max_variants:
                return
            new_seq = list(seq)
            for pos, _o, aa in muts:
                new_seq[pos - 1] = aa
            s = "".join(new_seq)
            if s in seen:
                return
            seen.add(s)
            variants.append(
                {
                    "sequence": s,
                    "mutations": muts,
                    "num_mutations": len(muts),
                    "combined_score": combined,
                }
            )

        # 1) All single mutants (best effort by score order until cap)
        for pos, orig, aa, sc in singles:
            if len(variants) >= max_variants:
                break
            add_variant([(pos, orig, aa)], sc)

        if max_simultaneous_mutations < 2 or len(variants) >= max_variants:
            logger.info("generate_variants: returning %d variants (singles only or cap)", len(variants))
            return variants[:max_variants]

        # 2) Double mutants: distinct positions, rank by product of marginal probabilities
        pair_scores: List[Tuple[float, int, int]] = []
        for i in range(len(singles)):
            for j in range(i + 1, len(singles)):
                p1, _, _, s1 = singles[i]
                p2, _, _, s2 = singles[j]
                if p1 == p2:
                    continue
                pair_scores.append((s1 * s2, i, j))
        pair_scores.sort(key=lambda x: x[0], reverse=True)
        for _comb, i, j in pair_scores:
            if len(variants) >= max_variants:
                break
            p1, o1, a1, s1 = singles[i]
            p2, o2, a2, s2 = singles[j]
            add_variant([(p1, o1, a1), (p2, o2, a2)], s1 * s2)

        if max_simultaneous_mutations < 3 or len(variants) >= max_variants:
            logger.info("generate_variants: returning %d variants", len(variants))
            return variants[:max_variants]

        # 3) Triple mutants: top-ranked non-overlapping triples (search bounded for speed)
        triple_scores: List[Tuple[float, int, int, int]] = []
        bound = min(len(singles), 20)
        for i in range(bound):
            for j in range(i + 1, bound):
                for k in range(j + 1, bound):
                    p1, _, _, s1 = singles[i]
                    p2, _, _, s2 = singles[j]
                    p3, _, _, s3 = singles[k]
                    if len({p1, p2, p3}) < 3:
                        continue
                    triple_scores.append((s1 * s2 * s3, i, j, k))
        triple_scores.sort(key=lambda x: x[0], reverse=True)
        for sc, i, j, k in triple_scores:
            if len(variants) >= max_variants:
                break
            muts = [
                (singles[i][0], singles[i][1], singles[i][2]),
                (singles[j][0], singles[j][1], singles[j][2]),
                (singles[k][0], singles[k][1], singles[k][2]),
            ]
            add_variant(muts, sc)

        logger.info("generate_variants: returning %d variants", len(variants))
        return variants[:max_variants]

    def score_sequence_plausibility(self, sequence: str, subsample_step: int = 5) -> float:
        """
        Masked pseudo-log-likelihood: sum of log P(true_aa | masked) at subsampled positions
        (every ``subsample_step`` residues, 1-based). Higher = more plausible under the LM.
        """
        seq = sequence.upper().strip()
        if not seq:
            raise ValueError("Empty sequence")
        step = max(1, int(subsample_step))
        positions = list(range(1, len(seq) + 1, step))
        logger.info(
            "score_sequence_plausibility: length=%d, subsample_step=%d -> %d positions",
            len(seq),
            step,
            len(positions),
        )

        total_ll = 0.0
        aa_ids_set = set(self._aa_token_ids)

        for start in range(0, len(positions), self.mlm_batch_size):
            batch_pos = positions[start : start + self.mlm_batch_size]
            texts = [self._sequence_with_mask(seq, p) for p in batch_pos]
            enc = self.tokenizer(texts, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            true_ids: List[int] = []
            for p in batch_pos:
                tid = self.tokenizer.convert_tokens_to_ids(seq[p - 1])
                true_ids.append(int(tid))

            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits

            for bi, _p in enumerate(batch_pos):
                mi = self._find_mask_index(enc["input_ids"], bi)
                pos_logits = logits[bi, mi]
                log_probs = F.log_softmax(pos_logits, dim=-1)
                tid = true_ids[bi]
                if tid not in aa_ids_set:
                    continue
                total_ll += float(log_probs[tid].item())
            self._clear_cuda()

        logger.info("score_sequence_plausibility: PLL (subsampled) = %.4f", total_ll)
        return total_ll

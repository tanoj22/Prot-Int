"""
Supporting utilities for protein design and relocalization reporting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

STANDARD_AA: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")

POSITIVE_AA: Set[str] = set("KR")
HYDROPHOBIC_SP: Set[str] = set("AVILMFW")  # signal peptide h-region (per spec)
HYDROPHOBIC_TM: Set[str] = set("AVILMFWY")  # TM / GPI hydrophobic
MTP_ENRICHED: Set[str] = set("RSLA")
CLEAVAGE_SMALL: Set[str] = set("AGS")


def align_and_diff(original_seq: str, designed_seq: str) -> Dict[str, Any]:
    """
    Compare two sequences of equal length (substitutions only).

    Returns sequence identity (%), mutation list (1-based positions), diff string
    (``.`` = unchanged, letter = new residue at mutations).
    """
    o = original_seq.upper().strip()
    d = designed_seq.upper().strip()
    if len(o) != len(d):
        raise ValueError(
            f"Sequences must be the same length for substitution-only diff (got {len(o)} vs {len(d)})."
        )
    if not o:
        return {
            "sequence_identity": 100.0,
            "mutations": [],
            "num_mutations": 0,
            "diff_string": "",
        }

    mutations: List[Tuple[int, str, str]] = []
    diff_chars: List[str] = []
    matches = 0
    for i, (a, b) in enumerate(zip(o, d)):
        if a == b:
            matches += 1
            diff_chars.append(".")
        else:
            mutations.append((i + 1, a, b))
            diff_chars.append(b)

    identity = 100.0 * matches / len(o)
    return {
        "sequence_identity": float(identity),
        "mutations": mutations,
        "num_mutations": len(mutations),
        "diff_string": "".join(diff_chars),
    }


def _sp_confidence(
    n_pos: int, h_len: int, cleavage_ok: bool, in_first_30: bool,
) -> str:
    if n_pos >= 1 and 7 <= h_len <= 15 and cleavage_ok and in_first_30:
        return "strong"
    if (n_pos >= 1 and h_len >= 7) or cleavage_ok:
        return "weak"
    return "none"


def check_localization_signals(sequence: str) -> Dict[str, Dict[str, Any]]:
    """
    Heuristic scan for localization-related sequence motifs.

    Returns per-signal dicts with ``detected``, ``region`` (1-based inclusive, or None),
    and ``confidence`` in ``strong`` / ``weak`` / ``none``.
    """
    seq = sequence.upper().strip()
    n = len(seq)
    out: Dict[str, Dict[str, Any]] = {}

    # --- signal_peptide (first ~30 aa): n-region KR, h-region hydrophobic, cleavage ~20–30 ---
    sp_detected = False
    sp_region: Optional[Tuple[int, int]] = None
    sp_conf = "none"
    head = seq[: min(30, n)]
    hn = len(head)
    for j in range(0, min(5, hn)):
        for plen in range(1, 6):
            if j + plen > hn:
                break
            block = head[j : j + plen]
            if not block or not all(a in POSITIVE_AA for a in block):
                continue
            rest = j + plen
            for hlen in range(7, 16):
                if rest + hlen > hn:
                    break
                hblock = head[rest : rest + hlen]
                if len(hblock) < 7 or not all(a in HYDROPHOBIC_SP for a in hblock):
                    continue
                # Cleavage site "near" positions 20–30 (1-based): small residue in window 20–30
                tail_start = rest + hlen  # 0-based index after h-region
                cleavage_window = head[19:30] if hn >= 20 else ""
                cleavage_ok = any(a in CLEAVAGE_SMALL for a in cleavage_window) if cleavage_window else False
                if not cleavage_ok and hn >= 22:
                    # fallback: polar/small at end of first 30
                    cleavage_ok = head[min(tail_start, hn - 1)] in CLEAVAGE_SMALL | {"T", "P"}

                sp_detected = True
                end_1 = min(30, rest + hlen + 3)
                sp_region = (1, end_1)
                sp_conf = _sp_confidence(plen, hlen, cleavage_ok, end_1 <= 30)
                break
            if sp_detected:
                break
        if sp_detected:
            break

    if not sp_detected and hn >= 10:
        # weak: hydrophobic stretch only in first 30
        for hlen in range(7, 16):
            for i in range(0, max(0, hn - hlen + 1)):
                frag = head[i : i + hlen]
                if all(a in HYDROPHOBIC_SP for a in frag):
                    sp_detected = True
                    sp_region = (i + 1, min(30, i + hlen))
                    sp_conf = "weak"
                    break
            if sp_detected:
                break

    out["signal_peptide"] = {
        "detected": sp_detected,
        "region": sp_region,
        "confidence": sp_conf if sp_detected else "none",
    }

    # --- mito_transit_peptide: residues 10–70 enriched R,S,L,A (>40%) ---
    mito_detected = False
    mito_region: Optional[Tuple[int, int]] = None
    mito_conf = "none"
    hi = min(70, n)
    if hi >= 10:
        best_frac = 0.0
        best_end = 10
        for end in range(10, hi + 1):
            frag = seq[:end]
            frac = sum(a in MTP_ENRICHED for a in frag) / len(frag)
            if frac > best_frac:
                best_frac = frac
                best_end = end
        if best_frac > 0.40:
            mito_detected = True
            mito_region = (1, best_end)
            mito_conf = "strong" if best_frac >= 0.50 else "weak"

    out["mito_transit_peptide"] = {
        "detected": mito_detected,
        "region": mito_region,
        "confidence": mito_conf if mito_detected else "none",
    }

    # --- nuclear_localization_signal: 7-mer with 4+ K/R ---
    nls_detected = False
    nls_region: Optional[Tuple[int, int]] = None
    nls_conf = "none"
    if n >= 7:
        for i in range(0, n - 7 + 1):
            frag = seq[i : i + 7]
            if sum(a in POSITIVE_AA for a in frag) >= 4:
                nls_detected = True
                nls_region = (i + 1, i + 7)
                kr = sum(a in POSITIVE_AA for a in frag)
                nls_conf = "strong" if kr >= 5 else "weak"
                break

    out["nuclear_localization_signal"] = {
        "detected": nls_detected,
        "region": nls_region,
        "confidence": nls_conf if nls_detected else "none",
    }

    # --- ER retention: C-terminal KDEL / HDEL / similar ---
    er_patterns = ("KDEL", "HDEL", "RDEL", "DDEL", "KQEL", "KEEL")
    er_detected = False
    er_region: Optional[Tuple[int, int]] = None
    er_conf = "none"
    if n >= 4:
        c4 = seq[-4:]
        if c4 in ("KDEL", "HDEL"):
            er_detected = True
            er_region = (n - 3, n)
            er_conf = "strong"
        elif c4 in er_patterns or (c4[1:] == "DEL" and c4[0] in "KHRDE"):
            er_detected = True
            er_region = (n - 3, n)
            er_conf = "weak"

    out["er_retention_signal"] = {
        "detected": er_detected,
        "region": er_region,
        "confidence": er_conf if er_detected else "none",
    }

    # --- transmembrane: 18–25 consecutive hydrophobic ---
    tm_detected = False
    tm_region: Optional[Tuple[int, int]] = None
    tm_conf = "none"
    for w in range(25, 17, -1):
        for i in range(0, max(0, n - w + 1)):
            frag = seq[i : i + w]
            if all(a in HYDROPHOBIC_TM for a in frag):
                tm_detected = True
                tm_region = (i + 1, i + w)
                hyd_run = sum(a in HYDROPHOBIC_TM for a in frag)
                tm_conf = "strong" if hyd_run == w else "weak"
                break
        if tm_detected:
            break

    out["transmembrane_domain"] = {
        "detected": tm_detected,
        "region": tm_region,
        "confidence": tm_conf if tm_detected else "none",
    }

    # --- GPI anchor: hydrophobic enrichment in C-terminal 20–30 residues ---
    gpi_detected = False
    gpi_region: Optional[Tuple[int, int]] = None
    gpi_conf = "none"
    if n >= 20:
        for win in (30, 25, 20):
            if n < win:
                continue
            tail = seq[-win:]
            frac = sum(a in HYDROPHOBIC_TM for a in tail) / len(tail)
            if frac >= 0.55:
                gpi_detected = True
                gpi_region = (n - win + 1, n)
                gpi_conf = "strong" if frac >= 0.65 else "weak"
                break

    out["gpi_anchor_signal"] = {
        "detected": gpi_detected,
        "region": gpi_region,
        "confidence": gpi_conf if gpi_detected else "none",
    }

    return out


def compare_signals(original_seq: str, designed_seq: str) -> Dict[str, Any]:
    """
    Compare localization-signal heuristics between two sequences.

    Returns structured diff plus ``summary`` (human-readable paragraph).
    """
    o = check_localization_signals(original_seq)
    d = check_localization_signals(designed_seq)
    keys = sorted(set(o.keys()) | set(d.keys()))

    added: List[str] = []
    removed: List[str] = []
    preserved: List[str] = []

    for k in keys:
        od = o.get(k, {}).get("detected", False)
        dd = d.get(k, {}).get("detected", False)
        if od and dd:
            preserved.append(k)
        elif (not od) and dd:
            added.append(k)
        elif od and (not dd):
            removed.append(k)

    lines = [
        "Signal comparison (heuristic):",
        f"  Preserved (both): {', '.join(preserved) if preserved else '-'}",
        f"  Gained in design: {', '.join(added) if added else '-'}",
        f"  Lost in design: {', '.join(removed) if removed else '-'}",
    ]
    summary = "\n".join(lines)
    return {
        "original": o,
        "designed": d,
        "added": added,
        "removed": removed,
        "preserved": preserved,
        "summary": summary,
    }


def generate_design_report(relocalization_results: Mapping[str, Any]) -> str:
    """
    Build a markdown report from :meth:`ProteinRelocalizer.relocalize` output.
    """
    r = relocalization_results
    src = str(r.get("source_location", ""))
    tgt = str(r.get("target_location", ""))
    orig = str(r.get("original_sequence", ""))
    os_ = r.get("original_scores") or {}
    o_probs: Mapping[str, Any] = (os_.get("localization_probs") or {}) if isinstance(os_, dict) else {}

    lines: List[str] = [
        f"# Relocalization design report: **{src}** -> **{tgt}**",
        "",
        "## Goal",
        "",
        f"Shift predicted localization from **{src}** toward **{tgt}** using iterative mutation proposals and classifier scoring.",
        "",
        "## Original sequence",
        "",
        f"- Length: **{len(orig)}**",
        f"- P({src}): **{float(o_probs.get(src, 0.0)):.4f}**",
        f"- P({tgt}): **{float(o_probs.get(tgt, 0.0)):.4f}**",
    ]
    if isinstance(os_, dict) and "plausibility_score" in os_:
        lines.append(f"- Plausibility (PLL proxy): **{float(os_['plausibility_score']):.4f}**")

    lines.extend(["", "## Top candidates", ""])

    tops: Sequence[Mapping[str, Any]] = r.get("top_candidates") or []
    if not tops:
        lines.append("*No candidates recorded.*")
    else:
        lines.append("| Rank | Mutations | P(target) | P(source) | Composite | Plausibility |")
        lines.append("|------|-----------|-----------|-----------|-----------|--------------|")
        for c in tops[:5]:
            rank = c.get("rank", "?")
            bp = c.get("localization_probs") or {}
            p_tgt = float(bp.get(tgt, 0.0))
            p_src = float(bp.get(src, 0.0))
            muts = c.get("mutations") or []
            mut_str = ", ".join(f"{p}{a}>{b}" for p, a, b in muts[:12])
            if len(muts) > 12:
                mut_str += ", ..."
            if not mut_str:
                mut_str = "-"
            lines.append(
                f"| {rank} | {mut_str} | {p_tgt:.4f} | {p_src:.4f} | "
                f"{float(c.get('composite_score', 0.0)):.4f} | {float(c.get('plausibility_score', 0.0)):.4f} |"
            )

    lines.extend(["", "## Best candidate: signal comparison", ""])
    if tops:
        best_seq = str(tops[0].get("sequence", ""))
        sig_cmp = compare_signals(orig, best_seq)
        lines.append("```")
        lines.append(sig_cmp["summary"])
        lines.append("```")
    else:
        lines.append("*N/A - no best candidate.*")

    lines.extend(["", "## Optimization trajectory", ""])
    traj = r.get("optimization_trajectory") or []
    if not traj:
        lines.append("*Empty.*")
    else:
        lines.append("| Iter | P(target) | P(source) | Muts | Evaluated |")
        lines.append("|------|-----------|-----------|------|-----------|")
        for step in traj:
            it = step.get("iteration", "")
            pt = step.get("best_target_prob")
            ps = step.get("best_source_prob")
            nm = step.get("num_mutations", "")
            ev = step.get("num_candidates_evaluated", "")
            lines.append(
                f"| {it} | {pt if pt is not None else ''} | {ps if ps is not None else ''} | "
                f"{nm if nm != '' else '-'} | {ev if ev != '' else '-'} |"
            )

    lines.extend(
        [
            "",
            "## Run statistics",
            "",
            f"- Variants evaluated: **{int(r.get('total_variants_evaluated', 0))}**",
            f"- Wall time (s): **{float(r.get('total_time_seconds', 0.0)):.2f}**",
            "",
            "## Caveats and limitations",
            "",
            "- **Heuristic signals** (signal peptide, transit peptides, NLS, TM, GPI) are approximate sequence motifs, not experimental validation.",
            "- **Classifier probabilities** reflect the trained model and label space; they are not ground-truth localization.",
            "- **Designed sequences** should be validated experimentally before any application.",
            "- **Plausibility scores** depend on the mutation proposer’s language model and sampling; treat as a soft prior.",
            "",
        ]
    )

    return "\n".join(lines)


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate protein sequence for design pipelines.

    Returns ``(True, \"\")`` if valid; otherwise ``(False, reason)``.
    Warns (non-blocking) if the sequence does not start with methionine.
    """
    s = sequence.upper().strip()
    if not s:
        return False, "Empty sequence."
    invalid = [a for a in s if a not in STANDARD_AA]
    if invalid:
        return False, f"Non-standard amino acid(s): {sorted(set(invalid))!r}"

    L = len(s)
    if L <= 10:
        return False, f"Sequence too short (length {L}; need > 10)."
    if L >= 5000:
        return False, f"Sequence too long (length {L}; need < 5000)."

    if not s.startswith("M"):
        return True, "Warning: sequence does not start with M (methionine); many pipelines expect an N-terminal Met."

    return True, ""


def export_candidates_csv(
    results: Mapping[str, Any],
    output_path: Union[str, Path],
    *,
    source_location: Optional[str] = None,
    target_location: Optional[str] = None,
) -> None:
    """
    Write ``top_candidates`` from relocalization ``results`` to CSV.

    ``source_location`` / ``target_location`` default to keys in ``results`` if omitted.
    """
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    src = source_location or str(results.get("source_location", "source"))
    tgt = target_location or str(results.get("target_location", "target"))
    tops: Sequence[Mapping[str, Any]] = results.get("top_candidates") or []

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "sequence",
                "num_mutations",
                "mutation_list",
                "target_prob",
                "source_prob",
                "plausibility_score",
            ]
        )
        for c in tops:
            probs = c.get("localization_probs") or {}
            muts = c.get("mutations") or []
            mut_list = ";".join(f"{p}{a}>{b}" for p, a, b in muts)
            w.writerow(
                [
                    c.get("rank", ""),
                    c.get("sequence", ""),
                    c.get("num_mutations", len(muts)),
                    mut_list,
                    f'{float(probs.get(tgt, 0.0)):.6f}',
                    f'{float(probs.get(src, 0.0)):.6f}',
                    f'{float(c.get("plausibility_score", 0.0)):.6f}',
                ]
            )

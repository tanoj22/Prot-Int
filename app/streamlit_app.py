"""
ProtLoc-AI — Streamlit UI for protein localization (residue-attention classifier + analysis).
"""

from __future__ import annotations

import html
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.variant_effect import VariantEffectPredictor
from src.design.relocalizer import ProteinRelocalizer
from src.design.utils import check_localization_signals, validate_sequence
from src.models.residue_classifier import FALLBACK_LABEL_NAMES, ResidueLocalizationClassifier
from src.utils.device import resolve_torch_device

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_TOKENS = 1024
RESIDUE_CKPT = ROOT / "models" / "best_residue_model.pt"
SEQUENCE_CKPT = ROOT / "models" / "best_model.pt"

ACCENT_CYAN = "#22D3EE"
ACCENT_EMERALD = "#34D399"
SIDEBAR_BG = "#0f172a"

EXAMPLE_SEQUENCE = (
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    "KMKDGLLQISIQVVIGLKEAVQIFNKDNGFVIGWTTPNLDLRLGGYSIHSYLPLQDSSLYK"
)

SIGNAL_LABELS: List[Tuple[str, str]] = [
    ("signal_peptide", "Signal peptide"),
    ("mito_transit_peptide", "Mito transit peptide"),
    ("nuclear_localization_signal", "NLS"),
    ("er_retention_signal", "ER retention"),
    ("transmembrane_domain", "Transmembrane"),
    ("gpi_anchor_signal", "GPI anchor"),
]


def inject_css() -> None:
    st.markdown(
        f"""
<style>
    .block-container {{ padding-top: 2rem; padding-bottom: 1rem; }}
    .sequence-display {{
      font-family: 'Consolas', 'Fira Code', monospace;
      font-size: 12px; line-height: 1.6; letter-spacing: 0.5px;
      word-break: break-all; background: #f8f9fa;
      padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;
    }}
    .mut-original {{ background: #fee2e2; color: #991b1b;
      padding: 1px 3px; border-radius: 2px; font-weight: 600; }}
    .mut-designed {{ background: #dcfce7; color: #166534;
      padding: 1px 3px; border-radius: 2px; font-weight: 600; }}
    div[data-testid="stMetric"] {{
      background: #f8f9fa; padding: 12px 16px;
      border-radius: 6px; border: 1px solid #e9ecef;
    }}
    section[data-testid="stSidebar"] {{ background: {SIDEBAR_BG}; }}
    section[data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
    .experimental-badge {{
      display: inline-block; background: #ea580c; color: white;
      font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px;
      margin-left: 8px; vertical-align: middle;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def read_performance_metrics() -> Tuple[Optional[float], Optional[float], Optional[str]]:
    paths = [
        ROOT / "models" / "final_test_metrics_residue.json",
        ROOT / "models" / "final_test_metrics.json",
    ]
    for p in paths:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mf = data.get("macro_f1")
        pc = data.get("per_class") or {}
        aus: List[float] = []
        for v in pc.values():
            if not isinstance(v, dict):
                continue
            au = v.get("auroc")
            if isinstance(au, (int, float)) and not (isinstance(au, float) and math.isnan(au)):
                aus.append(float(au))
        ma = float(np.mean(aus)) if aus else None
        try:
            mf_f = float(mf) if mf is not None else None
        except (TypeError, ValueError):
            mf_f = None
        return mf_f, ma, str(p)
    return None, None, None


@st.cache_resource
def cached_device() -> torch.device:
    return resolve_torch_device(None)


@st.cache_resource
def load_esm_bundle(model_name: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_residue_classifier(checkpoint_path: str, device_str: str) -> Tuple[Any, Dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    device = torch.device(device_str)
    ckpt = torch.load(str(path), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid residue checkpoint")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    embedding_dim = int(ckpt.get("embedding_dim", 1280))
    num_labels = int(ckpt.get("num_labels", 11))
    label_names = ckpt.get("label_names")
    if not label_names:
        label_names = FALLBACK_LABEL_NAMES[:num_labels]
    dropout = float(ckpt.get("dropout", 0.3))
    num_heads = int(ckpt.get("num_heads", 4))
    model = ResidueLocalizationClassifier(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        num_heads=num_heads,
        dropout=dropout,
        label_names=label_names,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    meta = {
        "path": str(path),
        "embedding_dim": embedding_dim,
        "num_labels": num_labels,
        "label_names": list(model.label_names),
    }
    return model, meta


@st.cache_resource
def load_variant_predictor(classifier_path: str, esm_name: str, device_str: str) -> VariantEffectPredictor:
    return VariantEffectPredictor(
        classifier_path=classifier_path,
        esm_model_name=esm_name,
        device=torch.device(device_str),
    )


@st.cache_resource
def load_relocalizer(classifier_path: str, esm_name: str, device_str: str) -> ProteinRelocalizer:
    return ProteinRelocalizer(classifier_path=classifier_path, esm_model_name=esm_name, device=device_str)


@torch.inference_mode()
def embed_sequence_residue_level(
    sequence: str,
    tokenizer: Any,
    esm: Any,
    device: torch.device,
    max_length: int = MAX_SEQ_TOKENS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = sequence.upper().strip()
    enc = tokenizer(
        [seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = esm(**enc, return_dict=True)
    hidden = out.last_hidden_state
    attn = enc["attention_mask"]
    valid_len = int(attn[0].sum().item())
    if valid_len < 3:
        raise ValueError("Sequence too short after tokenization.")
    core = hidden[0, 1 : valid_len - 1, :].float()
    mask = torch.ones(core.shape[0], dtype=torch.bool, device=device)
    return core.unsqueeze(0), mask.unsqueeze(0)


def predict_residue_localization(
    sequence: str,
    tokenizer: Any,
    esm: Any,
    clf: ResidueLocalizationClassifier,
    device: torch.device,
) -> Dict[str, Any]:
    x, mask = embed_sequence_residue_level(sequence, tokenizer, esm, device)
    clf.eval()
    with torch.no_grad():
        logits = clf(x, mask=mask)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    names = clf.label_names
    return {names[i]: float(probs[i]) for i in range(len(names))}


def attention_for_sequence(
    sequence: str,
    tokenizer: Any,
    esm: Any,
    clf: ResidueLocalizationClassifier,
    device: torch.device,
) -> Tuple[np.ndarray, str, np.ndarray]:
    x, mask = embed_sequence_residue_level(sequence, tokenizer, esm, device)
    clf.eval()
    with torch.no_grad():
        _logits, attn = clf.get_attention_weights(x, mask=mask)
    seq = sequence.upper().strip()
    L = min(int(mask[0].sum().item()), len(seq), attn.shape[1])
    weights = attn[0, :L].detach().cpu().numpy().astype(np.float64)
    aas = np.array(list(seq[:L]))
    return weights, seq[:L], aas


def hot_regions_from_attention(weights: np.ndarray, window: int = 15, top_k: int = 5) -> List[Dict[str, Any]]:
    n = len(weights)
    if n == 0:
        return []
    w = max(3, min(window, n))
    best: List[Tuple[float, int, int]] = []
    for i in range(0, n - w + 1):
        seg = weights[i : i + w]
        best.append((float(seg.mean()), i, i + w - 1))
    best.sort(key=lambda t: -t[0])
    out: List[Dict[str, Any]] = []
    seen = set()
    for mean_a, a, b in best:
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append({"start_1": a + 1, "end_1": b + 1, "mean_attention": mean_a})
        if len(out) >= top_k:
            break
    return out


def plotly_prob_bars(probs: Dict[str, float], threshold: float = 0.5) -> go.Figure:
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    colors: List[str] = []
    for v in values:
        if v > 0.5:
            colors.append(ACCENT_CYAN)
        elif v >= 0.3:
            colors.append("#94a3b8")
        else:
            colors.append("#cbd5e1")
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#64748b", annotation_text="threshold 0.5")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(range=[0, 1.02], showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        yaxis=dict(showgrid=False),
        title="Localization probabilities (sorted)",
        height=max(360, 28 * len(labels)),
    )
    return fig


def plotly_attention_heatmap(
    weights: np.ndarray,
    sequence: str,
) -> go.Figure:
    L = min(len(weights), len(sequence))
    w = weights[:L]
    pos = np.arange(1, L + 1)
    aa = list(sequence[:L])
    hover = [f"Pos {i}<br>{aa[j]}<br>{w[j]:.5f}" for j, i in enumerate(pos)]
    fig = go.Figure(
        data=go.Heatmap(
            z=[w],
            x=list(pos),
            y=["attention"],
            colorscale=[
                [0.0, ACCENT_CYAN],
                [0.5, "#ffffff"],
                [1.0, ACCENT_EMERALD],
            ],
            zmin=float(w.min()),
            zmax=float(w.max()),
            text=[hover],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(l=8, r=8, t=32, b=8),
        title="Residue attention weights",
        xaxis=dict(title="Position (1-based)", showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def parse_mutation_string(raw: str) -> Tuple[List[Tuple[int, str, str]], List[str]]:
    errors: List[str] = []
    muts: List[Tuple[int, str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", part.upper())
        if not m:
            errors.append(f"Invalid mutation token: {part!r} (expected e.g. R78W)")
            continue
        o, pos_s, mu = m.group(1), m.group(2), m.group(3)
        muts.append((int(pos_s), o, mu))
    return muts, errors


def sequence_input_section(key: str) -> str:
    st.markdown("**Sequence**")
    wkey = f"wseq_{key}"
    if wkey not in st.session_state:
        st.session_state[wkey] = ""

    tab_paste, tab_uni = st.tabs(["Paste sequence", "Fetch from UniProt"])
    with tab_paste:
        st.text_area(
            "Paste",
            height=140,
            placeholder="Enter protein sequence (amino acids only)...",
            key=wkey,
            label_visibility="collapsed",
        )
    with tab_uni:
        acc = st.text_input("UniProt accession", placeholder="P38398", key=f"acc_{key}")
        if st.button("Fetch", key=f"fetch_{key}"):
            if not acc.strip():
                st.error("Enter an accession.")
            else:
                try:
                    r = requests.get(
                        f"https://rest.uniprot.org/uniprotkb/{acc.strip()}.fasta",
                        timeout=60,
                    )
                    r.raise_for_status()
                    lines = r.text.strip().splitlines()
                    sq = "".join(lines[1:]).replace("\n", "")
                    st.session_state[wkey] = sq
                    st.rerun()
                except Exception as ex:
                    st.error(f"Fetch failed: {ex}")
    if st.button("Load example", key=f"ex_{key}"):
        st.session_state[wkey] = EXAMPLE_SEQUENCE
        st.rerun()

    seq = str(st.session_state.get(wkey, "")).upper().strip()

    ok, msg = validate_sequence(seq) if seq else (False, "")
    if seq and not ok:
        st.error(msg)
    elif seq and msg.startswith("Warning"):
        st.warning(msg)

    if seq and ok:
        st.caption(f"Length: **{len(seq)}** aa")
        try:
            dev = cached_device()
            tok, esm = load_esm_bundle(ESM_MODEL_NAME)
            esm = esm.to(dev)
            if RESIDUE_CKPT.is_file():
                clf, _ = load_residue_classifier(str(RESIDUE_CKPT), str(dev))
                prev = predict_residue_localization(seq, tok, esm, clf, dev)
                top = max(prev.items(), key=lambda kv: kv[1])
                st.caption(f"Quick preview — top predicted location: **{top[0]}** ({top[1]:.3f})")
        except Exception:
            pass

    return seq if ok else ""


def render_signal_row(signals: Dict[str, Any]) -> None:
    cols = st.columns(len(SIGNAL_LABELS))
    for i, (key, label) in enumerate(SIGNAL_LABELS):
        det = bool(signals.get(key, {}).get("detected"))
        with cols[i]:
            color = ACCENT_EMERALD if det else "#94a3b8"
            st.markdown(
                f'<span style="color:{color};font-weight:600;font-size:18px;">●</span> {label}',
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="ProtLoc-AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    device = cached_device()
    macro_f1, macro_auroc, metrics_src = read_performance_metrics()

    residue_ok = RESIDUE_CKPT.is_file()
    sequence_ok = SEQUENCE_CKPT.is_file()

    with st.sidebar:
        st.caption("Protein intelligence")
        st.markdown("### ProtLoc-AI")
        mode = st.radio(
            "Mode",
            ["Predict & Analyze", "Variant Effect", "Design (Experimental)"],
            label_visibility="collapsed",
        )
        st.divider()
        with st.expander("Configuration", expanded=False):
            st.markdown(
                """
- **Model:** ESM-2 650M (Residue-attention)
- **Embedding dim:** 1280
- **Compartments:** 11
- **Classifier:** Attention-pooled MLP
"""
            )
        with st.expander("Performance", expanded=False):
            if macro_f1 is not None:
                st.metric("Macro F1", f"{macro_f1:.3f}")
            else:
                st.caption("Macro F1: — (no metrics JSON found)")
            if macro_auroc is not None:
                st.metric("AUROC (macro avg.)", f"{macro_auroc:.3f}")
            else:
                st.caption("AUROC: —")
            if metrics_src:
                st.caption(f"Source: `{Path(metrics_src).name}`")
        with st.expander("About", expanded=False):
            st.markdown(
                """
AI-powered protein variant mislocalization predictor.

Predicts how disease mutations alter subcellular localization.

Built with ESM-2, PyTorch, FastAPI.
"""
            )

    if not residue_ok:
        st.error(
            f"Residue classifier not found: `{RESIDUE_CKPT}`. "
            "Train or place `best_residue_model.pt` to enable prediction."
        )

    if mode == "Predict & Analyze":
        st.header("Predict & Analyze")
        if not residue_ok:
            return
        seq = sequence_input_section("p1")
        if st.button("Predict", type="primary", disabled=not seq):
            try:
                tok, esm = load_esm_bundle(ESM_MODEL_NAME)
                esm = esm.to(device)
                clf, _ = load_residue_classifier(str(RESIDUE_CKPT), str(device))
                with st.spinner("Running model..."):
                    probs = predict_residue_localization(seq, tok, esm, clf, device)
                    w, sseq, _aas = attention_for_sequence(seq, tok, esm, clf, device)
                sorted_p = sorted(probs.items(), key=lambda kv: -kv[1])
                top_name, top_p = sorted_p[0]
                n_above = sum(1 for _k, v in probs.items() if v >= 0.5)
                c1, c2, c3 = st.columns(3)
                c1.metric("Top location", top_name)
                c2.metric("Confidence", f"{top_p:.3f}")
                c3.metric("Locations ≥ 0.5", str(n_above))
                st.plotly_chart(plotly_prob_bars(probs), use_container_width=True)
                sigs = check_localization_signals(seq)
                with st.expander("Residue attention analysis", expanded=True):
                    loc_pick = st.selectbox("Location context", list(clf.label_names), index=0)
                    if st.button("Analyze attention"):
                        st.caption(
                            f"Attention pooling is shared across labels; "
                            f"context: **{loc_pick}** (P={probs.get(loc_pick, 0):.3f})."
                        )
                        fig_h = plotly_attention_heatmap(w, sseq)
                        st.plotly_chart(fig_h, use_container_width=True)
                        regions = hot_regions_from_attention(w)
                        cL, cR = st.columns(2)
                        with cL:
                            st.markdown("**Key regions** (sliding-window mean attention)")
                            if regions:
                                st.dataframe(pd.DataFrame(regions), hide_index=True, use_container_width=True)
                            else:
                                st.caption("—")
                        with cR:
                            st.markdown("**Detected signals** (heuristic)")
                            render_signal_row(sigs)
            except torch.cuda.OutOfMemoryError:
                st.error("GPU out of memory. Try a shorter sequence or use CPU.")
            except Exception as ex:
                st.exception(ex)

    elif mode == "Variant Effect":
        st.header("Variant Effect")
        if not sequence_ok:
            st.warning(
                f"Sequence-level checkpoint missing: `{SEQUENCE_CKPT}`. "
                "Variant analysis requires `models/best_model.pt` (ESM pooled + IG)."
            )
            return
        seq = sequence_input_section("v1")
        mut_raw = st.text_input(
            "Mutations",
            placeholder="R78W, K231A, P214L",
        )
        muts, parse_err = parse_mutation_string(mut_raw)
        for e in parse_err:
            st.error(e)
        val_err: List[str] = []
        if seq and muts:
            s = seq.upper()
            for pos, o, m in muts:
                if pos < 1 or pos > len(s):
                    val_err.append(f"Position {pos} out of range (1–{len(s)}).")
                elif s[pos - 1] != o:
                    val_err.append(
                        f"At {pos}: sequence has {s[pos - 1]!r}, mutation expects original {o!r}."
                    )
        for e in val_err:
            st.error(e)

        if st.button("Analyze variant", type="primary", disabled=not seq or bool(parse_err or val_err)):
            try:
                vep = load_variant_predictor(str(SEQUENCE_CKPT), ESM_MODEL_NAME, str(device))
                with st.spinner("Computing variant effect (embeddings + IG)..."):
                    out = vep.predict_variant_effect(seq, muts)
                risk = str(out.get("mislocalization_risk", "none")).lower()
                if risk == "high":
                    st.markdown(
                        '<div style="background:#fecaca;color:#7f1d1d;padding:12px 16px;border-radius:6px;'
                        'font-weight:600;">High mislocalization risk — mutation may disrupt localization signals</div>',
                        unsafe_allow_html=True,
                    )
                elif risk == "medium":
                    st.markdown(
                        '<div style="background:#fef9c3;color:#854d0e;padding:12px 16px;border-radius:6px;'
                        'font-weight:600;">Medium mislocalization risk</div>',
                        unsafe_allow_html=True,
                    )
                elif risk == "low":
                    st.markdown(
                        '<div style="background:#dcfce7;color:#166534;padding:12px 16px;border-radius:6px;'
                        'font-weight:600;">Low mislocalization risk</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background:#e2e8f0;color:#334155;padding:12px 16px;border-radius:6px;'
                        'font-weight:600;">No strong localization shift detected</div>',
                        unsafe_allow_html=True,
                    )

                dmap = out.get("deltas") or {}
                most_aff = out.get("most_affected_location", "")
                gain_loc = max(dmap, key=lambda k: float(dmap[k]))
                loss_loc = min(dmap, key=lambda k: float(dmap[k]))
                c1, c2, c3 = st.columns(3)
                c1.metric("Most affected location", f"{most_aff}", delta=f"{float(dmap.get(most_aff, 0)):+.3f}")
                c2.metric("Largest increase", gain_loc, delta=f"{float(dmap[gain_loc]):+.3f}")
                c3.metric("Largest decrease", loss_loc, delta=f"{float(dmap[loss_loc]):+.3f}")

                p0 = out["original_predictions"]
                pm = out["mutant_predictions"]
                locs = list(p0.keys())
                fig = go.Figure()
                x = np.arange(len(locs))
                fig.add_trace(
                    go.Bar(name="Original", x=locs, y=[p0[k] for k in locs], marker_color="#94a3b8")
                )
                colors_m = []
                for k in locs:
                    dlt = float(pm[k] - p0[k])
                    if dlt >= 0:
                        colors_m.append(ACCENT_CYAN)
                    else:
                        colors_m.append("#f87171")
                fig.add_trace(go.Bar(name="Mutant", x=locs, y=[pm[k] for k in locs], marker_color=colors_m))
                fig.update_layout(barmode="group", template="plotly_white", height=480, title="Original vs mutant")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Signal analysis"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original**")
                        render_signal_row(out.get("signals_original") or {})
                    with c2:
                        st.markdown("**Mutant**")
                        render_signal_row(out.get("signals_mutant") or {})
                    disrupted = out.get("signals_disrupted") or []
                    gained = out.get("signals_gained") or []
                    st.caption(f"Disrupted vs original: **{', '.join(disrupted) or '—'}**")
                    st.caption(f"Gained: **{', '.join(gained) or '—'}**")

                with st.expander("Interpretability comparison"):
                    io = out.get("interpretation_original") or {}
                    im = out.get("interpretation_mutant") or {}
                    rs0 = io.get("residue_scores") or []
                    rsm = im.get("residue_scores") or []

                    def scores_to_arr(scores: Any, n: int) -> np.ndarray:
                        arr = np.zeros(max(n, 1), dtype=np.float64)
                        for item in scores:
                            if isinstance(item, (list, tuple)) and len(item) >= 3:
                                pos, _aa, sc = int(item[0]), str(item[1]), float(item[2])
                                if 1 <= pos <= len(arr):
                                    arr[pos - 1] = sc
                        return arr[:n]

                    nseq = len(seq)
                    a0 = scores_to_arr(rs0, nseq)
                    a1 = scores_to_arr(rsm, nseq)
                    Lm = min(len(a0), len(a1))
                    if Lm > 0:
                        z = np.vstack([a0[:Lm], a1[:Lm]])
                        fig_i = go.Figure(
                            data=go.Heatmap(
                                z=z,
                                x=list(range(1, Lm + 1)),
                                y=["IG original", "IG mutant"],
                                colorscale="Blues",
                            )
                        )
                        diff = np.abs(a0[:Lm] - a1[:Lm])
                        top_shift = int(np.argmax(diff)) + 1 if Lm else 0
                        st.caption(f"Largest |ΔIG| at position **{top_shift}**")
                        fig_i.update_layout(template="plotly_white", height=260)
                        st.plotly_chart(fig_i, use_container_width=True)

                st.markdown("**Clinical summary**")
                st.markdown(
                    f'<div class="sequence-display">{out.get("clinical_summary", "")}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Mutation scan"):
                    st.caption("This may take several minutes on long windows.")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        rs = st.number_input("Start", min_value=1, value=1, step=1)
                    with c2:
                        re_default = len(seq) if seq else 100
                        re = st.number_input("End", min_value=1, value=max(1, re_default), step=1)
                    with c3:
                        step = st.slider("Step", 1, 10, 5)
                    if st.button("Run scan"):
                        prog = st.progress(0, text="Scanning...")
                        try:
                            scan = vep.scan_single_mutations(
                                seq,
                                region_start=int(rs),
                                region_end=int(min(re, len(seq))),
                                step=int(step),
                                top_k=20,
                                batch_size=8,
                            )
                            prog.progress(100, text="Done.")
                            hd = scan.get("heatmap_data") or {}
                            pos = hd.get("positions") or []
                            deltas = hd.get("max_delta_per_position") or []
                            if pos and deltas:
                                fig_s = go.Figure(
                                    data=go.Heatmap(
                                        z=[deltas],
                                        x=pos,
                                        y=["|Δ| max"],
                                        colorscale=[[0, "#ffffff"], [1, "#dc2626"]],
                                    )
                                )
                                fig_s.update_layout(template="plotly_white", height=240, title="Sensitivity by position")
                                st.plotly_chart(fig_s, use_container_width=True)
                            tops = scan.get("top_mutations") or []
                            st.dataframe(pd.DataFrame(tops[:10]), use_container_width=True)
                        finally:
                            prog.progress(100, text="Finished.")

            except Exception as ex:
                st.exception(ex)

    else:
        st.markdown(
            '<h2 style="display:inline;">Design</h2>'
            '<span class="experimental-badge">Experimental</span>',
            unsafe_allow_html=True,
        )
        if not sequence_ok:
            st.warning(f"Requires `{SEQUENCE_CKPT}` for ProteinRelocalizer.")
            return
        seq = sequence_input_section("d1")
        rel = load_relocalizer(str(SEQUENCE_CKPT), ESM_MODEL_NAME, str(device))
        labs = rel.label_names
        src = st.selectbox("Source location", labs, index=min(2, len(labs) - 1))
        tgt = st.selectbox("Target location", labs, index=min(0, len(labs) - 1))
        max_mut = st.slider("Max mutations", 1, 50, 15)
        n_iter = st.slider("Iterations", 1, 20, 10)
        if st.button("Start design", type="primary", disabled=not seq):
            bar = st.progress(0, text="Initializing...")
            try:
                results = rel.relocalize(
                    seq,
                    source_location=src,
                    target_location=tgt,
                    n_iterations=int(n_iter),
                    max_total_mutations=int(max_mut),
                    candidates_per_iteration=20,
                )
                bar.progress(100, text="Complete.")
                summ = rel.get_summary(results)
                st.markdown(
                    '<div class="sequence-display" style="font-family:system-ui,sans-serif;line-height:1.5;">'
                    + html.escape(summ[:1500])
                    + ("…" if len(summ) > 1500 else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                os_ = results.get("original_scores") or {}
                lp0 = os_.get("localization_probs") or {}
                tops = results.get("top_candidates") or []
                best = tops[0] if tops else None
                if best:
                    bp = best.get("localization_probs") or {}
                    d_src = float(bp.get(src, 0) - float(lp0.get(src, 0)))
                    d_tgt = float(bp.get(tgt, 0) - float(lp0.get(tgt, 0)))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"P({src}) change", f"{d_src:+.3f}")
                    m2.metric(f"P({tgt}) change", f"{d_tgt:+.3f}")
                    m3.metric("Mutations (best)", str(best.get("num_mutations", 0)))
                    m4.metric("Variants scored", str(results.get("total_variants_evaluated", 0)))
                traj = results.get("optimization_trajectory") or []
                if traj:
                    xs = [t.get("iteration") for t in traj]
                    ys = [t.get("best_target_prob", 0) for t in traj]
                    fig_t = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=ACCENT_CYAN)))
                    fig_t.update_layout(template="plotly_white", title="Target probability trajectory", height=360)
                    st.plotly_chart(fig_t, use_container_width=True)
                rows = []
                for t in tops[:5]:
                    rows.append(
                        {
                            "rank": t.get("rank"),
                            "mutations": len(t.get("mutations") or []),
                            "P_target": float((t.get("localization_probs") or {}).get(tgt, 0)),
                            "composite": float(t.get("composite_score", 0)),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                for i, cand in enumerate(tops[:5], start=1):
                    with st.expander(f"Candidate {i}"):
                        oseq = results.get("original_sequence", "")
                        cseq = cand.get("sequence", "")
                        muts = cand.get("mutations") or []
                        parts = []
                        for j, ch in enumerate(cseq):
                            if j < len(oseq) and oseq[j] != ch:
                                parts.append(
                                    f'<span class="mut-original">{oseq[j]}</span>'
                                    f'<span class="mut-designed">{ch}</span>'
                                )
                            else:
                                parts.append(ch)
                        st.markdown(
                            '<div class="sequence-display">' + "".join(parts) + "</div>",
                            unsafe_allow_html=True,
                        )
                        render_signal_row(check_localization_signals(cseq))
                csv_buf = io.StringIO()
                pd.DataFrame(rows).to_csv(csv_buf, index=False)
                st.download_button("Download CSV", csv_buf.getvalue(), file_name="protloc_design.csv")
                md = rel.get_summary(results)
                st.download_button("Download report (markdown)", md, file_name="protloc_design_report.md")
            except Exception as ex:
                st.exception(ex)


if __name__ == "__main__":
    main()

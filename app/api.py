from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoModel, AutoTokenizer

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import (  # noqa: E402
    HealthResponse,
    MutationScanRequest,
    MutationScanResponse,
    MutationScanTopItem,
    PredictRequest,
    PredictResponse,
    VariantEffectRequest,
    VariantEffectResponse,
)
from src.design.utils import check_localization_signals, compare_signals  # noqa: E402
from src.models.residue_classifier import FALLBACK_LABEL_NAMES, ResidueLocalizationClassifier  # noqa: E402

HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_URL = "https://router.huggingface.co/v1/chat/completions"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_REQUEST_TIMEOUT = 20.0
LLM_VARIANT_WAIT = 18.0
LLM_PREDICT_WAIT = 12.0

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1024
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
ALL_AA_SORTED = sorted(AA20)


def resolve_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _validate_sequence(sequence: str) -> str:
    s = str(sequence).upper().strip()
    if not s:
        raise HTTPException(status_code=400, detail="Sequence is empty.")
    invalid = sorted({a for a in s if a not in AA20})
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid amino acids found: {invalid}")
    if len(s) < 10:
        raise HTTPException(status_code=400, detail=f"Sequence too short ({len(s)}). Need >= 10.")
    if len(s) > 5000:
        raise HTTPException(status_code=400, detail=f"Sequence too long ({len(s)}). Need <= 5000.")
    return s


def _risk_from_delta(abs_delta: float) -> str:
    if abs_delta > 0.3:
        return "high"
    if abs_delta >= 0.15:
        return "medium"
    if abs_delta >= 0.05:
        return "low"
    return "none"


def _parse_chat_completion_response(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices")
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        return None
    text = str(content).strip()
    return text if text else None


async def _hf_generate_text(prompt: str) -> Optional[str]:
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
            response = await client.post(LLM_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return _parse_chat_completion_response(data)
            if response.status_code == 503:
                await asyncio.sleep(5)
                response = await client.post(LLM_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return _parse_chat_completion_response(data)
    except Exception as e:
        print(f"LLM summary failed: {e}")
    return None


async def generate_clinical_summary(
    original_predictions: Dict[str, float],
    mutant_predictions: Dict[str, float],
    deltas: Dict[str, float],
    mutations: List[Dict[str, Any]],
    signals_disrupted: List[str],
    signals_gained: List[str],
    mislocalization_risk: str,
    sequence_length: int,
) -> Optional[str]:
    if not HF_TOKEN:
        return None

    sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    top3 = sorted_deltas[:3]
    lines: List[str] = []
    for loc, delta in top3:
        orig = float(original_predictions.get(loc, 0.0))
        newv = orig + float(delta)
        lines.append(f"  {loc}: {orig:.3f} -> {newv:.3f} (delta {delta:+.3f})")
    top_changes_str = "\n".join(lines)
    mut_strs = [f"{m['original']}{m['position']}{m['mutant']}" for m in mutations]
    top_mut_loc = max(mutant_predictions.items(), key=lambda kv: kv[1])[0]
    top_mut_val = float(mutant_predictions[top_mut_loc])

    prompt = f"""You are a computational biology expert writing a clinical report. Analyze this protein variant effect.

Protein: {sequence_length} residues
Mutations applied: {', '.join(mut_strs)}
Strongest mutant localization signal: {top_mut_loc} ({top_mut_val:.3f})

Top localization changes:
{top_changes_str}

Mislocalization risk: {mislocalization_risk}
Signals disrupted: {signals_disrupted if signals_disrupted else 'None'}
Signals gained: {signals_gained if signals_gained else 'None'}

Write exactly 2-3 sentences. Be specific about which compartments changed most and the biological mechanism, and what disease pattern this is consistent with. No bullet points. No hedging. Write like a clinical report."""

    return await _hf_generate_text(prompt)


async def generate_predict_interpretation(
    predictions: Dict[str, float],
    sequence_length: int,
) -> Optional[str]:
    if not HF_TOKEN:
        return None
    pred_str = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(predictions.items(), key=lambda x: -x[1])[:12])
    prompt = (
        f"You are a computational biology expert. Briefly interpret this protein's "
        f"localization profile ({sequence_length} residues). "
        f"Compartment scores: {pred_str}. "
        f"Write exactly 2 short sentences on likely subcellular role and relevance. "
        f"No bullet points."
    )
    return await _hf_generate_text(prompt)


def _apply_mutations(sequence: str, mutations: Sequence[Tuple[int, str, str]]) -> str:
    """Apply AA substitutions. Each mutation tuple is (position_1based, original, mutant).

    Positions are **1-based** (human numbering; first residue is 1). All `sequence[...]` access
    uses **0-based** indices: ``actual_index = position_1based - 1``.
    """
    seq = list(sequence.upper().strip())
    n = len(seq)
    for pos_1based, orig, mut in mutations:
        if pos_1based < 1 or pos_1based > n:
            raise HTTPException(
                status_code=400,
                detail=f"Mutation position {pos_1based} out of range 1..{n} (1-based).",
            )
        actual_index = pos_1based - 1
        o = orig.upper().strip()
        m = mut.upper().strip()
        if len(o) != 1 or len(m) != 1:
            raise HTTPException(status_code=400, detail=f"Mutation at {pos_1based} must use single-letter amino acids.")
        if o not in AA20:
            raise HTTPException(status_code=400, detail=f"Invalid original amino acid at {pos_1based}: {o}")
        if m not in AA20:
            raise HTTPException(status_code=400, detail=f"Invalid mutant amino acid at {pos_1based}: {m}")
        if seq[actual_index] != o:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Original AA mismatch at position {pos_1based} (1-based): "
                    f"expected {seq[actual_index]}, got {o}."
                ),
            )
        seq[actual_index] = m
    return "".join(seq)


def embed_sequence(
    sequence: str,
    esm_model: Any,
    tokenizer: Any,
    device: torch.device,
) -> torch.Tensor:
    seq = _validate_sequence(sequence)
    toks = tokenizer(
        [seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = esm_model(**toks, return_dict=True)
    hidden = out.last_hidden_state  # [1, T, D]
    attn = toks["attention_mask"]
    valid_len = int(attn[0].sum().item())
    if valid_len < 3:
        raise HTTPException(status_code=400, detail="Sequence too short after tokenization.")
    core = hidden[0, 1 : valid_len - 1, :].float()  # (L, 1280)
    return core


def _embed_sequences_batched(
    sequences: Sequence[str],
    esm_model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 16,
) -> List[torch.Tensor]:
    out_list: List[torch.Tensor] = []
    for i in range(0, len(sequences), batch_size):
        batch = [str(s).upper().strip() for s in sequences[i : i + batch_size]]
        toks = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = esm_model(**toks, return_dict=True)
        hidden = out.last_hidden_state
        attn = toks["attention_mask"]
        for b in range(hidden.shape[0]):
            valid_len = int(attn[b].sum().item())
            if valid_len < 3:
                out_list.append(torch.zeros((1, hidden.shape[-1]), dtype=torch.float32, device=device))
            else:
                out_list.append(hidden[b, 1 : valid_len - 1, :].float())
    return out_list


def _pad_embeddings(embeddings: Sequence[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not embeddings:
        raise ValueError("No embeddings to pad.")
    max_len = max(int(e.shape[0]) for e in embeddings)
    dim = int(embeddings[0].shape[1])
    bsz = len(embeddings)
    x = torch.zeros((bsz, max_len, dim), dtype=torch.float32, device=device)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    for i, e in enumerate(embeddings):
        L = int(e.shape[0])
        x[i, :L, :] = e
        mask[i, :L] = True
    return x, mask


def _predict_from_embeddings(
    embeddings: Sequence[torch.Tensor],
    model: ResidueLocalizationClassifier,
    label_names: Sequence[str],
    device: torch.device,
) -> Tuple[List[Dict[str, float]], List[List[float]]]:
    x, mask = _pad_embeddings(embeddings, device=device)
    with torch.no_grad():
        logits, attn = model.get_attention_weights(x, mask=mask)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds: List[Dict[str, float]] = []
    atts: List[List[float]] = []
    for i in range(len(embeddings)):
        L = int(mask[i].sum().item())
        preds.append({str(label_names[j]): float(probs[i, j]) for j in range(len(label_names))})
        atts.append(attn[i, :L].detach().cpu().tolist())
    return preds, atts


def _fetch_uniprot_sequence(uniprot_id: str) -> str:
    uid = str(uniprot_id).strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uniprot_id is empty.")
    try:
        resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=60)
        resp.raise_for_status()
    except requests.RequestException as ex:
        raise HTTPException(status_code=400, detail=f"Failed to fetch UniProt {uid}: {ex}") from ex
    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        raise HTTPException(status_code=400, detail=f"Invalid FASTA response from UniProt for {uid}.")
    seq = "".join(lines[1:]).upper().strip()
    return _validate_sequence(seq)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device()
    ckpt_path = ROOT / "models" / "best_residue_model.pt"
    if not ckpt_path.is_file():
        raise RuntimeError(f"Missing model checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid residue checkpoint format.")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    embedding_dim = int(ckpt.get("embedding_dim", 1280))
    num_labels = int(ckpt.get("num_labels", 11))
    label_names = list(ckpt.get("label_names") or FALLBACK_LABEL_NAMES[:num_labels])
    model = ResidueLocalizationClassifier(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        label_names=label_names,
        dropout=float(ckpt.get("dropout", 0.3)),
        num_heads=int(ckpt.get("num_heads", 4)),
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = AutoModel.from_pretrained(
        ESM_MODEL_NAME,
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )
    esm_model.eval().to(device)

    app.state.device = device
    app.state.classifier = model
    app.state.label_names = label_names
    app.state.tokenizer = tokenizer
    app.state.esm_model = esm_model
    app.state.model_loaded = True
    print(f"[startup] ProtLoc-AI models loaded on device={device}")

    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[shutdown] ProtLoc-AI API shutting down.")


app = FastAPI(title="ProtLoc-AI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    return FileResponse("app/frontend.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        device=str(getattr(app.state, "device", "cpu")),
        model_loaded=bool(getattr(app.state, "model_loaded", False)),
    )


@app.get("/locations", response_model=List[str])
def locations() -> List[str]:
    return list(getattr(app.state, "label_names", []))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request) -> PredictResponse:
    t0 = time.perf_counter()
    try:
        seq = req.sequence or ""
        if req.uniprot_id:
            seq = _fetch_uniprot_sequence(req.uniprot_id)
        seq = _validate_sequence(seq)
        emb = embed_sequence(seq, app.state.esm_model, app.state.tokenizer, app.state.device)
        preds, atts = _predict_from_embeddings(
            [emb],
            app.state.classifier,
            app.state.label_names,
            app.state.device,
        )
        pred = preds[0]
        attn = atts[0]
        top_location, top_confidence = max(pred.items(), key=lambda kv: kv[1])
        above = [k for k, v in pred.items() if float(v) >= 0.5]

        interpretation_summary: Optional[str] = None
        llm_enhanced = False
        if HF_TOKEN:
            try:
                interpretation_summary = await asyncio.wait_for(
                    generate_predict_interpretation(pred, len(seq)),
                    timeout=LLM_PREDICT_WAIT,
                )
                if interpretation_summary:
                    llm_enhanced = True
            except asyncio.TimeoutError:
                pass
            except Exception as ex:
                print(f"LLM predict interpretation failed: {ex}")

        return PredictResponse(
            predictions=pred,
            attention_weights=[float(x) for x in attn],
            sequence_length=len(seq),
            sequence=seq,
            top_location=str(top_location),
            top_confidence=float(top_confidence),
            locations_above_threshold=above,
            interpretation_summary=interpretation_summary,
            llm_enhanced=llm_enhanced,
        )
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Inference failed: {ex}") from ex
    finally:
        dt = time.perf_counter() - t0
        print(f"[request] {request.url.path} len={len((req.sequence or '').strip())} time={dt:.3f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/variant-effect", response_model=VariantEffectResponse)
async def variant_effect(req: VariantEffectRequest, request: Request) -> VariantEffectResponse:
    t0 = time.perf_counter()
    try:
        seq0 = _validate_sequence(req.sequence)
        # Request uses 1-based positions (human numbering). _apply_mutations converts to 0-based for indexing.
        mutations_1based = [(int(m.position), m.original.upper(), m.mutant.upper()) for m in req.mutations]
        if not mutations_1based:
            raise HTTPException(status_code=400, detail="mutations is empty.")
        seqm = _apply_mutations(seq0, mutations_1based)

        emb0 = embed_sequence(seq0, app.state.esm_model, app.state.tokenizer, app.state.device)
        embm = embed_sequence(seqm, app.state.esm_model, app.state.tokenizer, app.state.device)
        preds, atts = _predict_from_embeddings(
            [emb0, embm],
            app.state.classifier,
            app.state.label_names,
            app.state.device,
        )
        p0, pm = preds[0], preds[1]
        a0, am = atts[0], atts[1]
        deltas = {k: float(pm[k] - p0[k]) for k in app.state.label_names}
        most = max(app.state.label_names, key=lambda n: abs(deltas[n]))
        max_delta = float(deltas[most])
        risk = _risk_from_delta(abs(max_delta))

        sig0 = check_localization_signals(seq0)
        sigm = check_localization_signals(seqm)
        cmp_sig = compare_signals(seq0, seqm)
        disrupted = list(cmp_sig.get("removed", []))
        gained = list(cmp_sig.get("added", []))
        mut_txt = ", ".join(f"{p}{o}>{m}" for p, o, m in mutations_1based)
        top_gain = max(app.state.label_names, key=lambda n: deltas[n])
        top_loss = min(app.state.label_names, key=lambda n: deltas[n])
        clinical = (
            f"Mutation(s) {mut_txt} most strongly affect {most} (delta={max_delta:+.3f}). "
            f"P({most}) changes {p0[most]:.2f} -> {pm[most]:.2f}. "
            f"Largest gain: {top_gain} ({deltas[top_gain]:+.3f}); "
            f"largest loss: {top_loss} ({deltas[top_loss]:+.3f})."
        )
        if disrupted:
            clinical += f" Disrupted signal(s): {', '.join(disrupted)}."
        if gained:
            clinical += f" Gained signal(s): {', '.join(gained)}."

        llm_enhanced = False
        if HF_TOKEN:
            try:
                llm_summary = await asyncio.wait_for(
                    generate_clinical_summary(
                        p0,
                        pm,
                        deltas,
                        [{"original": m.original, "position": m.position, "mutant": m.mutant} for m in req.mutations],
                        disrupted,
                        gained,
                        risk,
                        len(seq0),
                    ),
                    timeout=LLM_VARIANT_WAIT,
                )
                if llm_summary:
                    clinical = llm_summary
                    llm_enhanced = True
            except asyncio.TimeoutError:
                pass
            except Exception as ex:
                print(f"LLM clinical summary failed: {ex}")

        return VariantEffectResponse(
            original_predictions=p0,
            mutant_predictions=pm,
            deltas=deltas,
            most_affected_location=str(most),
            max_delta=max_delta,
            mislocalization_risk=risk,
            clinical_summary=clinical,
            llm_enhanced=llm_enhanced,
            original_attention=[float(x) for x in a0],
            mutant_attention=[float(x) for x in am],
            signals_original=sig0,
            signals_mutant=sigm,
            signals_disrupted=disrupted,
            signals_gained=gained,
            original_sequence=seq0,
            mutant_sequence=seqm,
        )
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Variant-effect inference failed: {ex}") from ex
    finally:
        dt = time.perf_counter() - t0
        print(f"[request] {request.url.path} len={len(req.sequence.strip())} time={dt:.3f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/mutation-scan", response_model=MutationScanResponse)
def mutation_scan(req: MutationScanRequest, request: Request) -> MutationScanResponse:
    t0 = time.perf_counter()
    try:
        seq = _validate_sequence(req.sequence)
        n = len(seq)
        rs = int(req.start)
        re = int(req.end) if req.end is not None else n
        stp = int(req.step)
        if rs < 1 or re > n or rs > re:
            raise HTTPException(status_code=400, detail=f"Invalid range [{rs}, {re}] for sequence length {n}.")

        base_emb = embed_sequence(seq, app.state.esm_model, app.state.tokenizer, app.state.device)
        base_pred, _ = _predict_from_embeddings([base_emb], app.state.classifier, app.state.label_names, app.state.device)
        base_map = base_pred[0]

        variants: List[Tuple[int, str, str, str]] = []
        positions = list(range(rs, re + 1, stp))
        for pos in positions:
            orig = seq[pos - 1]
            for aa in ALL_AA_SORTED:
                if aa == orig:
                    continue
                seqm = seq[: pos - 1] + aa + seq[pos:]
                variants.append((pos, orig, aa, seqm))

        all_seqs = [v[3] for v in variants]
        embs = _embed_sequences_batched(
            all_seqs,
            app.state.esm_model,
            app.state.tokenizer,
            app.state.device,
            batch_size=16,
        )
        pred_list, _ = _predict_from_embeddings(embs, app.state.classifier, app.state.label_names, app.state.device)

        rows: List[MutationScanTopItem] = []
        per_pos_max: Dict[int, float] = {p: 0.0 for p in positions}
        per_pos_loc: Dict[int, str] = {p: "none" for p in positions}
        for i, (pos, orig, aa, _seqm) in enumerate(variants):
            pm = pred_list[i]
            deltas = {name: float(pm[name] - base_map[name]) for name in app.state.label_names}
            loc = max(app.state.label_names, key=lambda n_: abs(deltas[n_]))
            delta = float(deltas[loc])
            absd = abs(delta)
            if absd > per_pos_max[pos]:
                per_pos_max[pos] = absd
                per_pos_loc[pos] = str(loc)
            rows.append(
                MutationScanTopItem(
                    position=int(pos),
                    original=str(orig),
                    mutant=str(aa),
                    max_delta=delta,
                    location=str(loc),
                )
            )
        rows.sort(key=lambda x: abs(float(x.max_delta)), reverse=True)
        elapsed = time.perf_counter() - t0
        return MutationScanResponse(
            positions=positions,
            max_delta_per_position=[float(per_pos_max[p]) for p in positions],
            most_affected_per_position=[str(per_pos_loc[p]) for p in positions],
            top_mutations=rows[:20],
            total_variants_scored=len(rows),
            time_seconds=float(elapsed),
        )
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Mutation scan failed: {ex}") from ex
    finally:
        dt = time.perf_counter() - t0
        print(f"[request] {request.url.path} len={len(req.sequence.strip())} time={dt:.3f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


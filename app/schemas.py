from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: Optional[str] = None
    uniprot_id: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: Dict[str, float]
    attention_weights: List[float]
    sequence_length: int
    sequence: str
    top_location: str
    top_confidence: float
    locations_above_threshold: List[str]
    interpretation_summary: Optional[str] = None
    llm_enhanced: bool = False


class MutationItem(BaseModel):
    position: int = Field(
        ...,
        ge=1,
        description="1-based residue index in the protein sequence (first residue is 1).",
    )
    original: str = Field(..., min_length=1, max_length=1)
    mutant: str = Field(..., min_length=1, max_length=1)


class VariantEffectRequest(BaseModel):
    sequence: str
    mutations: List[MutationItem]


class VariantEffectResponse(BaseModel):
    original_predictions: Dict[str, float]
    mutant_predictions: Dict[str, float]
    deltas: Dict[str, float]
    most_affected_location: str
    max_delta: float
    mislocalization_risk: str
    clinical_summary: str
    llm_enhanced: bool = False
    original_attention: List[float]
    mutant_attention: List[float]
    signals_original: Dict[str, Dict[str, Any]]
    signals_mutant: Dict[str, Dict[str, Any]]
    signals_disrupted: List[str]
    signals_gained: List[str]
    original_sequence: str
    mutant_sequence: str


class MutationScanRequest(BaseModel):
    sequence: str
    start: int = Field(1, ge=1)
    end: Optional[int] = Field(default=None, ge=1)
    step: int = Field(5, ge=1, le=10)


class MutationScanTopItem(BaseModel):
    position: int
    original: str
    mutant: str
    max_delta: float
    location: str


class MutationScanResponse(BaseModel):
    positions: List[int]
    max_delta_per_position: List[float]
    most_affected_per_position: List[str]
    top_mutations: List[MutationScanTopItem]
    total_variants_scored: int
    time_seconds: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


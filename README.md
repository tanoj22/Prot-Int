# ProtLoc-AI: Protein Variant Mislocalization Predictor

AI-powered platform that predicts how disease mutations alter protein subcellular localization using ESM-2 protein language models with residue-level attention.

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![ESM-2](https://img.shields.io/badge/ESM--2-Protein%20LM-6f42c1)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Protein mislocalization causes disease across multiple systems, including cystic fibrosis, ALS, and cancer. Roughly 1 in 6 pathogenic mutations are associated with incorrect subcellular trafficking. ProtLoc-AI predicts localization across 11 compartments and quantifies how mutations shift localization probabilities. The model stack uses ESM-2 residue embeddings and an attention-based classifier to preserve mutation-local context. In variant sensitivity benchmarks, the residue-attention model is 3.7x more responsive to mutation effects than mean-pooled baselines.

## Live Demo

- Hugging Face Space: [Coming soon](https://huggingface.co/spaces/your-org/protloc-ai)
- Screenshot placeholder: `docs/images/demo-placeholder.png`

## Key Features

- Localization prediction: 11 compartments, 0.93 AUROC, 0.738 macro F1.
- Variant effect analysis: 3.7x higher mutation sensitivity with mislocalization risk scoring.
- Residue interpretability: attention-weighted residue signals with biological validation hooks.

## Model Comparison Table

| Metric | Mean-pooled | Residue-attention |
| --- | --- | --- |
| Macro F1 | 0.731 | 0.738 |
| AUROC | 0.931 | 0.932 |
| Mutation sensitivity | 0.022 avg delta | 0.082 avg delta (3.7x) |
| Peroxisome F1 | 0.519 | 0.682 |

## Variant Effect Comparison Table

| Test | Mutations | Mean-pooled max delta | Residue max delta |
| --- | --- | --- | --- |
| Single (F45D) | 1 | 0.031 | 0.045 |
| Triple | 3 | 0.050 | 0.091 |
| Five mutations | 5 | 0.148 | 0.568 |

## Quick Start

```bash
git clone https://github.com/your-org/protloc-ai.git
cd protloc-ai
pip install -r requirements.txt
# Generate embeddings
python -m src.data.generate_residue_embeddings --device cuda
# Train
python -m src.training.train_residue
# Run
uvicorn app.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

## Model Weights

Model checkpoints are intentionally excluded from GitHub to keep the repository lightweight.  
The API expects this file at runtime:

- `models/best_residue_model.pt`

If the file is missing, API startup will fail with a checkpoint-not-found error.  
To reproduce weights locally, run training and place/save the best checkpoint to the path above before starting `uvicorn`.

## Project Structure

```text
Protein_Seq/
├── app/
│   ├── api.py
│   ├── frontend.html
│   └── schemas.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── design/
├── scripts/
├── models/
├── data/
├── docs/
├── requirements.txt
└── README.md
```

## Architecture

Sequence -> ESM-2 (650M) -> Per-residue embeddings (seq_len x 1280) -> Learned attention pooling -> MLP classifier -> 11 compartment probabilities.

Attention weights provide built-in interpretability at inference time. Variant analysis performs paired wild-type and mutant predictions, then compares compartment probability deltas and attention shifts to estimate trafficking disruption risk.

## Dataset

DeepLoc dataset, 28,303 proteins, 11 subcellular compartments. The task is multi-label, so a protein can belong to multiple locations.

## Tech Stack

- ESM-2: pretrained protein language model for residue-level embeddings.
- PyTorch: training and inference framework for localization classifiers.
- FastAPI: backend API serving prediction and variant-analysis endpoints.
- Three.js: interactive 3D background and visual polish in the UI.
- Plotly: probability, delta, and attention visualizations.
- MLflow: experiment tracking and metric comparison.
- DVC: dataset and artifact versioning.
- Docker: containerized deployment and environment reproducibility.

## Future Work

- Fine-tune ESM with LoRA for localization-specific embeddings.
- Integrate ClinVar for automatic pathogenic variant lookup.
- Experimentally validate predicted mislocalization events.

## License

MIT

## Citation

If you use this project, cite the ESM-2 paper and the DeepLoc dataset.

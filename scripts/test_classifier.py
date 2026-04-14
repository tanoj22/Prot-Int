"""
Smoke-test src/models/classifier.py.

From project root:
    .\\venv\\Scripts\\python.exe scripts\\test_classifier.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classifier import (  # noqa: E402
    ProteinLocalizationClassifier,
    count_parameters,
    load_model,
)


def main() -> None:
    torch.manual_seed(0)

    label_columns_path = (
        ROOT / "data" / "processed" / "embeddings" / "esm2_t33_650M" / "label_columns.json"
    )
    with label_columns_path.open("r", encoding="utf-8") as f:
        label_names = json.load(f)["label_columns"]

    embedding_dim = 1280
    num_labels = len(label_names)
    model = ProteinLocalizationClassifier(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        label_names=label_names,
    )

    print("Model created.")
    count_parameters(model)
    print(f"label_names ({len(model.label_names)}): {model.label_names}")

    x1 = torch.randn(embedding_dim)
    xB = torch.randn(4, embedding_dim)

    # BatchNorm requires eval mode for batch size 1 in inference-like checks.
    model.eval()
    with torch.no_grad():
        logits1 = model(x1.unsqueeze(0))
        logitsB = model(xB)
    print(f"\nforward single (as batch): logits shape={tuple(logits1.shape)}")
    print(f"forward batch: logits shape={tuple(logitsB.shape)}")

    p1 = model.predict_proba(x1)
    pB = model.predict_proba(xB)
    print(f"\npredict_proba single keys={list(p1.keys())[:3]}... len={len(p1)}")
    print(f"predict_proba batch len={len(pB)}; first item Membrane={pB[0]['Membrane']:.4f}")

    y1 = model.predict(x1)
    yB = model.predict(xB, thresholds={"Membrane": 0.4})
    print(f"\npredict single example: Membrane={y1['Membrane']}, Cytoplasm={y1['Cytoplasm']}")
    print(f"predict batch len={len(yB)}; first item Membrane={yB[0]['Membrane']}")

    # Optional: checkpoint round-trip
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "classifier_ckpt.pt"
        torch.save({"state_dict": model.state_dict(), "label_names": model.label_names}, ckpt_path)
        loaded = load_model(ckpt_path, embedding_dim=embedding_dim, num_labels=num_labels, device="cpu")
        with torch.no_grad():
            diff = (loaded(xB) - model(xB)).abs().max().item()
        print(f"\nload_model checkpoint round-trip max|diff|={diff:.6g}")

    print("\nOK — classifier.py smoke test passed.")


if __name__ == "__main__":
    main()

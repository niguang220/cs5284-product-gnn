"""Encode TAPE's GPT explanation of every node with the frozen sentence encoder (arms G/H).

Usage: python scripts/encode_tape.py
Needs outputs/tape_responses/ogbn-products.zip (see results/shared/tape_gpt_responses_manifest.json).
Writes the embeddings to outputs/embeddings/ (not in git; about 83 MB) and their manifest
(encoder, revision, shape, bytes, SHA-256) to results/shared/embeddings/.
"""
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products, load_tape_explanations  # noqa: E402
from src.encode import MODEL_NAME, MODEL_REVISION, encode  # noqa: E402
from src.text_clean import clean_text  # noqa: E402

NAME = "tape__all-MiniLM-L6-v2"

graph, _ = load_products()
texts = [clean_text(t) for t in load_tape_explanations(graph.num_nodes)]

t0 = time.time()
emb = encode(texts, batch_size=64, show_progress_bar=True)
minutes = (time.time() - t0) / 60

out = REPO / "outputs" / "embeddings" / f"{NAME}.npy"
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, emb)

meta = {
    "file": out.name,
    "text_variant": "tape_explanation",
    "encoder": MODEL_NAME,
    "encoder_revision": MODEL_REVISION,
    "shape": list(emb.shape),
    "dtype": str(emb.dtype),
    "bytes": out.stat().st_size,
    "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "created": date.today().isoformat(),
    "minutes_cpu": round(minutes, 1),
}
manifest = REPO / "results" / "shared" / "embeddings" / f"{NAME}.json"
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))

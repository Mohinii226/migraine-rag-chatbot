"""A tiny stand-in embedder so tests run instantly, offline, with no model download."""
import re
import zlib

import numpy as np


class HashingEmbedder:
    model_name = "hashing-test-embedder"

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


CHUNKS = [
    {"id": 0, "text": "XGBoost reached the highest accuracy for migraine subtype classification.", "source": "paper_a.pdf", "page": 1},
    {"id": 1, "text": "SMOTE was applied only inside training folds to handle class imbalance.", "source": "paper_a.pdf", "page": 2},
    {"id": 2, "text": "Sleep diary records were merged with wearable heart rate variability data.", "source": "paper_b.pdf", "page": 1},
]

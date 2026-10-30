"""Step 3 of RAG: turn text into embeddings (lists of numbers that capture meaning).

Texts with similar meaning get similar numbers, so we can search by meaning
instead of by exact keywords.
"""
import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name):
        # Imported here so the rest of the project (and the tests) work without it.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)  # downloads once, then cached

    def encode(self, texts, show_progress=False):
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype="float32")

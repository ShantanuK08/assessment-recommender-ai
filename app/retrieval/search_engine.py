import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SearchEngine:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = faiss.read_index(
            "data/embeddings/catalog.index"
        )

        with open(
            "data/embeddings/catalog_metadata.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.catalog = json.load(file)

    def search(self, query: str, top_k: int = 10):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for index in indices[0]:
            results.append(self.catalog[index])

        return results
"""
app/retrieval/search_engine.py

Single responsibility: retrieve products from the catalog.
    - No LLM calls.
    - No classification or decision-making.
    - Accepts a structured Search Context (dict) instead of a raw string:
      builds a semantic query from it, runs FAISS, then applies metadata
      filtering/re-ranking on top (hybrid retrieval).

NOTE ON ASSUMED CATALOG FIELDS:
    This assumes catalog_metadata.json entries look roughly like:
        {
          "name": ..., "description": ..., "url": ...,
          "test_type": ..., "keys": [...], "job_levels": [...],
          "remote": true/false, "adaptive": true/false,
          "languages": [...]
        }
    Adjust the field names in _apply_metadata_filters / _rank if your
    processed_catalog.json uses different keys — I don't have the actual
    file so these are best-guess based on your project summary.
"""

import json
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SearchEngine:

    def __init__(
        self,
        index_path: str = "data/embeddings/catalog.index",
        metadata_path: str = "data/embeddings/catalog_metadata.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.catalog = json.load(file)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def search(self, context: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        context: the Search Context dict produced by ConversationAnalyzer.
        """
        query_text = self._build_semantic_query(context)

        # Over-fetch so metadata filtering still leaves top_k good results.
        candidates = self._semantic_search(query_text, top_k=max(top_k * 4, 20))

        filtered = self._apply_metadata_filters(candidates, context)
        ranked = self._rank(filtered or candidates, context)

        return ranked[:top_k]

    def find_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """Direct lookup used for /compare — exact-ish match on product name."""
        lowered = [n.lower().strip() for n in names]
        matches = [
            product for product in self.catalog
            if product.get("name", "").lower().strip() in lowered
        ]

        # Fall back to a semantic search per name if an exact match is
        # missing, so slightly different phrasing of a real assessment
        # name still resolves to something.
        found_names = {m.get("name", "").lower().strip() for m in matches}
        for name, lowered_name in zip(names, lowered):
            if lowered_name not in found_names:
                matches.extend(self._semantic_search(name, top_k=1))

        return matches

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_semantic_query(self, context: Dict[str, Any]) -> str:
        parts: List[str] = []
        if context.get("role"):
            parts.append(context["role"])
        if context.get("job_level"):
            parts.append(context["job_level"])
        if context.get("experience"):
            parts.append(context["experience"])
        parts.extend(context.get("keys", []))
        parts.extend(context.get("assessment_preferences", []))
        return " ".join(parts) or "general hiring assessment"

    def _semantic_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [query_text], convert_to_numpy=True
        ).astype(np.float32)

        top_k = min(top_k, len(self.catalog))
        _, indices = self.index.search(query_embedding, top_k)

        return [self.catalog[i] for i in indices[0] if 0 <= i < len(self.catalog)]

    def _apply_metadata_filters(
        self, candidates: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        filtered = candidates

        if context.get("remote") is not None:
            narrowed = [p for p in filtered if p.get("remote") == context["remote"]]
            filtered = narrowed or filtered  # never filter down to nothing

        if context.get("adaptive") is not None:
            narrowed = [p for p in filtered if p.get("adaptive") == context["adaptive"]]
            filtered = narrowed or filtered

        if context.get("job_level"):
            level = context["job_level"].lower()
            narrowed = [
                p for p in filtered
                if level in [jl.lower() for jl in p.get("job_levels", [])]
            ]
            filtered = narrowed or filtered

        if context.get("languages"):
            wanted = {l.lower() for l in context["languages"]}
            narrowed = [
                p for p in filtered
                if wanted & {l.lower() for l in p.get("languages", [])}
            ]
            filtered = narrowed or filtered

        return filtered

    def _rank(
        self, candidates: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        wanted_keys = {k.lower() for k in context.get("keys", [])}
        if not wanted_keys:
            return candidates

        def score(product: Dict[str, Any]) -> int:
            product_keys = {k.lower() for k in product.get("keys", [])}
            return -len(wanted_keys & product_keys)  # more overlap = better rank

        return sorted(candidates, key=score)
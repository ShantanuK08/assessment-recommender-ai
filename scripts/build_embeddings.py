import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------------
# Paths
# -------------------------------

CATALOG_PATH = Path("data/processed/processed_catalog.json")

INDEX_PATH = Path("data/embeddings/catalog.index")
METADATA_PATH = Path("data/embeddings/catalog_metadata.json")

# -------------------------------
# Load Catalog
# -------------------------------


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# -------------------------------
# Main
# -------------------------------

def main():

    print("=" * 60)
    print("Loading processed catalog...")
    print("=" * 60)

    catalog = load_catalog()

    print(f"Products : {len(catalog)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    texts = [product["search_text"] for product in catalog]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(np.float32)

    print("\nBuilding FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=4, ensure_ascii=False)

    print("\nDone!")
    print(f"Vectors : {index.ntotal}")
    print(f"Dimension : {dimension}")
    print(f"Index saved : {INDEX_PATH}")
    print(f"Metadata saved : {METADATA_PATH}")


if __name__ == "__main__":
    main()
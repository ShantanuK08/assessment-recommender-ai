import json

with open("data/embeddings/catalog_metadata.json", encoding="utf-8") as f:
    catalog = json.load(f)

print(json.dumps(catalog[0], indent=4))

import json

with open("data/processed/processed_catalog.json", encoding="utf-8") as f:
    catalog = json.load(f)

for p in catalog:
    if p["name"] == "Java 8 (New)":
        print(json.dumps(p, indent=4))
        break
import json

with open("data/raw/shl_product_catalog.json", encoding="utf-8") as f:
    catalog = json.load(f)

print(catalog[0].keys())
print()
print(json.dumps(catalog[0], indent=2))
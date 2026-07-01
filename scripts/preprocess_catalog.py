import json
from pathlib import Path


RAW_CATALOG_PATH = Path("data/raw/shl_product_catalog.json")
PROCESSED_CATALOG_PATH = Path("data/processed/processed_catalog.json")


def load_catalog():
    """Load the original SHL catalog."""

    with open(RAW_CATALOG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_product(product):
    """Clean one product."""

    search_text = " ".join([
        product["name"],
        product["description"],
        " ".join(product["job_levels"]),
        " ".join(product["keys"]),
        product["duration"],
        " ".join(product["languages"])
    ])

    cleaned_product = {
        "id": product["entity_id"],
        "name": product["name"],
        "url": product["link"],
        "description": product["description"],
        "job_levels": product["job_levels"],
        "languages": product["languages"],
        "duration": product["duration"],
        "keys": product["keys"],
        "remote": product["remote"] == "yes",
        "adaptive": product["adaptive"] == "yes",
        "search_text": search_text
    }

    return cleaned_product

def main():

    catalog = load_catalog()

    processed_catalog = []

    for product in catalog:
        processed_catalog.append(clean_product(product))

    with open(PROCESSED_CATALOG_PATH, "w", encoding="utf-8") as file:
        json.dump(processed_catalog, file, indent=4)

    print("Catalog processed successfully.")
    print(f"Products Processed : {len(processed_catalog)}")


if __name__ == "__main__":
    main()
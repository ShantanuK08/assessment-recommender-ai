import json
from pathlib import Path


RAW_CATALOG_PATH = Path("data/raw/shl_product_catalog.json")
PROCESSED_CATALOG_PATH = Path("data/processed/processed_catalog.json")


# Derive a high-level test type from SHL's existing "keys" field.
# This is deterministic and does not invent information.
TEST_TYPE_MAPPING = {
    "Knowledge & Skills": "Technical",
    "Ability & Aptitude": "Ability",
    "Personality & Behavior": "Personality",
    "Assessment Exercises": "Simulation",
    "Biodata & Situational Judgment": "Situational Judgment",
    "Competencies": "Competency",
    "Development & 360": "Development",
}


def load_catalog():
    """Load the original SHL catalog."""

    with open(RAW_CATALOG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def derive_test_type(keys):
    """
    Derive a human-readable test type from the SHL 'keys' field.
    """

    if not keys:
        return "General"

    test_types = []

    for key in keys:
        mapped = TEST_TYPE_MAPPING.get(key)
        if mapped and mapped not in test_types:
            test_types.append(mapped)

    if not test_types:
        return "General"

    return ", ".join(test_types)


def clean_product(product):
    """Clean one product."""

    search_text = " ".join([
        product.get("name", ""),
        product.get("description", ""),
        " ".join(product.get("job_levels", [])),
        " ".join(product.get("keys", [])),
        product.get("duration", ""),
        " ".join(product.get("languages", []))
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

        # NEW FIELD
        "test_type": derive_test_type(product.get("keys", [])),

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
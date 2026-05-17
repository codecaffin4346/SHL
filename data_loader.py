import json
import os

def load_catalog():
    with open('shl_product_catalog.json', 'r', encoding='utf-8') as f:
        # Some JSON files have unescaped control characters. strict=False handles them.
        catalog = json.load(f, strict=False)
    
    # We need to filter "Individual Test Solutions"
    # Let's inspect the keys to understand where they are.
    return catalog

if __name__ == "__main__":
    catalog = load_catalog()
    if isinstance(catalog, dict):
        print("Keys:", catalog.keys())
        first_key = list(catalog.keys())[0]
        print(f"Sample of {first_key}:", json.dumps(catalog[first_key][:2] if isinstance(catalog[first_key], list) else catalog[first_key], indent=2))
    elif isinstance(catalog, list):
        print("List of length", len(catalog))
        print("Sample:", json.dumps(catalog[0], indent=2))

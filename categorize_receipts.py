import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


CATEGORIES = ["Produce", "Meat", "Dairy", "Dry Goods", "Beverages", "Other"]
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_receipts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("receipts file must contain a JSON array")
    return data


def save_receipts(data: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


HIGH_CONF_RULES = {
    "Produce": [
        "apple",
        "spinach",
        "kale",
        "pepper",
        "tomato",
        "avocado",
        "broccoli",
        "strawberr",
        "lettuce",
        "greens",
        "quinoa",
        "mango",
        "watermelon",
        "bunch",
    ],
    "Meat": [
        "chicken",
        "beef",
        "pork",
        "turkey",
        "salmon",
        "meat",
        "ground beef",
        "bacon",
        "sirloin",
        "thighs",
        "fillet",
    ],
    "Dairy": [
        "milk",
        "yogurt",
        "cheese",
        "mozzarella",
        "parmesan",
        "butter",
        "eggs",
        "cream",
    ],
    "Dry Goods": [
        "rice",
        "oats",
        "granola",
        "bread",
        "baguette",
        "beans",
        "oil",
        "pasta",
        "flour",
        "quinoa",
        "oat",
        "breading",
        "grain",
    ],
    "Beverages": [
        "water",
        "kombucha",
        "coffee",
        "espresso",
        "tea",
        "coconut water",
        "sparkling water",
        "juice",
        "ale",
        "soda",
    ],
}


def normalize_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[()]", " ", n)
    n = n.replace("-", " ")
    n = re.sub(r"\s+", " ", n).strip()
    return n


def high_conf_category(name: str) -> Optional[str]:
    for cat, keywords in HIGH_CONF_RULES.items():
        for kw in keywords:
            if kw in name:
                return cat
    return None


def build_llm():
    if AutoModelForCausalLM is None or AutoTokenizer is None or pipeline is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            do_sample=False,
        )
    except Exception:
        return None

#AI helped with this function
def batch_llm_categories(gen_pipeline, names: List[str]) -> List[Optional[str]]:
    if gen_pipeline is None or not names:
        return ["Other" for _ in names]

    def run_chunk(chunk: List[str]) -> List[Optional[str]]:
        numbered = "\n".join([f"{i+1}. {n}" for i, n in enumerate(chunk, start=1)])
        prompt = (
            "You are a classification engine.\n\n"
            "Task:\n"
            "Assign ONE category to EACH item below.\n\n"
            "Categories:\n"
            "Produce\n"
            "Meat\n"
            "Dairy\n"
            "Dry Goods\n"
            "Beverages\n"
            "Other\n\n"
            "Rules:\n"
            "- Choose the most appropriate category\n"
            "- Use restaurant purchasing logic\n"
            "- Household or cleaning items must be \"Other\"\n"
            "- If an item fits multiple categories, choose the one most commonly used in restaurant purchasing.\n"
            "- Respond with ONLY the category names\n"
            "- One category per line\n"
            "- Same order as the items\n"
            "- Do not explain\n"
            "- Do not repeat item names\n\n"
            "Items:\n"
            f"{numbered}\n\n"
            "Output:"
        )

        max_tokens = max(2 * len(chunk), 6)

        def attempt() -> List[Optional[str]]:
            try:
                out = gen_pipeline(
                    prompt,
                    pad_token_id=gen_pipeline.tokenizer.eos_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )
                text = out[0]["generated_text"]
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                results: List[Optional[str]] = []
                for ln in lines[: len(chunk)]:
                    match = next((cat for cat in CATEGORIES if ln.lower() == cat.lower()), None)
                    results.append(match)
                while len(results) < len(chunk):
                    results.append(None)
                return results[: len(chunk)]
            except Exception:
                return [None for _ in chunk]

        first = attempt()
        if any(r is None for r in first) or len(first) != len(chunk):
            second = attempt()
            merged: List[Optional[str]] = []
            for a, b in zip(first, second):
                merged.append(a or b)
            return [m or "Other" for m in merged]
        return [r or "Other" for r in first]

    results: List[Optional[str]] = []
    chunk_size = 16
    for start in range(0, len(names), chunk_size):
        chunk = names[start : start + chunk_size]
        results.extend(run_chunk(chunk))
    return results


def categorize_items(receipts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gen = build_llm()
    categorized = []
    for receipt in receipts:
        new_items = []
        items = receipt.get("items", [])
        normalized = [normalize_name(item.get("name", "")) for item in items]

        prelim_cats: List[Optional[str]] = []
        ambiguous_names: List[str] = []
        ambiguous_indices: List[int] = []

        for idx, norm in enumerate(normalized):
            cat = high_conf_category(norm)
            prelim_cats.append(cat)
            if cat is None:
                ambiguous_indices.append(idx)
                ambiguous_names.append(norm)

        batch_cats = batch_llm_categories(gen, ambiguous_names)

        for idx, item in enumerate(items):
            if idx in ambiguous_indices:
                batch_pos = ambiguous_indices.index(idx)
                llm_cat = batch_cats[batch_pos] if batch_pos < len(batch_cats) else None
                cat = llm_cat or "Other"
            else:
                cat = prelim_cats[idx] or "Other"
            new_items.append({"name": item.get("name", ""), "price": item.get("price", ""), "category": cat})
        categorized.append(
            {
                "supplier": receipt.get("supplier", ""),
                "date": receipt.get("date", ""),
                "items": new_items,
            }
        )
    return categorized


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "receipts_dataset.json"
    dest = sys.argv[2] if len(sys.argv) > 2 else "receipts_dataset_test.json"

    receipts = load_receipts(src)
    categorized = categorize_items(receipts)
    save_receipts(categorized, dest)
    print(f"Saved categorized receipts to {Path(dest).resolve()}")


if __name__ == "__main__":
    main()

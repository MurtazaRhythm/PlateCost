import json
import sys
import re
from datetime import datetime
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
            price_raw = item.get("price", "")
            price_float = parse_price(price_raw)
            new_items.append(
                {
                    "name": item.get("name", ""),
                    "price": price_raw,
                    "price_float": price_float,
                    "category": cat,
                }
            )
        categorized.append(
            {
                "supplier": receipt.get("supplier", ""),
                "date": receipt.get("date", ""),
                "items": new_items,
            }
        )
    return categorized


def parse_price(price: Any) -> float:
    if isinstance(price, (int, float)):
        return float(price)
    if not isinstance(price, str):
        return 0.0
    cleaned = price.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except Exception:
        return 0.0


def week_key(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    except Exception:
        return "unknown"


def aggregate_weekly(receipts: List[Dict[str, Any]]):
    weekly_category: Dict[str, Dict[str, float]] = {}
    weekly_supplier: Dict[str, Dict[str, float]] = {}
    week_totals: Dict[str, float] = {}

    for receipt in receipts:
        wk = week_key(receipt.get("date", ""))
        weekly_category.setdefault(wk, {})
        weekly_supplier.setdefault(wk, {})

        supplier = receipt.get("supplier", "unknown")
        for item in receipt.get("items", []):
            cat = item.get("category", "Other")
            amt = item.get("price_float", 0.0)
            weekly_category[wk][cat] = weekly_category[wk].get(cat, 0.0) + amt
            weekly_supplier[wk][supplier] = weekly_supplier[wk].get(supplier, 0.0) + amt
            week_totals[wk] = week_totals.get(wk, 0.0) + amt

    return weekly_category, weekly_supplier, week_totals


def compute_insights(weekly_category, weekly_supplier, week_totals):
    insights = []
    weeks_sorted = sorted(weekly_category.keys())

    # Category week-over-week
    for idx, wk in enumerate(weeks_sorted):
        if idx == 0:
            continue
        prev = weeks_sorted[idx - 1]
        for cat, val in weekly_category[wk].items():
            prev_val = weekly_category.get(prev, {}).get(cat, 0.0)
            delta = val - prev_val
            if prev_val == 0:
                change_pct = 100.0 if val > 0 else 0.0
            else:
                change_pct = (delta / prev_val) * 100.0
            if delta > 0:
                insights.append(
                    {
                        "type": "category_increase",
                        "category": cat,
                        "change_pct": round(change_pct, 2),
                        "change_absolute": round(delta, 2),
                        "week": wk,
                    }
                )
            elif delta < 0:
                insights.append(
                    {
                        "type": "category_decrease",
                        "category": cat,
                        "change_pct": round(change_pct, 2),
                        "change_absolute": round(delta, 2),
                        "week": wk,
                    }
                )
            if abs(change_pct) >= 50.0:
                insights.append(
                    {
                        "type": "spike_unusual",
                        "category": cat,
                        "change_pct": round(change_pct, 2),
                        "change_absolute": round(delta, 2),
                        "week": wk,
                    }
                )

    # Supplier concentration per week
    for wk, suppliers in weekly_supplier.items():
        total = week_totals.get(wk, 0.0)
        if total <= 0:
            continue
        for supplier, amt in suppliers.items():
            share = amt / total
            if share > 0.5:
                insights.append(
                    {
                        "type": "supplier_concentration",
                        "supplier": supplier,
                        "share": round(share * 100, 2),
                        "week": wk,
                    }
                )

    return insights


def build_output(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    weekly_category, weekly_supplier, week_totals = aggregate_weekly(receipts)
    insights = compute_insights(weekly_category, weekly_supplier, week_totals)

    weekly_category_list = [
        {"week": wk, "categories": {k: round(v, 2) for k, v in cats.items()}}
        for wk, cats in weekly_category.items()
    ]
    weekly_supplier_list = [
        {"week": wk, "suppliers": {k: round(v, 2) for k, v in sups.items()}}
        for wk, sups in weekly_supplier.items()
    ]

    return {
        "receipts": receipts,
        "weekly_category": weekly_category_list,
        "weekly_supplier": weekly_supplier_list,
        "insights": insights,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/receipts_ocr.json"
    dest = sys.argv[2] if len(sys.argv) > 2 else "data/receipts_dataset_categorized.json"

    receipts = load_receipts(src)
    categorized = categorize_items(receipts)
    output = build_output(categorized)
    save_receipts(output, dest)
    print(f"Saved categorized receipts to {Path(dest).resolve()}")


if __name__ == "__main__":
    main()

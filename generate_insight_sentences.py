import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 16


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def _prompt_for_batch(batch: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ins in enumerate(batch, start=1):
        parts = [f"{i})"]
        for k in ("type", "category", "supplier", "change_pct", "change_absolute", "week", "share"):
            if k in ins:
                parts.append(f"{k}:{ins.get(k)}")
        lines.append(" ".join(parts))
    insights_block = "\n".join(lines)
    return (
        "You are a concise analytics writer.\n"
        "Task: For each insight below, write exactly one short sentence using only its data.\n"
        "Rules:\n"
        "- One sentence per insight, same order\n"
        "- Use given numbers; include percent/absolute changes and supplier when provided\n"
        "- No extra commentary, no explanations, no bullet characters\n"
        "- Output must be plain English sentences only\n"
        "- Deterministic wording; avoid variation\n"
        "Insights:\n"
        f"{insights_block}\n"
        "Output:\n"
        "- Do not use <sentence for insight> tags\n"
        "1. <sentence for insight 1>\n"
        "2. <sentence for insight 2>\n"
        "... continue for all insights."
    )


def _fallback_sentence(ins: Dict[str, Any]) -> str:
    itype = ins.get("type", "")
    category = ins.get("category", "")
    week = ins.get("week", "")
    change_pct = ins.get("change_pct")
    change_abs = ins.get("change_absolute")
    supplier = ins.get("supplier", "")
    share = ins.get("share")

    if itype in ("category_increase", "category_decrease", "spike_unusual"):
        direction = "increased"
        if change_abs is not None and isinstance(change_abs, (int, float)) and change_abs < 0:
            direction = "decreased"
        if change_pct is None:
            return f"{category} spending changed this week vs last week."
        return f"{category} spending {direction} {change_pct}% this week vs last week."
    if itype == "supplier_concentration":
        if share is None:
            return f"Supplier {supplier} has concentrated spend this week."
        return f"Supplier {supplier} accounts for {share}% of spend this week."
    return "Insight summary unavailable."


def _parse_lines(text: str, count: int) -> List[Optional[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results: List[Optional[str]] = [None] * count
    for ln in lines:
        if "." in ln:
            prefix, rest = ln.split(".", 1)
            try:
                idx = int(prefix.strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < count:
                results[idx] = rest.strip()
    # If lines came back without numbering, try positional mapping
    if all(r is None for r in results):
        for i, ln in enumerate(lines[:count]):
            results[i] = ln
    return results


def llm_sentences(gen, insights: List[Dict[str, Any]]) -> List[str]:
    sentences: List[str] = []
    for start in range(0, len(insights), BATCH_SIZE):
        batch = insights[start : start + BATCH_SIZE]
        prompt = _prompt_for_batch(batch)
        max_tokens = max(5 * len(batch), 12)
        try:
            out = gen(
                prompt,
                pad_token_id=gen.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            text = out[0]["generated_text"]
            parsed = _parse_lines(text, len(batch))
        except Exception:
            parsed = [None] * len(batch)

        for ins, line in zip(batch, parsed):
            if not line or line.strip().startswith("<sentence for insight"):
                sentences.append(_fallback_sentence(ins))
            else:
                sentences.append(line)
    return sentences


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "receipts_dataset_test.json"
    dest = sys.argv[2] if len(sys.argv) > 2 else "receipts_dataset_with_insights.json"

    data = load_json(src)
    insights = data.get("insights", [])

    gen = build_llm()
    sentences = llm_sentences(gen, insights)

    for ins, sent in zip(insights, sentences):
        ins["ai_sentence"] = sent

    output = dict(data)
    output["insights"] = insights

    save_json(output, dest)
    print(f"Saved insight sentences to {Path(dest).resolve()}")


if __name__ == "__main__":
    main()

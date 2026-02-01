# OCR pipeline and receipt text parsing utilities for image uploads.
import cv2
import numpy as np
import easyocr
import sys
import re
import json
from pathlib import Path


# Prepare receipt image to improve OCR accuracy (denoise, contrast, threshold).
def preprocess_receipt(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    contrast = clahe.apply(denoised)

    thresh = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# Run EasyOCR on a preprocessed receipt image and normalize common patterns.
def run_ocr(img_path):
    reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
    img = preprocess_receipt(img_path)  # Preprocess the image
    result = reader.readtext(img)  # Run OCR on the preprocessed image
    fixed = []
    for (_, text, _) in result:
        # EasyOCR often reads '$' as 'S'; fix common money patterns like "S12.50"
        text = re.sub(r'\bS(?=\s?\d)', '$', text)
        # Normalize price formatting that may use a comma instead of a decimal, e.g., "$7,50" -> "$7.50"
        text = re.sub(r'(\$\d+),(\d{2}\b)', r'\1.\2', text)
        # Normalize compact dates like "0131/2026" -> "01/31/2026" and "01312026" -> "01/31/2026"
        text = re.sub(r'\b(\d{2})(\d{2})/(\d{4})\b', r'\1/\2/\3', text)
        text = re.sub(r'\b(\d{2})(\d{2})(\d{4})\b', r'\1/\2/\3', text)
        fixed.append(text)
    return fixed  # Return the text from the OCR result

#AI assisted with this function
# Parse OCR lines into supplier, date, and items with prices.
def parse_receipt(lines):
    price_re = re.compile(r'\$\s?\d+(?:[.,]\d{2})?')
    date_re = re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b')
    skip_keywords = ["subtotal", "subtal", "tax", "vat", "total"]

    supplier = None
    date = None
    items = []
    prev_non_price = None
    seen_items = set()  # avoid duplicate name/price pairs

    def normalize_price(raw):
        if not raw:
            return None
        cleaned = raw.replace(' ', '').replace(',', '.')
        if not cleaned.startswith('$'):
            cleaned = '$' + cleaned.lstrip('$')
        return cleaned

    for line in lines:
        if not line or line.isspace():
            continue

        # Capture date if we haven't already
        if date is None:
            m_date = date_re.search(line)
            if m_date:
                date = m_date.group(0)
                continue

        m_price = price_re.search(line)

        # Supplier: first meaningful line without price/date
        if supplier is None and m_price is None:
            supplier = line.strip()

        # Item + price on the same line
        if m_price and (line.strip() != m_price.group(0)):
            price = normalize_price(m_price.group(0))
            name = line.replace(m_price.group(0), '').strip(' :-')
            key = (name.lower(), price) if name else None
            if (
                name
                and not any(k in name.lower() for k in skip_keywords)
                and key not in seen_items
            ):
                items.append({"name": name, "price": price})
                seen_items.add(key)
                prev_non_price = None
                continue

        # Price on its own line: pair with previous non-price line
        if m_price and line.strip() == m_price.group(0):
            price = normalize_price(m_price.group(0))
            if (
                prev_non_price
                and not any(k in prev_non_price.lower() for k in skip_keywords)
            ):
                key = (prev_non_price.strip().lower(), price)
                if key not in seen_items:
                    items.append({"name": prev_non_price.strip(), "price": price})
                    seen_items.add(key)
            prev_non_price = None
            continue

        # Remember last non-price line to pair with following price
        if m_price is None:
            if not any(k in line.lower() for k in skip_keywords):
                prev_non_price = line
            else:
                prev_non_price = None

    return {"supplier": supplier, "date": date, "items": items}

#AI assisted with this function
# Save a single parsed receipt and merge it into a dataset file (deduping).
def save_receipt_json(data, dataset_path="receipts_dataset.json", compact=False):
    """
    Write the single receipt to output_path and append/merge into dataset_path.
    If compact is True, JSON is written without extra whitespace; otherwise it is pretty-printed.
    """
    json_kwargs = {"ensure_ascii": False}
    if compact:
        json_kwargs["separators"] = (",", ":")
    else:
        json_kwargs["indent"] = 2

    # Save single receipt
    with open("receipts_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, **json_kwargs)

    dataset_file = Path(dataset_path)
    existing = []
    if dataset_file.exists():
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    # Avoid duplicates (same supplier/date/items)
    if data not in existing:
        existing.append(data)
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, **json_kwargs)


if __name__ == "__main__":
    receipt = sys.argv[1]
    lines = run_ocr(receipt)
    parsed = parse_receipt(lines)

    # Export JSON outputs
    save_receipt_json(parsed)

    # Console preview
    print(f"Supplier: {parsed.get('supplier', '')}")
    print(f"Date: {parsed.get('date', '')}")
    print("Items:")
    for item in parsed.get("items", []):
        print(f" - {item['name']}: {item['price']}")

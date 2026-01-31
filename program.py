import cv2
import numpy as np
import easyocr
import sys
import re


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

#AI helped with this function
def parse_receipt(lines):
    price_re = re.compile(r'\$\s?\d+(?:[.,]\d{2})?')
    date_re = re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b')
    skip_keywords = ["subtotal", "subtal", "tax", "vat", "total"]

    supplier = None
    date = None
    items = []
    prev_non_price = None

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
            if name and not any(k in name.lower() for k in skip_keywords):
                items.append({"name": name, "price": price})
                prev_non_price = None
                continue

        # Price on its own line: pair with previous non-price line
        if m_price and line.strip() == m_price.group(0):
            price = normalize_price(m_price.group(0))
            if prev_non_price and not any(k in prev_non_price.lower() for k in skip_keywords):
                items.append({"name": prev_non_price.strip(), "price": price})
            prev_non_price = None
            continue

        # Remember last non-price line to pair with following price
        if m_price is None:
            if not any(k in line.lower() for k in skip_keywords):
                prev_non_price = line
            else:
                prev_non_price = None

    return {"supplier": supplier, "date": date, "items": items}


if __name__ == "__main__":
    receipt = sys.argv[1]
    lines = run_ocr(receipt)
    parsed = parse_receipt(lines)

    print(f"Supplier: {parsed.get('supplier', '')}")
    print(f"Date: {parsed.get('date', '')}")
    print("Items:")
    for item in parsed.get("items", []):
        print(f" - {item['name']}: {item['price']}")

import cv2
import numpy as np
import easyocr
import sys

def preprocess_receipt(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    contrast = clahe.apply(denoised)

    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

def run_ocr(img_path):
    reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
    img = preprocess_receipt(img_path)  # Preprocess the image
    result = reader.readtext(img)  # Run OCR on the preprocessed image
    return [text for (_, text, _) in result]  # Return the text from the OCR result
    

if __name__ == "__main__":
    receipt = sys.argv[1]
    lines = run_ocr(receipt)
    for line in lines:
        print(line)

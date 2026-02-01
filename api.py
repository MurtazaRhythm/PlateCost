import json
import uuid
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException

# Use program.py's OCR and parsing (EasyOCR + receipt parsing)
from program import run_ocr as run_ocr_image, parse_receipt
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RECEIPTS_FILE = DATA_DIR / "receipts.json"
OCR_FILE = DATA_DIR / "receipts_ocr.json"
ALLOWED_TYPES = {"image/png", "image/jpeg", "text/plain"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_json_receipt_txt(txt_path: Path) -> Dict[str, Any] | None:
    """
    Parse a .txt file containing JSON receipt data.
    Supports: single object {supplier, date, items} or array of such objects (uses first).
    """
    try:
        raw = json.loads(txt_path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and len(raw) > 0:
            obj = raw[0]
        elif isinstance(raw, dict):
            obj = raw
        else:
            return None
        if not isinstance(obj.get("items"), list):
            return None
        items = []
        for it in obj["items"]:
            if isinstance(it, dict) and "name" in it:
                items.append({
                    "name": str(it["name"]),
                    "price": str(it.get("price", "")),
                    "category": str(it.get("category", "")),
                })
        return {
            "supplier": obj.get("supplier"),
            "date": obj.get("date"),
            "items": items,
        }
    except Exception:
        return None


def run_ocr(image_path: Path) -> Dict[str, Any]:
    """
    Run OCR on receipt image using program.py (EasyOCR + parse_receipt).
    Returns supplier, date, and items with name/price.
    """
    try:
        lines = run_ocr_image(str(image_path))
        parsed = parse_receipt(lines)
        # Add category field for frontend (empty; can be filled by categorize_receipts later)
        items = [
            {"name": it["name"], "price": it["price"], "category": it.get("category", "")}
            for it in parsed.get("items", [])
        ]
        return {
            "supplier": parsed.get("supplier"),
            "date": parsed.get("date"),
            "items": items,
        }
    except Exception:
        return {
            "supplier": None,
            "date": None,
            "items": [],
        }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    rid = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    saved_path = DATA_DIR / f"{rid}{ext}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update receipts metadata
    receipts: List[Dict[str, Any]] = load_json(RECEIPTS_FILE, [])
    record = {
        "id": rid,
        "filename": saved_path.name,
        "uploaded_at": datetime.datetime.utcnow().isoformat(),
    }
    receipts.insert(0, record)
    save_json(RECEIPTS_FILE, receipts)

    # If image, run OCR and store results
    if file.content_type in {"image/png", "image/jpeg"}:
        parsed = run_ocr(saved_path)
        ocr_list: List[Dict[str, Any]] = load_json(OCR_FILE, [])
        ocr_entry = {
            "id": rid,
            "filename": saved_path.name,
            "supplier": parsed.get("supplier"),
            "date": parsed.get("date"),
            "items": parsed.get("items", []),
        }
        ocr_list.insert(0, ocr_entry)
        save_json(OCR_FILE, ocr_list)

    # If text file, parse as JSON receipt and store
    elif file.content_type == "text/plain" and ext == ".txt":
        parsed = _parse_json_receipt_txt(saved_path)
        if parsed:
            ocr_list = load_json(OCR_FILE, [])
            ocr_entry = {
                "id": rid,
                "filename": saved_path.name,
                "supplier": parsed.get("supplier"),
                "date": parsed.get("date"),
                "items": parsed.get("items", []),
            }
            ocr_list.insert(0, ocr_entry)
            save_json(OCR_FILE, ocr_list)

    return record


@app.get("/receipts")
def list_receipts():
    return load_json(RECEIPTS_FILE, [])


@app.delete("/receipts/{rid}")
def delete_receipt(rid: str):
    receipts: List[Dict[str, Any]] = load_json(RECEIPTS_FILE, [])
    receipts = [r for r in receipts if r.get("id") != rid]
    save_json(RECEIPTS_FILE, receipts)

    ocr_list: List[Dict[str, Any]] = load_json(OCR_FILE, [])
    ocr_list = [o for o in ocr_list if o.get("id") != rid]
    save_json(OCR_FILE, ocr_list)

    # remove stored files for this receipt
    for p in DATA_DIR.glob(f"{rid}.*"):
        try:
            p.unlink()
        except Exception:
            pass
    return {"deleted": rid}


@app.get("/receipts_ocr.json")
def get_ocr():
    return load_json(OCR_FILE, [])


@app.post("/receipts/{rid}/re-extract")
def re_extract_ocr(rid: str):
    """Re-run OCR on an already-uploaded receipt (image or JSON .txt)."""
    receipts = load_json(RECEIPTS_FILE, [])
    rec = next((r for r in receipts if r.get("id") == rid), None)
    if not rec:
        raise HTTPException(status_code=404, detail="Receipt not found")
    # Try image first
    img_path = next(
        (p for p in DATA_DIR.glob(f"{rid}.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        None,
    )
    txt_path = DATA_DIR / f"{rid}.txt"
    if img_path:
        parsed = run_ocr(img_path)
    elif txt_path.exists():
        parsed = _parse_json_receipt_txt(txt_path)
        if not parsed:
            raise HTTPException(status_code=400, detail="Could not parse JSON from text file")
    else:
        raise HTTPException(status_code=400, detail="No image or text file found for this receipt")
    ocr_list = load_json(OCR_FILE, [])
    ocr_list = [o for o in ocr_list if o.get("id") != rid]
    ocr_entry = {
        "id": rid,
        "filename": rec.get("filename", (img_path or txt_path).name),
        "supplier": parsed.get("supplier"),
        "date": parsed.get("date"),
        "items": parsed.get("items", []),
    }
    ocr_list.insert(0, ocr_entry)
    save_json(OCR_FILE, ocr_list)
    return ocr_entry


@app.get("/")
def serve_frontend():
    """Serve the frontend so upload, receipts, etc. work from one server."""
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")

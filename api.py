import json
import uuid
import shutil
import subprocess
import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


def run_ocr(image_path: Path) -> Dict[str, Any]:
    """
    Call program.py on an image and wrap stdout lines as items.
    """
    try:
        out = subprocess.check_output(
            ["python", "program.py", str(image_path)],
            text=True,
            timeout=60,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        items = [{"name": ln, "price": "", "category": ""} for ln in lines]
        return {
            "supplier": None,
            "date": None,
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

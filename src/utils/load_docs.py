# File: src/utils/load_docs.py
# Purpose: Convert files in the Database/ folder (PDF, DOCX, TXT/MD) into plain text files
# and write per-document metadata. Output lives in processed_docs/<Department>/
#
# Usage:
#   pip install -r requirements.txt
#   python src/utils/load_docs.py
#
# Output:
#   processed_docs/
#     <Department>/
#       my_file.txt
#       my_file.json   # metadata for the extracted text
#   processed_docs/metadata.csv  # master manifest
#
# NOTE: If PDFs are scanned images (no selectable text), you will need OCR (optional helper provided below).

import json
import uuid
import re
import csv
import datetime
from pathlib import Path
from typing import Dict, Optional, List

# third-party libs (optional where possible)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x


# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# We'll support both the Database folder and a CSVfiles folder if present
RAW_DIRS: List[Path] = [PROJECT_ROOT / "Database", PROJECT_ROOT / "CSVfiles", PROJECT_ROOT / "CSVfiles"]
PROCESSED_DIR = PROJECT_ROOT / "processed_docs"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz). Returns concatenated page text."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required to extract PDF text. Install via: pip install PyMuPDF")
    all_text = []
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            all_text.append(text)
    return "\n".join(all_text)


def extract_text_from_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx (if installed)."""
    if DocxDocument is None:
        raise ImportError("python-docx not installed. Install with: pip install python-docx")
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_csv(path: Path, max_rows: int = 1000) -> str:
    """Extract a human readable summary from a CSV using pandas (if available)."""
    if pd is None:
        raise ImportError("pandas is required to extract CSV/Excel files. Install via: pip install pandas")

    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, nrows=max_rows)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"Could not read CSV with common encodings: {path}")

    parts = []
    parts.append(f"CSV File: {path.name}")
    parts.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    parts.append("")
    parts.append("COLUMNS:")
    for i, col in enumerate(df.columns):
        parts.append(f"  {i+1}. {col}")
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No data"
        dtype = str(df[col].dtype)
        parts.append(f"     Type: {dtype}, Sample: {str(sample_val)[:100]}")
    parts.append("")
    parts.append("SAMPLE DATA:")
    for idx, row in df.head(5).iterrows():
        parts.append(f"Row {idx + 1}:")
        for col in df.columns:
            value = str(row[col]) if pd.notna(row[col]) else "NULL"
            if len(value) > 200:
                value = value[:197] + "..."
            parts.append(f"  {col}: {value}")
        parts.append("")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        parts.append("NUMERIC COLUMN STATISTICS:")
        stats = df[numeric_cols].describe()
        for col in numeric_cols:
            parts.append(f"  {col}:")
            parts.append(f"    Count: {stats[col]['count']:.0f}")
            parts.append(f"    Mean: {stats[col]['mean']:.2f}")
            parts.append(f"    Std: {stats[col]['std']:.2f}")
            parts.append(f"    Min: {stats[col]['min']:.2f}")
            parts.append(f"    Max: {stats[col]['max']:.2f}")
        parts.append("")

    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    parts.append("DATA QUALITY:")
    parts.append(f"  Total cells: {total_cells}")
    parts.append(f"  Null cells: {null_cells} ({(null_cells/total_cells*100) if total_cells else 0:.1f}%)")

    return "\n".join(parts)


def extract_text_from_excel(path: Path, max_rows: int = 1000) -> str:
    """Extract a readable summary from Excel file using pandas."""
    if pd is None:
        raise ImportError("pandas is required to extract CSV/Excel files. Install via: pip install pandas")

    excel_file = pd.ExcelFile(path)
    parts = []
    parts.append(f"Excel File: {path.name}")
    parts.append(f"Sheets: {', '.join(excel_file.sheet_names)}")
    parts.append("")
    for sheet_name in excel_file.sheet_names:
        parts.append(f"SHEET: {sheet_name}")
        df = pd.read_excel(path, sheet_name=sheet_name, nrows=min(1000, max_rows))
        parts.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        parts.append("Columns: " + ", ".join(df.columns.astype(str)))
        parts.append("")
        parts.append("SAMPLE DATA:")
        for idx, row in df.head(3).iterrows():
            parts.append(f"Row {idx + 1}: {dict(row.dropna())}")
        parts.append("")
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    """Clean up whitespace, unify newlines, remove repeated blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def save_processed_text_and_metadata(text: str, original_path: Path, department: str) -> Optional[Dict]:
    """Save text and metadata into processed_docs/<department>/ and return metadata dict."""
    if not text:
        return None

    out_dir = PROCESSED_DIR / department
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = original_path.stem
    txt_path = out_dir / f"{stem}.txt"
    meta_path = out_dir / f"{stem}.json"

    txt_path.write_text(text, encoding="utf-8")

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except Exception:
            return str(p)

    metadata = {
        "doc_id": str(uuid.uuid4()),
        "title": stem,
        "department": department,
        "original_path": rel(original_path),
        "processed_text_path": rel(txt_path),
        "processed_meta_path": rel(meta_path),
        "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "original_size_bytes": original_path.stat().st_size,
        "original_modified": datetime.datetime.fromtimestamp(original_path.stat().st_mtime).isoformat(),
        "file_type": original_path.suffix.lower()
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def process_file(path: Path, department: str) -> Optional[Dict]:
    """Detect type and extract text. Returns metadata if successful."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            text = extract_text_from_pdf(path)
        elif ext in (".docx", ".doc"):
            text = extract_text_from_docx(path)
        elif ext in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".csv":
            text = extract_text_from_csv(path)
        elif ext in (".xlsx", ".xls"):
            text = extract_text_from_excel(path)
        else:
            print(f"[skip] unsupported file type: {path}")
            return None
    except Exception as e:
        print(f"[error] failed to extract {path}: {e}")
        return None

    text = normalize_text(text)
    if not text:
        print(f"[warning] no text extracted from {path}")
        return None

    return save_processed_text_and_metadata(text, path, department)


def process_single_file(file_path: str, department: Optional[str] = None) -> Optional[Dict]:
    """Process a single file and write metadata.
    
    Args:
        file_path: Path to the file to process
        department: Department name (defaults to 'Database' if not provided)
    
    Returns:
        Metadata dict if successful, None otherwise
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"[error] File not found: {path}")
        return None
    
    if department is None:
        department = path.parent.name
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta = process_file(path, department)
    
    if meta:
        # write or update metadata CSV
        all_meta = []
        if METADATA_CSV.exists():
            with open(METADATA_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_meta = list(reader)
        
        # remove any existing entry with same doc_id to avoid duplicates
        all_meta = [m for m in all_meta if m.get("original_path") != meta["original_path"]]
        all_meta.append(meta)
        
        with open(METADATA_CSV, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=[
                "doc_id", "title", "department", "original_path", "processed_text_path",
                "processed_meta_path", "processed_at", "original_size_bytes", "original_modified", "file_type"
            ])
            writer.writeheader()
            for m in all_meta:
                writer.writerow({k: m.get(k, "") for k in writer.fieldnames})
        print(f"Processed 1 document. Manifest updated: {METADATA_CSV}")
    
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a single file into processed_docs and update manifest.")
    parser.add_argument("--file", "-f", help="Path to the file to process (required)")
    parser.add_argument("--department", "-d", default=None, help="Department name to store the processed file under (default: parent folder name or 'Database')")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed output dir: {PROCESSED_DIR}\n")

    if not args.file:
        parser.print_help()
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
        else:
            dept = args.department if args.department is not None else file_path.parent.name or "Database"
            print(f"Processing file: {file_path} as department: {dept}\n")
            meta = process_single_file(str(file_path), department=dept)
            if meta:
                print(f"\n✓ Successfully processed:")
                print(f"  Title: {meta['title']}")
                print(f"  Department: {meta['department']}")
                print(f"  Output: {meta['processed_text_path']}")
            else:
                print(f"\n✗ Failed to process the file.")

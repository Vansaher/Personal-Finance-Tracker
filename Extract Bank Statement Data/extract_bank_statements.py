#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from datetime import datetime
import pdfplumber
import pandas as pd

DATE_ROW_RE = re.compile(
    r'^(?P<date>\d{2}/\d{2})\s+(?P<mid>.+?)\s+'
    r'(?P<amount>\-?\d{1,3}(?:,\d{3})*\.\d{2}[+-])\s+'
    r'(?P<balance>\-?\d{1,3}(?:,\d{3})*\.\d{2})\s*$'
)

SKIP_KEYWORDS = [
    "URUSNIAGA AKAUN", "ACCOUNT TRANSACTIONS", "ENTRY DATE", "VALUE DATE",
    "TRANSACTION DESCRIPTION", "TRANSACTION AMOUNT", "STATEMENT DATE",
    "ACCOUNT NUMBER", "Maybank Islamic Berhad", "Perhatian / Note",
    "BEGINNING BALANCE", "ENDING BALANCE", "LEDGER BALANCE",
    "TOTAL DEBIT", "TOTAL CREDIT", "MUKA", "NOT PROTECTED BY PIDM",
    "BAKI LEGAR", "IBS ", "MR / ENCIK", "SELANGOR", "FCN",
    "PLEASE BE REMINDED", "NOTICE:", "KINDLY BE INFORMED",
]

def looks_like_header(line: str) -> bool:
    up = line.upper()
    return any(k in up for k in SKIP_KEYWORDS)

def parse_pdf(path: Path, default_year: int | None = None) -> pd.DataFrame:
    with pdfplumber.open(str(path)) as pdf:
        all_lines = []
        for page in pdf.pages:
            txt = page.extract_text() or ""
            all_lines.extend([l.rstrip() for l in txt.splitlines() if l.strip()])

    # try infer year from “STATEMENT DATE : dd/mm/yy”
    inferred_year = default_year
    for line in all_lines:
        if "STATEMENT DATE" in line.upper():
            m = re.search(r'(\d{2})/(\d{2})/(\d{2,4})', line)
            if m:
                y = m.group(3)
                inferred_year = int("20"+y if len(y) == 2 else y)
            break

    rows = []
    current = None

    def flush_current():
        nonlocal current
        if not current:
            return
        desc = current.get("mid", "").strip()
        cont = current.get("cont", [])
        if cont:
            parts = [desc] if desc else []
            parts.extend([c.strip() for c in cont if c.strip()])
            desc = " | ".join(parts)

        amt_tok = current["amount"]
        sign = -1.0 if amt_tok.endswith("-") else 1.0
        amount = float(amt_tok[:-1].replace(",", "")) * sign
        balance = float(current["balance"].replace(",", ""))

        dd, mm = current["date"].split("/")
        year = current.get("year") or inferred_year or datetime.now().year
        try:
            dt = datetime(int(year), int(mm), int(dd))
        except ValueError:
            dt = None

        rows.append({
            "date": dt,
            "date_str": current["date"],
            "description": desc,
            "amount": amount,
            "balance": balance,
            "source_file": path.name,
        })
        current = None

    for line in all_lines:
        if looks_like_header(line):
            continue
        m = DATE_ROW_RE.match(line)
        if m:
            flush_current()
            gd = m.groupdict()
            current = {
                "date": gd["date"],
                "year": inferred_year,
                "mid": gd["mid"].strip(),
                "amount": gd["amount"],
                "balance": gd["balance"],
                "cont": [],
            }
        else:
            if current and (re.match(r'^\s{2,}\S', line) or "*" in line or re.match(r'^[A-Z0-9].*\*\s*$', line)):
                if not looks_like_header(line):
                    current["cont"].append(line.strip())
    flush_current()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["description"] = (
        df["description"].fillna("")
        .str.replace(r'\s*\|\s*\|\s*', ' | ', regex=True)
        .str.strip(" |")
    )
    return df.sort_values(["date", "amount"], ascending=[True, True]).reset_index(drop=True)

def safe_sheet_name(name: str) -> str:
    # Excel sheet name: max 31 chars, no []:*?/\
    bad = '[]:*?/\\'
    cleaned = "".join(ch for ch in name if ch not in bad)
    return cleaned[:31] if cleaned else "Sheet"

def main():
    ap = argparse.ArgumentParser(description="Extract Maybank statements (PDF) → Excel with per-file sheets + Combined")
    ap.add_argument("pdfs", nargs="+", help="Input PDF files")
    ap.add_argument("-o", "--output", default="Finance_Tracker.xlsx", help="Output .xlsx path")
    ap.add_argument("--year", type=int, default=None, help="Force year for DD/MM dates (optional)")
    args = ap.parse_args()

    per_file_frames = []
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for pdf in args.pdfs:
            p = Path(pdf)
            if not p.exists():
                print(f"[WARN] Skipping missing: {p}")
                continue
            df = parse_pdf(p, default_year=args.year)
            if df.empty:
                print(f"[WARN] No rows parsed for: {p.name}")
                continue
            per_file_frames.append(df)
            sheet = safe_sheet_name(p.stem)
            df.to_excel(writer, sheet_name=sheet, index=False)

        if per_file_frames:
            combined = pd.concat(per_file_frames, ignore_index=True)
            combined = combined.sort_values(["date", "source_file", "amount"]).reset_index(drop=True)
            combined.to_excel(writer, sheet_name="Combined", index=False)

    total_rows = sum(len(df) for df in per_file_frames)
    print(f"[OK] Wrote {total_rows} rows across {len(per_file_frames)} sheet(s) + Combined → {args.output}")

if __name__ == "__main__":
    main()

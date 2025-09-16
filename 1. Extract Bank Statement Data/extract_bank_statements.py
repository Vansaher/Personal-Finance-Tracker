#!/usr/bin/env python3
"""
Maybank PDF → Excel (per-file sheets, Combined, Monthly Summary)
- Preserves statement order using a running sequence (seq)
- Handles multi-line descriptions
- Produces sheets: <each PDF>, Combined, Monthly Summary
# Steps
# Put all PDFs in one folder
# py extract_bank_statements_to_excel.py "Statement_June 2025.pdf" "Statement_July 2025.pdf" -o Finance_Tracker.xlsx -v
# Makes sure files are sorted by month
"""

import re
import argparse
from pathlib import Path
from datetime import datetime
import pdfplumber
import pandas as pd

# -------- Regex & constants --------
DATE_ROW_RE = re.compile(
    r'^(?P<date>\d{2}/\d{2})\s+(?P<mid>.+?)\s+'
    r'(?P<amount>-?\d{1,3}(?:,\d{3})*\.\d{2}[+-])\s+'
    r'(?P<balance>-?\d{1,3}(?:,\d{3})*\.\d{2})\s*$'
)

SKIP = [
    "URUSNIAGA AKAUN", "ACCOUNT TRANSACTIONS", "ENTRY DATE", "VALUE DATE",
    "TRANSACTION DESCRIPTION", "TRANSACTION AMOUNT", "STATEMENT DATE",
    "ACCOUNT NUMBER", "Maybank Islamic Berhad", "Perhatian / Note",
    "BEGINNING BALANCE", "ENDING BALANCE", "LEDGER BALANCE",
    "TOTAL DEBIT", "TOTAL CREDIT", "MUKA", "NOT PROTECTED BY PIDM",
    "BAKI LEGAR", "IBS ", "MR / ENCIK", "SELANGOR", "FCN",
    "PLEASE BE REMINDED", "NOTICE:", "KINDLY BE INFORMED",
]

def is_skip(line: str) -> bool:
    u = line.upper()
    return any(k in u for k in SKIP)

def safe_sheet_name(name: str) -> str:
    # Excel sheet name: max 31 chars, forbidden: []:*?/\
    bad = '[]:*?/\\'
    cleaned = "".join(ch for ch in name if ch not in bad)
    return (cleaned or "Sheet")[:31]

# -------- Core parser (preserves order) --------
def parse_pdf(path: Path, year_hint: int | None = None, verbose: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    seq, page, line, date, date_str, description, amount, balance, source_file
    """
    if verbose: print(f"[INFO] Opening: {path}")
    with pdfplumber.open(str(path)) as pdf:
        raw = []  # (page_no, line_idx, text)
        for i, p in enumerate(pdf.pages, 1):
            txt = p.extract_text() or ""
            lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
            if verbose: print(f"  - Page {i}: {len(lines)} lines")
            raw.extend((i, j, lines[j]) for j in range(len(lines)))

    # Infer year from "STATEMENT DATE : dd/mm/yy"
    inferred_year = year_hint
    for _, _, line in raw:
        if "STATEMENT DATE" in line.upper():
            m = re.search(r'(\d{2})/(\d{2})/(\d{2,4})', line)
            if m:
                y = m.group(3)
                inferred_year = int("20" + y) if len(y) == 2 else int(y)
            break

    rows = []
    cur = None
    seq = 0

    def flush():
        nonlocal cur
        if not cur:
            return
        # Build description
        desc = (cur.get("mid") or "").strip()
        cont = cur.get("cont", [])
        if cont:
            parts = ([desc] if desc else []) + [c.strip() for c in cont if c.strip()]
            desc = " | ".join(parts)

        # Amount (last char + or - controls sign)
        amt_tok = cur["amount"]
        sign = -1.0 if amt_tok.endswith("-") else 1.0
        amount = float(amt_tok[:-1].replace(",", "")) * sign
        balance = float(cur["balance"].replace(",", ""))

        # Date
        dd, mm = cur["date"].split("/")
        year = cur.get("year") or inferred_year or datetime.now().year
        try:
            dt = datetime(int(year), int(mm), int(dd))
        except ValueError:
            dt = None

        rows.append({
            "seq": cur["seq"],                 # original order
            "page": cur["page"],
            "line": cur["line"],
            "date": dt,
            "date_str": cur["date"],
            "description": desc,
            "amount": amount,
            "balance": balance,
            "source_file": path.name,
        })
        cur = None

    # Parse lines in natural page/line order, increment seq per new transaction row
    for page_no, line_idx, line in raw:
        if is_skip(line):
            continue
        m = DATE_ROW_RE.match(line)
        if m:
            flush()
            seq += 1
            gd = m.groupdict()
            cur = {
                "seq": seq,
                "page": page_no,
                "line": line_idx,
                "date": gd["date"],
                "year": inferred_year,
                "mid": gd["mid"].strip(),
                "amount": gd["amount"],
                "balance": gd["balance"],
                "cont": [],
            }
        else:
            # Continuation lines (merchant/location) — keep with current tx
            if cur and (re.match(r'^\s{2,}\S', line) or "*" in line or re.match(r'^[A-Z0-9].*\*\s*$', line)):
                if not is_skip(line):
                    cur["cont"].append(line.strip())

    flush()

    df = pd.DataFrame(rows)
    if df.empty:
        if verbose: print(f"[WARN] Parsed 0 rows from {path.name}")
        return df

    # Tidy description; DO NOT resort (keep seq)
    df["description"] = (
        df["description"].fillna("")
        .str.replace(r'\s*\|\s*\|\s*', ' | ', regex=True)
        .str.strip(" |")
    )
    df = df.sort_values(["seq"]).reset_index(drop=True)
    if verbose:
        dmin = df["date"].min()
        dmax = df["date"].max()
        print(f"[OK] {path.name}: {len(df)} rows | {dmin} → {dmax}")
    return df

# -------- Monthly summary from Combined --------
def monthly_summary(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return combined
    tmp = combined.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    agg = tmp.groupby("month").agg(
        income=("amount", lambda s: s[s > 0].sum()),
        expense=("amount", lambda s: s[s < 0].sum()),
    ).reset_index()
    agg["net"] = agg["income"] + agg["expense"]
    # Optional: running net savings over time
    agg["cumulative_net"] = agg["net"].cumsum()
    return agg

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Extract Maybank statements (PDF) → Excel with per-file sheets, Combined, Monthly Summary")
    ap.add_argument("pdfs", nargs="+", help="PDF paths or globs (e.g., *.pdf)")
    ap.add_argument("-o", "--output", default="Finance_Tracker.xlsx", help="Output .xlsx path")
    ap.add_argument("--year", type=int, default=None, help="Force year for DD/MM dates (optional)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    # Expand globs so you know exactly which files are parsed
    files: list[Path] = []
    for pat in args.pdfs:
        if any(c in pat for c in "*?[]"):
            files.extend(sorted(Path().glob(pat)))
        else:
            files.append(Path(pat))
    files = [f for f in files if f.exists()]

    if args.verbose:
        print("[INFO] Files to parse:")
        for f in files:
            print("  -", f)
    if not files:
        print("[ERR] No input files matched. Check names/paths.")
        return

    frames: list[pd.DataFrame] = []
    with pd.ExcelWriter(args.output, engine="openpyxl") as xw:
        # Per-file sheets (keep original order via seq)
        for f in files:
            try:
                df = parse_pdf(f, year_hint=args.year, verbose=args.verbose)
            except Exception as e:
                print(f"[ERR] {f.name}: {e}")
                continue
            if df.empty:
                continue
            frames.append(df)
            df.to_excel(xw, sheet_name=safe_sheet_name(f.stem), index=False)

        # Combined + Monthly Summary
        if frames:
            combined = pd.concat(frames, ignore_index=True)

            # Add a month column
            combined["month"] = combined["date"].dt.to_period("M").astype(str)

            # Sort by month then by seq (original within-month order)
            combined = combined.sort_values(["month", "seq"]).reset_index(drop=True)

            combined.to_excel(xw, sheet_name="Combined", index=False)

            summary = monthly_summary(combined)
            summary.to_excel(xw, sheet_name="Monthly Summary", index=False)


    total_rows = sum(len(df) for df in frames)
    if total_rows == 0:
        print("[INFO] Finished but parsed 0 rows (layout mismatch or wrong files?).")
    else:
        dmin = min(df["date"].min() for df in frames)
        dmax = max(df["date"].max() for df in frames)
        print(f"[DONE] Wrote {total_rows} rows from {len(frames)} file(s) → {args.output}")
        print(f"       Date range: {dmin} → {dmax}")

if __name__ == "__main__":
    main()

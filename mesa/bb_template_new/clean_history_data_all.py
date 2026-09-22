#!/usr/bin/env python3
"""
Clean MESA history.data files:
- Find all files named 'history.data' recursively under a root directory.
- Parse columns from the header line containing 'model_number'.
- Drop malformed data lines (wrong number of columns).
- Convert numeric columns.
- Deduplicate by model_number keeping the latest snapshot (by star_age_sec, then star_age).
- Sort by model_number (or by time if you prefer; see --sort-by).
- Write a cleaned file next to the original (history.cleaned.data) or overwrite with --inplace.

Usage:
  python clean_mesa_history.py /path/to/root [--inplace] [--sort-by model_number|star_age_sec|star_age] [--workers 0]

Notes:
- Works even if header lines above the column line vary across runs.
- If star_age_sec is missing, falls back to star_age; otherwise uses file order.
"""

import argparse
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

def find_history_files(root: Path):
    # any depth; if you truly want to restrict to 4 layers, change to rglob with pattern depth
    return list(root.rglob("history.data"))

def parse_history(path: Path):
    """Return (names, rows) where names is list of column names and rows is list[list[str]] of tokens."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Find the header line that lists column names (contains 'model_number')
    header_idx = None
    for i, ln in enumerate(lines[:200]):  # headers are near the top; guard against massive scans
        if "model_number" in ln and not ln.strip().startswith("#"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find a column header containing 'model_number' in {path}")

    names = lines[header_idx].split()
    ncols = len(names)

    # Data lines are after header; keep only rows with exactly ncols fields
    raw_rows = []
    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue
        toks = ln.split()
        if len(toks) == ncols:
            raw_rows.append(toks)
        else:
            # skip malformed rows
            continue

    if not raw_rows:
        raise ValueError(f"No valid data rows found in {path}")

    return names, raw_rows, header_idx

def to_numeric_df(names, rows):
    df = pd.DataFrame(rows, columns=names)
    # Convert columns to numeric when possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def clean_df(df: pd.DataFrame, sort_by: str):
    # Ensure model_number exists and numeric
    if "model_number" not in df.columns:
        raise ValueError("Missing 'model_number' column")
    df = df.copy()

    # Track order if needed
    df["_row_idx"] = range(len(df))

    # Choose time priority for deduplication
    time_cols = [c for c in ("star_age_sec", "star_age") if c in df.columns]
    dedupe_sort = ["model_number"] + (time_cols if time_cols else ["_row_idx"])

    df = df.sort_values(dedupe_sort).drop_duplicates(subset=["model_number"], keep="last")

    # Final sort for output
    if sort_by not in df.columns and sort_by != "model_number":
        # fallback chain
        for alt in ("star_age_sec", "star_age", "model_number"):
            if alt in df.columns:
                sort_by = alt
                break
        else:
            sort_by = "model_number"

    df = df.sort_values(sort_by).reset_index(drop=True)
    df.drop(columns=["_row_idx"], inplace=True, errors="ignore")
    return df

def write_cleaned(path: Path, df: pd.DataFrame, header_idx: int, inplace: bool):
    # Reconstruct header (preserve original header lines up to and including column line)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    header = lines[: header_idx + 1]  # includes the column names line

    # Format rows with same whitespace convention (space-separated)
    body_lines = [" ".join(str(v) for v in row) for row in df.to_numpy()]
    out_text = "\n".join(header + body_lines) + "\n"

    if inplace:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.write_text(out_text, encoding="utf-8")
        return str(path), str(backup)
    else:
        out_path = path.with_name("history.cleaned.data")
        out_path.write_text(out_text, encoding="utf-8")
        return str(out_path), None

def process_file(path: Path, sort_by: str, inplace: bool):
    try:
        names, rows, header_idx = parse_history(path)
        ncols = len(names)
        total_rows = len(rows)

        # filter rows strictly matching ncols already done in parse_history
        df = to_numeric_df(names, rows)

        # Count before cleaning
        before = len(df)

        df_clean = clean_df(df, sort_by=sort_by)
        after = len(df_clean)

        out_path, backup_path = write_cleaned(path, df_clean, header_idx, inplace=inplace)

        return {
            "file": str(path),
            "out_file": out_path,
            "backup": backup_path,
            "columns": ncols,
            "rows_in": total_rows,
            "valid_in": before,
            "rows_out": after,
            "dropped": before - after,
        }
    except Exception as e:
        return {"file": str(path), "error": repr(e)}

def main():
    ap = argparse.ArgumentParser(description="Fix shuffling/interleaving in MESA history.data files.")
    ap.add_argument("root", type=str, help="Root directory to scan")
    ap.add_argument("--inplace", action="store_true", help="Overwrite history.data (backup to history.data.bak)")
    ap.add_argument("--sort-by", choices=["model_number", "star_age_sec", "star_age"], default="model_number",
                    help="Final sort key for cleaned output (default: model_number)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0/1 = no parallelism).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    files = find_history_files(root)
    if not files:
        print("No history.data files found.")
        return

    print(f"Found {len(files)} history.data files under {root}")

    results = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_file, p, args.sort_by, args.inplace) for p in files]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for p in files:
            results.append(process_file(p, args.sort_by, args.inplace))

    # Report
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    for r in ok:
        print(f"[OK] {r['file']} → {r['out_file']}  "
              f"(cols={r['columns']} rows_in={r['rows_in']} valid_in={r['valid_in']} rows_out={r['rows_out']} dropped={r['dropped']})")
        if args.inplace and r["backup"]:
            print(f"     backup: {r['backup']}")

    for r in bad:
        print(f"[ERR] {r['file']} :: {r['error']}", file=sys.stderr)

    print(f"\nDone. Cleaned: {len(ok)} files. Errors: {len(bad)}.")

if __name__ == "__main__":
    main()

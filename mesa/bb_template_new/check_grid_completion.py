#!/usr/bin/env python3
"""Scan every L3 model in the active grid, classify by run status.

Walks the Z x M x P grid structure (z_div_zsun_grid, mass_grid, period_grid),
reads each model's .log file, and classifies it as COMPLETED, CRASHED,
RUNNING, PENDING, or STALLED based on pattern matching.

Outputs:
  grid_status_report.txt      summary + reason tally + per-model detail
  status.csv                  machine-readable per-model table
  failed_models_to_delete.txt STALLED + PENDING model paths
  managed_crashed_models.txt  CRASHED model paths
  running_models.txt          RUNNING model paths
"""

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ZSUN = 0.017
DEFAULT_STALE_MINUTES = 15

# ---------------------------------------------------------------------------
# Fortran-d formatting (matches the bash scripts' printf + sed convention)
# ---------------------------------------------------------------------------

def fortran_d(val, fmt):
    """Format a float as Fortran-d notation. fmt is a Python format spec."""
    return format(val, fmt).replace("e", "d")


def read_lines(path):
    """Read non-empty lines from a grid file."""
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Grid walking
# ---------------------------------------------------------------------------

def walk_grid(root):
    """Yield (l1_name, l2_name, l3_path, log_path) for every L3 model on disk.

    Discovers L3 dirs by globbing the filesystem rather than recomputing
    omega strings (bc's scale=8 truncation produces different last digits
    from Python's full-precision math.pi, making computed paths unreliable).
    Only walks active L1s (those not in z_ignore_grid).
    """
    z_grid = read_lines(root / "z_div_zsun_grid")
    z_ignore = set()
    if (root / "z_ignore_grid").exists():
        z_ignore = set(read_lines(root / "z_ignore_grid"))

    physics_id = root.name.split("_")[0]

    active_l1s = []
    for cnt_z, z_div_zsun_str in enumerate(z_grid):
        if z_div_zsun_str in z_ignore:
            continue
        z_suffix = fortran_d(float(z_div_zsun_str), ".0e")
        active_l1s.append(f"{physics_id}{cnt_z:02d}_ZdivZsun_{z_suffix}")

    for l1 in sorted(active_l1s):
        l1_path = root / l1
        if not l1_path.is_dir():
            continue
        for l2_path in sorted(l1_path.glob("[0-9][0-9][0-9]_m*")):
            if not l2_path.is_dir():
                continue
            for l3_path in sorted(l2_path.glob("m*_p*_w*")):
                if not l3_path.is_dir():
                    continue
                l3_rel = f"{l1}/{l2_path.name}/{l3_path.name}"
                log_rel = f"{l3_rel}.log"
                yield l1, l2_path.name, l3_rel, log_rel


# ---------------------------------------------------------------------------
# Classification patterns (ported from gwtc4 check_status.py)
# ---------------------------------------------------------------------------

TERM_PATTERNS = [
    re.compile(r"termination code:\s*[^\n]+", re.IGNORECASE),
    re.compile(r"stop because [^\n]+", re.IGNORECASE),
]

NUMERICAL_FAILURES = {"min_timestep_limit", "max_model_number", "adjust_mesh_failed"}

CRASH_PATTERN = re.compile(r"stopping because of problems[^\n]*", re.IGNORECASE)

FAIL_PATTERNS = [
    re.compile(r"forrtl: severe[^\n]*"),
    re.compile(r"^\*\*\* Error in[^\n]*", re.MULTILINE),
    re.compile(r"DUE TO TIME LIMIT[^\n]*", re.IGNORECASE),
    re.compile(r"CANCELLED AT[^\n]*", re.IGNORECASE),
    re.compile(r"^Killed\b[^\n]*", re.MULTILINE),
    re.compile(r"oom-?killer[^\n]*", re.IGNORECASE),
    re.compile(r"\bSIG(SEGV|ABRT|FPE|KILL|TERM|BUS)\b[^\n]*"),
    re.compile(r"^Backtrace for this error[^\n]*", re.MULTILINE),
    re.compile(r"^\s*failed in [^\n]+", re.MULTILINE),
    re.compile(r"^\s*STOP\s+[^\n]+", re.MULTILINE),
    re.compile(r"^\s*\w+\s+ierr\s+-?[1-9]\d*\b[^\n]*", re.MULTILINE),
]


def _last_match(text, pattern):
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


REASON_NORMALIZERS = [
    (re.compile(r"(stop because first omega > omega_crit)\s+[\d.eE+-]+\s+[\d.eE+-]+"),
     r"\1"),
]


def normalize_reason(reason):
    for pat, repl in REASON_NORMALIZERS:
        reason = pat.sub(repl, reason)
    return reason


def extract_termination(text):
    for pat in TERM_PATTERNS:
        m = _last_match(text, pat)
        if m:
            return normalize_reason(m)
    return None


def extract_crash(text):
    m = CRASH_PATTERN.search(text)
    return m.group(0).strip() if m else None


def extract_failure(text):
    best = None
    for prio, pat in enumerate(FAIL_PATTERNS):
        for m in pat.finditer(text):
            key = (m.end(), -prio)
            cand = (key, m.group(0).strip())
            if best is None or cand[0] > best[0]:
                best = cand
    return best[1] if best else None


def classify(log_path, stale_s):
    """Return (status, reason, mtime)."""
    p = Path(log_path)
    if not p.exists() or p.stat().st_size == 0:
        return ("PENDING", "", 0.0)

    mtime = p.stat().st_mtime
    text = p.read_text(errors="replace")

    term = extract_termination(text)
    if term:
        if any(nf in term for nf in NUMERICAL_FAILURES):
            return ("CRASHED", term, mtime)
        return ("COMPLETED", term, mtime)

    crash = extract_crash(text)
    if crash:
        return ("CRASHED", crash, mtime)

    fail = extract_failure(text)
    if fail:
        return ("CRASHED", fail, mtime)

    if (time.time() - mtime) < stale_s:
        return ("RUNNING", "", mtime)
    return ("STALLED", "", mtime)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_time(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) if t > 0 else ""


FIXED_LABELS = {
    "STALLED": "[??] STALLED (no termination message)",
    "PENDING": "[--] PENDING",
    "RUNNING": "[>>] RUNNING",
}
SECTION_ORDER = ["STALLED", "PENDING", "RUNNING", "CRASHED", "COMPLETED"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stale-minutes", type=int, default=DEFAULT_STALE_MINUTES,
                        help="mtime threshold for RUNNING vs STALLED (default %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress terminal output (files still written)")
    args = parser.parse_args()

    root = Path.cwd()
    stale_s = args.stale_minutes * 60

    rows = []
    by_status = defaultdict(list)
    tally = Counter()

    for l1, l2, l3, log in walk_grid(root):
        if not Path(l3).is_dir():
            continue
        status, reason, mtime = classify(log, stale_s)
        rows.append({
            "z_folder": l1,
            "m_folder": Path(l2).name,
            "model": Path(l3).name,
            "path": l3,
            "status": status,
            "reason": reason,
            "last_modified": fmt_time(mtime),
        })
        by_status[status].append((reason, l3))
        if reason:
            tally[(status, reason)] += 1

    total = len(rows)
    counts = {k: len(by_status.get(k, [])) for k in SECTION_ORDER}

    # --- status.csv ---
    with open("status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z_folder", "m_folder", "model",
                                          "status", "reason", "last_modified"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # --- side files ---
    with open("failed_models_to_delete.txt", "w") as f:
        for _, path in sorted(by_status.get("STALLED", []) + by_status.get("PENDING", [])):
            f.write(path + "\n")

    with open("managed_crashed_models.txt", "w") as f:
        for _, path in sorted(by_status.get("CRASHED", [])):
            f.write(path + "\n")

    with open("running_models.txt", "w") as f:
        for _, path in sorted(by_status.get("RUNNING", [])):
            f.write(path + "\n")

    # --- grid_status_report.txt ---
    bar = "=" * 70
    sub = "-" * 70
    with open("grid_status_report.txt", "w") as f:
        f.write(f"{bar}\n")
        f.write(" MESA GRID COMPLETION REPORT\n")
        f.write(f"{bar}\n")
        f.write(f" Total Models in Grid : {total}\n")
        f.write(f"{sub}\n")
        f.write(f" [OK] COMPLETED       : {counts['COMPLETED']}\n")
        f.write(f" [!!] CRASHED         : {counts['CRASHED']}\n")
        f.write(f" [>>] RUNNING         : {counts['RUNNING']}\n")
        f.write(f" [--] PENDING         : {counts['PENDING']}\n")
        f.write(f" [??] STALLED         : {counts['STALLED']}\n")
        f.write(f"{bar}\n\n")

        f.write(f"{bar}\n")
        f.write(" REASON TALLY\n")
        f.write(f"{bar}\n")
        f.write(" STATUS    | COUNT | REASON\n")
        f.write(f"{sub}\n")
        if tally:
            for (status, reason), n in tally.most_common():
                f.write(f" {status:<9s} | {n:<5d} | {reason}\n")
        else:
            f.write(" --        |   0   | No reasons recorded yet.\n")
        f.write(f"{bar}\n\n")

        f.write(f"{bar}\n")
        f.write(" DETAILED MODEL LIST\n")
        f.write(f"{bar}\n")
        f.write(f" {'REASON / STATUS':<55s} | MODEL DIRECTORY\n")
        f.write(f"{sub}\n")
        for status in SECTION_ORDER:
            entries = sorted(by_status.get(status, []))
            for reason, path in entries:
                if status in FIXED_LABELS:
                    label = FIXED_LABELS[status]
                elif status == "CRASHED":
                    label = f"[!!] {reason}"
                else:
                    label = reason
                f.write(f" {label:<55s} | {path}\n")

    # --- terminal output ---
    if not args.quiet:
        with open("grid_status_report.txt") as f:
            # Print only the header + tally, not the huge detail list
            for line in f:
                sys.stdout.write(line)
                if line.strip() == bar and "REASON TALLY" not in line:
                    pass
                if line.startswith(bar):
                    # Count bar lines; stop after the second block
                    pass
            # Simpler: just reprint the summary
        print()

        print(f"{bar}")
        print(" MESA GRID COMPLETION REPORT")
        print(f"{bar}")
        print(f" Total Models in Grid : {total}")
        print(f"{sub}")
        print(f" [OK] COMPLETED       : {counts['COMPLETED']}")
        print(f" [!!] CRASHED         : {counts['CRASHED']}")
        print(f" [>>] RUNNING         : {counts['RUNNING']}")
        print(f" [--] PENDING         : {counts['PENDING']}")
        print(f" [??] STALLED         : {counts['STALLED']}")
        print(f"{bar}\n")

        print(f"{bar}")
        print(" REASON TALLY")
        print(f"{bar}")
        print(f" STATUS    | COUNT | REASON")
        print(f"{sub}")
        if tally:
            for (status, reason), n in tally.most_common():
                print(f" {status:<9s} | {n:<5d} | {reason}")
        else:
            print(" --        |   0   | No reasons recorded yet.")
        print(f"{bar}")

        if counts["RUNNING"] > 0:
            print("\n--- Currently Running ---")
            for _, path in sorted(by_status["RUNNING"]):
                print(f"  [>>] {path}")

        n_rerun = counts["CRASHED"] + counts["STALLED"] + counts["PENDING"]
        if n_rerun > 0:
            print(f"\n--- {n_rerun} models need attention "
                  f"(CRASHED + STALLED + PENDING) ---")
            print(f"  See: managed_crashed_models.txt, failed_models_to_delete.txt")

        print(f"\n[OK] Full report : grid_status_report.txt")
        print(f"[OK] Per-model CSV : status.csv")


if __name__ == "__main__":
    main()

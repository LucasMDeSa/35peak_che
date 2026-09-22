#!/usr/bin/env python3
"""Classify grid models by evolutionary outcome and progress.

Categories:
  NON-CHE  — stop because Ysurf/Ycntr < 0.7
  MERGER   — stop because first omega > omega_crit
  CHE      — everything else that completed or crashed

For CHE models, reports how far they got based on saved .mod files
(H_depl, He_depl, C_depl, O_depl, CHE_single_core_collapse) and,
for those that did not reach C_depl, why they stopped and their
initial settings (Z, M, P).

Reads status.csv (from check_grid_completion.py) and walks the
filesystem for .mod files and inlist values.

Outputs:
  grid_outcome_summary.txt   human-readable report
  grid_outcome_detail.csv    per-model CSV
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ZSUN = 0.017

MOD_MILESTONES = [
    "O_depl.mod",
    "CHE_single_core_collapse.mod",
    "C_depl.mod",
    "CHE_logT895.mod",
    "He_depl.mod",
    "H_depl.mod",
]

MILESTONE_RANK = {
    "H_depl": 1,
    "He_depl": 2,
    "CHE_logT895": 3,
    "C_depl": 4,
    "CHE_single_core_collapse": 5,
    "O_depl": 6,
}


def highest_milestone(l3_path):
    best_rank = 0
    best_name = ""
    for mod in MOD_MILESTONES:
        if os.path.isfile(os.path.join(l3_path, mod)):
            name = mod.replace(".mod", "")
            rank = MILESTONE_RANK.get(name, 0)
            if rank > best_rank:
                best_rank = rank
                best_name = name
    return best_name or "none"


def parse_initial_settings(l3_path):
    z_val = m_val = p_val = ""
    inlist_both = os.path.join(l3_path, "inlist_both")
    if os.path.isfile(inlist_both):
        with open(inlist_both) as f:
            for line in f:
                m = re.match(r"\s*new_Z\s*=\s*(\S+)", line)
                if m:
                    z_val = m.group(1)
    inlist1 = os.path.join(l3_path, "inlist1")
    if os.path.isfile(inlist1):
        with open(inlist1) as f:
            for line in f:
                m = re.match(r"\s*initial_mass\s*=\s*(\S+)", line)
                if m:
                    m_val = m.group(1)
                m2 = re.match(r"\s*new_omega\s*=\s*(\S+)", line)
                if m2:
                    p_val = m2.group(1)
    return z_val, m_val, p_val


def read_central_abundances_and_dt(l3_path):
    hist = os.path.join(l3_path, "LOGS", "history.data")
    c_h1 = c_he4 = c_c12 = c_o16 = log_dt = log_teff = ""
    if not os.path.isfile(hist):
        return c_h1, c_he4, c_c12, c_o16, log_dt, log_teff
    try:
        with open(hist) as f:
            lines = f.readlines()
        hdr = lines[5].split()
        last = lines[-1].split()
        col = {n: i for i, n in enumerate(hdr)}
        c_h1 = last[col["center_h1"]] if "center_h1" in col else ""
        c_he4 = last[col["center_he4"]] if "center_he4" in col else ""
        c_c12 = last[col["center_c12"]] if "center_c12" in col else ""
        c_o16 = last[col["center_o16"]] if "center_o16" in col else ""
        log_dt = last[col["log_dt"]] if "log_dt" in col else ""
        log_teff = last[col["log_Teff"]] if "log_Teff" in col else ""
    except Exception:
        pass
    return c_h1, c_he4, c_c12, c_o16, log_dt, log_teff


_DATE_TIME_RE = re.compile(r"^DATE:\s*(\d{4}-\d{2}-\d{2})\s*\nTIME:\s*(\d{2}:\d{2}:\d{2})",
                           re.MULTILINE)


def read_runtime_hours(l3_path):
    log_path = l3_path + ".log"
    if not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
        stamps = _DATE_TIME_RE.findall(text)
        if not stamps:
            return ""
        t_start = datetime.strptime(f"{stamps[0][0]} {stamps[0][1]}", "%Y-%m-%d %H:%M:%S")
        if len(stamps) >= 2:
            t_end = datetime.strptime(f"{stamps[-1][0]} {stamps[-1][1]}", "%Y-%m-%d %H:%M:%S")
        else:
            t_end = datetime.now()
        hours = (t_end - t_start).total_seconds() / 3600
        return f"{hours:.1f}"
    except Exception:
        return ""


def z_from_l1(l1_name):
    m = re.search(r"ZdivZsun_(.+)$", l1_name)
    if not m:
        return ""
    z_div = float(m.group(1).replace("d", "e"))
    return f"{z_div * ZSUN:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status-csv", default="status.csv",
                        help="Path to status.csv from check_grid_completion.py")
    args = parser.parse_args()

    if not os.path.isfile(args.status_csv):
        sys.exit(f"Error: {args.status_csv} not found. Run check_grid_completion.py first.")

    rows = []
    with open(args.status_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    categories = {"NON-CHE": [], "MERGER": [], "CHE": []}
    che_by_milestone = defaultdict(list)
    incomplete_che = []
    detail_rows = []

    for r in rows:
        path = f"{r['z_folder']}/{r['m_folder']}/{r['model']}"
        reason = r.get("reason", "")
        status = r.get("status", "")

        if "Ysurf/Ycntr" in reason:
            cat = "NON-CHE"
        elif "omega_crit" in reason:
            cat = "MERGER"
        else:
            cat = "CHE"

        milestone = highest_milestone(path)
        z_nominal = z_from_l1(r["z_folder"])

        detail = {
            "path": path,
            "category": cat,
            "status": status,
            "reason": reason,
            "milestone": milestone,
            "z_folder": r["z_folder"],
            "z_nominal": z_nominal,
        }

        if cat == "CHE":
            che_by_milestone[milestone].append(detail)
            rank = MILESTONE_RANK.get(milestone, 0)
            if rank < MILESTONE_RANK["C_depl"]:
                z_val, m_val, p_val = parse_initial_settings(path)
                c_h1, c_he4, c_c12, c_o16, log_dt, log_teff = read_central_abundances_and_dt(path)
                detail["z_inlist"] = z_val
                detail["mass"] = m_val
                detail["omega"] = p_val
                detail["c_h1"] = c_h1
                detail["c_he4"] = c_he4
                detail["c_c12"] = c_c12
                detail["c_o16"] = c_o16
                detail["log_dt"] = log_dt
                detail["log_teff"] = log_teff
                detail["runtime_hr"] = read_runtime_hours(path)
                incomplete_che.append(detail)

        categories[cat].append(detail)
        detail_rows.append(detail)

    n_total = len(rows)
    n_nonche = len(categories["NON-CHE"])
    n_merger = len(categories["MERGER"])
    n_che = len(categories["CHE"])

    bar = "=" * 78
    sub = "-" * 78

    with open("grid_outcome_summary.txt", "w") as f:
        f.write(f"{bar}\n")
        f.write(" GRID OUTCOME SUMMARY\n")
        f.write(f"{bar}\n")
        f.write(f" Total models        : {n_total}\n")
        f.write(f" NON-CHE (Ysurf/Ycntr): {n_nonche}  ({100*n_nonche/n_total:.1f}%)\n")
        f.write(f" MERGER  (omega_crit) : {n_merger}  ({100*n_merger/n_total:.1f}%)\n")
        f.write(f" CHE                  : {n_che}  ({100*n_che/n_total:.1f}%)\n")
        f.write(f"{bar}\n\n")

        f.write(f"{bar}\n")
        f.write(" CHE MODELS — EVOLUTIONARY PROGRESS\n")
        f.write(f"{bar}\n")
        ordered = sorted(MILESTONE_RANK.items(), key=lambda x: -x[1])
        for name, _ in ordered:
            n = len(che_by_milestone[name])
            if n > 0:
                f.write(f"  {name:<30s} : {n:>5d}\n")
        n_none = len(che_by_milestone["none"])
        if n_none > 0:
            f.write(f"  {'(no .mod files)':<30s} : {n_none:>5d}\n")
        f.write(f"  {'TOTAL CHE':<30s} : {n_che:>5d}\n")
        f.write(f"{bar}\n\n")

        f.write(f"{bar}\n")
        f.write(" CHE MODELS NOT REACHING C_depl — BREAKDOWN\n")
        f.write(f"{bar}\n\n")

        inc_by_reason = defaultdict(list)
        for d in incomplete_che:
            inc_by_reason[d["reason"] or d["status"]].append(d)

        for reason in sorted(inc_by_reason, key=lambda r: -len(inc_by_reason[r])):
            models = inc_by_reason[reason]
            f.write(f" [{len(models):>4d}] {reason}\n")

        f.write(f"\n{sub}\n")
        f.write(f" CHE pre-C_depl — by Z\n")
        f.write(f"{sub}\n")
        inc_by_z = defaultdict(list)
        for d in incomplete_che:
            inc_by_z[d["z_folder"]].append(d)
        for z in sorted(inc_by_z):
            models = inc_by_z[z]
            reasons = Counter(d["reason"] or d["status"] for d in models)
            f.write(f"  {z}  ({len(models)} models)\n")
            for r, c in reasons.most_common():
                f.write(f"    {c:>4d}  {r}\n")

        f.write(f"\n{sub}\n")
        f.write(f" CHE pre-C_depl — by mass\n")
        f.write(f"{sub}\n")
        inc_by_mass = defaultdict(list)
        for d in incomplete_che:
            m = re.search(r"(\d{3}_m\d+)", d["path"])
            inc_by_mass[m.group(1) if m else "?"].append(d)
        for mass in sorted(inc_by_mass):
            models = inc_by_mass[mass]
            milestones = Counter(d["milestone"] for d in models)
            f.write(f"  {mass}  ({len(models)} models): "
                    + ", ".join(f"{m}={c}" for m, c in milestones.most_common())
                    + "\n")

        f.write(f"\n{sub}\n")
        f.write(f" DETAIL: CHE models that did not reach C_depl\n")
        f.write(f"{sub}\n")
        def fmt_abund(v):
            if not v:
                return "    —   "
            try:
                fv = float(v)
                if fv > 0.01:
                    return f"{fv:>8.3f}"
                elif fv > 1e-6:
                    return f"{fv:>8.1e}"
                else:
                    return f"{fv:>8.0e}"
            except ValueError:
                return f"{v:>8s}"

        f.write(f" {'MILESTONE':<14s} {'STATUS':<8s} {'Z_NOM':>10s} {'MASS':>8s} "
                f"{'c_h1':>8s} {'c_he4':>8s} {'c_c12':>8s} {'c_o16':>8s} "
                f"{'log_dt':>8s} {'log_Teff':>8s} {'HRS':>6s}  "
                f"{'REASON':<40s} PATH\n")
        f.write(" " + "-" * 190 + "\n")
        for d in sorted(incomplete_che, key=lambda x: (x["z_folder"],
                                                        x.get("mass", ""),
                                                        x["path"])):
            mass_short = d.get("mass", "").replace("d0", "").replace("d+", "e+").replace("d-", "e-")
            log_dt_str = fmt_abund(d.get("log_dt")) if d.get("log_dt") else "    —   "
            log_teff_str = fmt_abund(d.get("log_teff")) if d.get("log_teff") else "    —   "
            runtime_str = d.get("runtime_hr", "")
            f.write(f" {d['milestone']:<14s} {d['status']:<8s} {d['z_nominal']:>10s} "
                    f"{mass_short:>8s} "
                    f"{fmt_abund(d.get('c_h1')):>8s} {fmt_abund(d.get('c_he4')):>8s} "
                    f"{fmt_abund(d.get('c_c12')):>8s} {fmt_abund(d.get('c_o16')):>8s} "
                    f"{log_dt_str:>8s} {log_teff_str:>8s} {runtime_str:>6s}  "
                    f"{(d['reason'] or '(no reason)'):<40s} {d['path']}\n")

        f.write(f"\n{bar}\n")

    with open("grid_outcome_detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "category", "status", "reason",
                                           "milestone", "z_folder", "z_nominal",
                                           "z_inlist", "mass", "omega",
                                           "c_h1", "c_he4", "c_c12", "c_o16",
                                           "log_dt", "log_teff", "runtime_hr"])
        w.writeheader()
        for d in detail_rows:
            w.writerow({k: d.get(k, "") for k in w.fieldnames})

    with open("grid_outcome_summary.txt") as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    main()

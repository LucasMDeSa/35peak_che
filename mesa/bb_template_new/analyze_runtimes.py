#!/usr/bin/env python3

# ==============================================================================
# GRID RUNTIME ANALYZER
# ==============================================================================
# Scans all MESA log files in the grid structure.
# Filters for successful runs.
# Excludes models that terminated early due to CHE criteria:
#   - stop because first omega > omega_crit
#   - stop because Ysurf/Ycntr < 0.7
# Calculates exact runtimes for the remaining completed models.
# Calculates Mean, Std, Mode, Min, Max for:
#   - The overall grid
#   - Individual Metallicities (Z)
#   - Individual (Mass, Z) combinations
# Outputs summary to terminal and an exhaustive tally to a text file.
# ==============================================================================

import os
import glob
import re
import statistics
from datetime import datetime
from collections import defaultdict, Counter

OUTPUT_FILE = "grid_runtime_statistics.txt"

def float_from_label(label):
    """Converts labels like 'Z=1d-02' into float 0.01 for proper numerical sorting"""
    try:
        num_str = label.split('=')[1].replace('d', 'e').replace('D', 'e')
        return float(num_str)
    except Exception:
        return 0.0

def get_stats(data):
    """Returns (N, Mean, StdDev, Mode, Min, Max) for a list of times in minutes"""
    n = len(data)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    mean_val = statistics.mean(data)
    min_val = min(data)
    max_val = max(data)
    std_val = statistics.stdev(data) if n > 1 else 0.0
    
    # Mode calculation: Binning to the nearest minute creates a more meaningful mode
    binned_data = [round(x) for x in data]
    c = Counter(binned_data)
    mode_val = c.most_common(1)[0][0]
    
    return n, mean_val, std_val, mode_val, min_val, max_val

def format_row(label, stats, che_count, total_count):
    """Formats a row for the ASCII table"""
    n, mean, std, mode, min_v, max_v = stats
    che_str = f"{che_count}/{total_count}"
    return f"{label:<20} | {che_str:<10} | {mean:<10.1f} | {std:<10.1f} | {mode:<10} | {min_v:<8.1f} | {max_v:<8.1f}"

def main():
    print("Scanning grid for completed MESA logs...")
    
    log_files = glob.glob("*_ZdivZsun_*/*_m*/*.log")
    print(f"Found {len(log_files)} log files total. Parsing runtimes...")

    all_times = []
    z_times = defaultdict(list)
    mz_times = defaultdict(list)
    individual_tallies = []

    che_overall = 0
    total_completed_overall = 0
    
    z_che_count = defaultdict(int)
    z_total_count = defaultdict(int)
    
    mz_che_count = defaultdict(int)
    mz_total_count = defaultdict(int)

    count_crashed = 0
    count_missing_time = 0

    for log_file in log_files:
        path_parts = log_file.split(os.sep)
        if len(path_parts) < 3:
            continue
            
        z_folder = path_parts[-3]
        m_folder = path_parts[-2]
        model_name = path_parts[-1]

        z_match = re.search(r'_ZdivZsun_(.*)', z_folder)
        m_match = re.search(r'_m([0-9.]+)', m_folder)
        
        if not z_match or not m_match:
            continue
            
        z_label = f"Z={z_match.group(1)}"
        m_label = f"M={m_match.group(1)}"

        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()

        # Check for problem crashes
        if "stopping because of problems" in content:
            count_crashed += 1
            continue
            
        # Check for successful completion
        if not re.search(r"termination code|stop because", content, re.IGNORECASE):
            continue

        # Increment total completed counters
        total_completed_overall += 1
        z_total_count[z_label] += 1
        mz_total_count[(m_label, z_label)] += 1

        # Check for early termination conditions indicating it is NOT CHE
        is_non_che = re.search(r"stop because first omega\s*>\s*omega_crit|stop because Ysurf/Ycntr\s*<\s*0\.7", content, re.IGNORECASE)
        
        if is_non_che:
            continue  # Skip runtime parsing entirely for non-CHE models
            
        # If it didn't terminate early due to those conditions, it IS a CHE model
        che_overall += 1
        z_che_count[z_label] += 1
        mz_che_count[(m_label, z_label)] += 1

        # Extract runtime from MESA's standard DATE and TIME tags
        dates = re.findall(r"DATE:\s*(\d{4}[-/]\d{2}[-/]\d{2})", content, re.IGNORECASE)
        times = re.findall(r"TIME:\s*(\d{2}:\d{2}:\d{2})", content, re.IGNORECASE)
        
        # Fallback: Generic combined timestamps (e.g., 2024-05-15 12:34:56)
        generic_matches = re.findall(r"(\d{4}[-/]\d{2}[-/]\d{2})[T\s]+(\d{2}:\d{2}:\d{2})", content)
        
        if len(dates) >= 2 and len(times) >= 2:
            start_str = f"{dates[0].replace('/', '-')} {times[0]}"
            end_str = f"{dates[-1].replace('/', '-')} {times[-1]}"
        elif len(generic_matches) >= 2:
            start_str = f"{generic_matches[0][0].replace('/', '-')} {generic_matches[0][1]}"
            end_str = f"{generic_matches[-1][0].replace('/', '-')} {generic_matches[-1][1]}"
        else:
            count_missing_time += 1
            continue
            
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            
            time_mins = (end_dt - start_dt).total_seconds() / 60.0
            time_mins = max(0.0, time_mins)

            all_times.append(time_mins)
            z_times[z_label].append(time_mins)
            mz_times[(m_label, z_label)].append(time_mins)
            individual_tallies.append((z_label, m_label, model_name, time_mins))
            
        except ValueError:
            count_missing_time += 1

    # --- Generate Report ---
    report_lines = []
    report_lines.append("=============================================================================================")
    report_lines.append("                                 MESA GRID RUNTIME STATISTICS                                ")
    report_lines.append("=============================================================================================")
    report_lines.append(f"Total Completed     : {total_completed_overall} models")
    report_lines.append(f"Successfully Parsed : {len(all_times)} CHE models (full runtime stats)")
    report_lines.append(f"Non-CHE Excluded    : {total_completed_overall - che_overall} models (skipped from stats)")
    report_lines.append(f"Crashed/Failed      : {count_crashed} models (skipped)")
    if count_missing_time > 0:
        report_lines.append(f"Missing Time Data   : {count_missing_time} CHE models (no time string found)")
    report_lines.append("---------------------------------------------------------------------------------------------")
    
    header = f"{'GROUP':<20} | {'CHE/TOTAL':<10} | {'MEAN (min)':<10} | {'STD (min)':<10} | {'MODE (min)':<10} | {'MIN':<8} | {'MAX'}"
    report_lines.append(header)
    report_lines.append("-" * 93)

    # 1. Overall Stats
    overall_stats = get_stats(all_times)
    report_lines.append(format_row("ENTIRE GRID", overall_stats, che_overall, total_completed_overall))
    report_lines.append("-" * 93)

    # 2. Stats per Z
    sorted_z_keys = sorted(z_times.keys(), key=float_from_label)
    # Ensure Z levels that ONLY had CHE models still appear in the table
    for z in set(sorted_z_keys).union(z_total_count.keys()):
        stats = get_stats(z_times.get(z, []))
        report_lines.append(format_row(z, stats, z_che_count[z], z_total_count[z]))
    report_lines.append("-" * 93)

    # 3. Stats per (M, Z)
    all_mz_keys = set(mz_times.keys()).union(mz_total_count.keys())
    sorted_mz_keys = sorted(all_mz_keys, key=lambda k: (float_from_label(k[0]), float_from_label(k[1])))
    for mz in sorted_mz_keys:
        label = f"{mz[0]}, {mz[1]}"
        stats = get_stats(mz_times.get(mz, []))
        report_lines.append(format_row(label, stats, mz_che_count[mz], mz_total_count[mz]))
    
    report_lines.append("=============================================================================================")

    # Print summary to terminal
    summary_text = "\n".join(report_lines)
    print("\n" + summary_text)

    # Save exhaustive report to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(summary_text + "\n\n")
        f.write("=============================================================================================\n")
        f.write("                            INDIVIDUAL MODEL RUNTIMES (MINUTES)                              \n")
        f.write("                           (Non-CHE Models are excluded from list)                           \n")
        f.write("=============================================================================================\n")
        
        individual_tallies.sort(key=lambda x: (float_from_label(x[0]), float_from_label(x[1]), x[2]))
        
        for tally in individual_tallies:
            f.write(f"{tally[0]:<12} | {tally[1]:<12} | {tally[2]:<45} | {tally[3]:.1f} mins\n")

    print(f"\n[OK] Detailed statistics and the full individual model tally have been saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
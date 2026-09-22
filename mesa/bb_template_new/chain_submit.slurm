#!/bin/bash
#SBATCH --job-name=chain_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --partition=cpu-single
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/chain_submit_%j.out
#SBATCH --error=logs/chain_submit_%j.out
#SBATCH --mail-type=END,FAIL

# Chain-submit dispatch_grid parts sequentially, waiting for each to finish
# before submitting the next. Auto-discovers part count from
# submit_<tag>_part*.slurm files in the working directory.
#
# Usage:
#   sbatch chain_submit.slurm [--first-wait N] [--last N] [--poll N] [--tag T]
#
#   --first-wait N  Part already running to wait on (default: 0)
#   --last N        Last part to submit (default: auto-detect)
#   --poll N        Poll interval in seconds (default: 1800)
#   --tag T         Run list tag from dispatch_grid (default: grid;
#                   use missing_grid for run_missing_grid parts)

set -euo pipefail

# ── Parse arguments ─────────────────────────────────────────────────────────
FIRST_WAIT=0
LAST_PART=""
POLL_INTERVAL=1800
TAG="grid"

while [ $# -gt 0 ]; do
    case "$1" in
        --first-wait) FIRST_WAIT="$2"; shift 2 ;;
        --last)       LAST_PART="$2";  shift 2 ;;
        --poll)       POLL_INTERVAL="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        *)            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-detect last part from submit_<tag>_part*.slurm files ────────────────
if [ -z "$LAST_PART" ]; then
    max_found=-1
    for f in submit_${TAG}_part[0-9]*.slurm; do
        [ -f "$f" ] || continue
        n="${f#submit_${TAG}_part}"
        n="${n%.slurm}"
        if [ "$n" -gt "$max_found" ] 2>/dev/null; then
            max_found="$n"
        fi
    done
    if [ "$max_found" -lt 0 ]; then
        echo "ERROR: No submit_${TAG}_part*.slurm files found." >&2
        exit 1
    fi
    LAST_PART="$max_found"
fi

# ── Validate ────────────────────────────────────────────────────────────────
if [ "$FIRST_WAIT" -gt "$LAST_PART" ]; then
    echo "ERROR: --first-wait ($FIRST_WAIT) > --last ($LAST_PART)" >&2
    exit 1
fi

for (( i=FIRST_WAIT+1; i<=LAST_PART; i++ )); do
    script="submit_${TAG}_part${i}.slurm"
    if [ ! -f "$script" ]; then
        echo "ERROR: $script not found (need parts $((FIRST_WAIT+1))–$LAST_PART)." >&2
        exit 1
    fi
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_part() {
    local part=$1
    local jobname="${TAG}_part${part}"

    log "Waiting for $jobname to finish ..."
    while true; do
        local n
        n=$(squeue --name="$jobname" --user="$USER" --noheader 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ]; then
            log "$jobname: no jobs remaining in queue."
            return 0
        fi
        log "$jobname: $n job(s) still in queue. Sleeping ${POLL_INTERVAL}s."
        sleep "$POLL_INTERVAL"
    done
}

submit_part() {
    local part=$1
    local script="submit_${TAG}_part${part}.slurm"

    log "Submitting $script ..."
    local out
    out=$(sbatch "$script" 2>&1)
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "ERROR: sbatch failed (rc=$rc): $out"
        exit 1
    fi
    log "Submitted: $out"
}

# ── Main loop ───────────────────────────────────────────────────────────────
log "Chain submit: wait on part $FIRST_WAIT, then submit parts $((FIRST_WAIT+1))–$LAST_PART."

for (( part=FIRST_WAIT; part<LAST_PART; part++ )); do
    wait_for_part "$part"
    submit_part "$((part + 1))"
done

wait_for_part "$LAST_PART"
log "All parts through $LAST_PART complete. Chain finished."

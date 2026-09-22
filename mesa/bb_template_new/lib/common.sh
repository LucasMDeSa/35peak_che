#!/bin/bash
# Shared functions and constants for the CHE MESA grid infrastructure.
# Source this from any script via:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/lib/common.sh"        # top-level scripts
#   source "${SCRIPT_DIR}/../lib/common.sh"     # ZdivZsun_template/ scripts
#   source "${SCRIPT_DIR}/../../lib/common.sh"  # ZdivZsun_template/template/ scripts

ZSUN=0.017

# ---------------------------------------------------------------------------
# patch_inline FILE KEY VALUE
# Rewrite the "KEY = ..." line in FILE, key-anchored (independent of the
# previous value). Aborts with non-zero exit if the substitution didn't take.
# This replaces fragile value-anchored seds like s/= 0.0017d0/= ${z_str}/
# which silently no-op when the placeholder drifts.
# ---------------------------------------------------------------------------
patch_inline() {
    local file="$1" key="$2" value="$3"
    if [ ! -f "$file" ]; then
        echo "ERROR patch_inline: file not found: $file" >&2; exit 1
    fi
    if ! grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$file"; then
        echo "ERROR patch_inline: key '${key}' not found in $file" >&2; exit 1
    fi
    sed -i -E "s|^([[:space:]]*${key}[[:space:]]*=[[:space:]]*).*$|\1${value}|" "$file"
    local actual
    actual=$(awk -F= -v k="$key" '$0 ~ "^[[:space:]]*"k"[[:space:]]*=" {
        sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit
    }' "$file")
    if [ "$actual" != "$value" ]; then
        echo "ERROR patch_inline failed for key '${key}' in ${file}" >&2
        echo "       Expected: '${value}'" >&2
        echo "       Actual  : '${actual}'" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# get_physics_id
# Derive the physics ID (e.g. "00", "04") from the current directory name.
# ---------------------------------------------------------------------------
get_physics_id() {
    basename "$PWD" | cut -d_ -f1
}

# ---------------------------------------------------------------------------
# read_z_ignore
# Print z_ignore_grid entries (one per line) if the file exists.
# Usage: mapfile -t Z_IGNORE < <(read_z_ignore)
# ---------------------------------------------------------------------------
read_z_ignore() {
    local ignore_file="${1:-z_ignore_grid}"
    if [ -f "$ignore_file" ]; then
        grep -vE '^[[:space:]]*(#|$)' "$ignore_file"
    fi
}

# ---------------------------------------------------------------------------
# is_z_ignored Z_VALUE
# Returns 0 (true) if the Z value is in z_ignore_grid.
# ---------------------------------------------------------------------------
is_z_ignored() {
    local z_val="$1"
    local ignore_file="${2:-z_ignore_grid}"
    [ -f "$ignore_file" ] || return 1
    local z_float ignored_float
    z_float=$(printf "%.10g" "$z_val" 2>/dev/null) || return 1
    while IFS= read -r ignored; do
        ignored_float=$(printf "%.10g" "$ignored" 2>/dev/null) || continue
        [ "$z_float" = "$ignored_float" ] && return 0
    done < <(read_z_ignore "$ignore_file")
    return 1
}

# ---------------------------------------------------------------------------
# fortran_d VALUE FORMAT_SPEC
# Format a number in Fortran d-notation.
# Example: fortran_d 0.01 "%.0e" -> "1d-02"
#          fortran_d 0.00034 "%.2e" -> "3.40d-04"
# ---------------------------------------------------------------------------
fortran_d() {
    printf "$2" "$1" | sed 's/e/d/'
}

# ---------------------------------------------------------------------------
# load_grid_conf
# Source grid.conf from the grid root directory (walks up from SCRIPT_DIR).
# Sets variables with defaults so env vars can override.
# ---------------------------------------------------------------------------
load_grid_conf() {
    local conf
    # Find grid.conf by walking up from the calling script's directory
    local search_dir="${GRID_ROOT:-$PWD}"
    for d in "$search_dir" "$search_dir/.." "$search_dir/../.." "$search_dir/../../.."; do
        if [ -f "$d/grid.conf" ]; then
            conf="$(cd "$d" && pwd)/grid.conf"
            break
        fi
    done
    if [ -z "${conf:-}" ]; then
        echo "WARNING: grid.conf not found" >&2
        return 1
    fi
    # Source grid.conf, then apply env var overrides
    source "$conf"
}

# ---------------------------------------------------------------------------
# activate_mesa_snippet
# Print the shell snippet for activating MESA in a Slurm batch script.
# ---------------------------------------------------------------------------
activate_mesa_snippet() {
    local cmd="${MESA_ACTIVATE_CMD:-mesa-24031}"
    cat <<MESA_EOF
# Activate MESA (function defined in ~/.bashrc). Slurm batch shells
# don't source rc files automatically; wrap with set +u so unbound vars
# in the rc files don't abort the task.
set +u
source ~/.bashrc
${cmd}
set -u
MESA_EOF
}

# ---------------------------------------------------------------------------
# Shared MESA build
# One star binary per physics set, compiled in place in the L3 template
# (ZdivZsun_template/template/MESA_input_src). L3 model dirs symlink to it.
# L3 dirs always sit at SET/L1/L2/L3, so the relative link prefix is fixed.
# ---------------------------------------------------------------------------
SHARED_MESA_SRC="ZdivZsun_template/template/MESA_input_src"
SHARED_MESA_SRC_REL_FROM_L3="../../../${SHARED_MESA_SRC}"
# submit.sh / re_submit.sh: for manually rerunning a single model (cd L3 && sbatch submit.sh);
# no grid script calls them.
SHARED_MESA_LINKS="star rn re submit.sh re_submit.sh inlist inlist_pgstar history_columns.list profile_columns.list"

# ---------------------------------------------------------------------------
# build_shared_star SET_DIR
# Compile SET_DIR/ZdivZsun_template/template/MESA_input_src once, if star is
# missing or older than src/*.f90 or make/makefile. Aborts on failure.
# ---------------------------------------------------------------------------
build_shared_star() {
    local src="$1/${SHARED_MESA_SRC}"
    if [ ! -f "$src/star" ] || \
       [ -n "$(find "$src/src" "$src/make/makefile" -newer "$src/star" -print -quit)" ]; then
        if [ -z "${MESA_DIR:-}" ]; then
            echo "ERROR build_shared_star: MESA_DIR not set (activate MESA first, e.g. mesa-24031)" >&2
            exit 1
        fi
        echo "Compiling shared star binary in $src"
        chmod +x "$src/mk" "$src/clean" "$src/rn" "$src/re" "$src/submit.sh" "$src/re_submit.sh" 2>/dev/null || true
        (cd "$src" && ./mk)
    fi
    if [ ! -x "$src/star" ]; then
        echo "ERROR build_shared_star: $src/star not found after compilation" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# copy_without_build SRC/ DST/
# rsync a template tree, skipping MESA build products and run output so the
# shared star binary is never duplicated into L1/L2 copies.
# ---------------------------------------------------------------------------
copy_without_build() {
    rsync -a \
        --exclude='star' \
        --exclude='make/*.o' \
        --exclude='*.mod' \
        --exclude='*.smod' \
        --exclude='.mesa_temp_cache/' \
        --exclude='LOGS/' \
        --exclude='photos/' \
        --exclude='output.txt' \
        "$1" "$2"
}

# ---------------------------------------------------------------------------
# mk_thin_model_dir INLIST_SRC_DIR TARGET_L3 MASS_D OMEGA_D
# Create an L3 model dir: real inlist1 (patched) and inlist_both copied from
# INLIST_SRC_DIR, everything else symlinked to the shared build. TARGET_L3
# must be SET/L1/L2/L3 relative to the physics set layout.
# ---------------------------------------------------------------------------
mk_thin_model_dir() {
    local inlist_src="$1" target="$2" mass="$3" omega="$4" f
    mkdir -p "$target/LOGS"
    cp "$inlist_src/inlist1" "$inlist_src/inlist_both" "$target/"
    patch_inline "$target/inlist1" initial_mass "${mass}"
    patch_inline "$target/inlist1" new_omega "${omega} ! rad s-1"
    for f in $SHARED_MESA_LINKS; do
        ln -sfn "${SHARED_MESA_SRC_REL_FROM_L3}/$f" "$target/$f"
    done
    if [ ! -x "$target/star" ]; then
        echo "ERROR mk_thin_model_dir: $target/star does not resolve to a compiled binary." >&2
        echo "       Run ./mk_grid at the physics-set level first (it builds ${SHARED_MESA_SRC}/star)." >&2
        exit 1
    fi
}

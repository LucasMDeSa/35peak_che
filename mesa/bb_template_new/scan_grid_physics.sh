#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

PHYSICS_ID=$(get_physics_id)

INPUT_Z_GRID="z_div_zsun_grid"
INPUT_M_GRID="ZdivZsun_template/mass_grid"
INPUT_P_GRID="ZdivZsun_template/period_grid"

OUT="grid_physics_scan.txt"
SUM="grid_physics_scan_summary.txt"
: > "$OUT"
: > "$SUM"

Z_BAD=0; M_BAD=0; W_BAD=0; OK=0; MISSING_INL=0; TOTAL=0
declare -A SEEN_Z_PER_ZDIR
declare -A SEEN_M_PER_MDIR

CNT_Z=0
while read -r z_div_zsun; do
    ID_Z=$(printf "%02d" "$CNT_Z")

    if is_z_ignored "$z_div_zsun"; then
        CNT_Z=$((CNT_Z + 1)); continue
    fi

    Z_VAL=$(echo "scale=8; $z_div_zsun * $ZSUN" | bc -l)
    Z_EXPECTED=$(fortran_d "$Z_VAL" "%.2e")
    Z_SUFFIX=$(fortran_d "$z_div_zsun" "%.0e")
    DIR_L1="${PHYSICS_ID}${ID_Z}_ZdivZsun_${Z_SUFFIX}"

    CNT_M=0
    while read -r MASS; do
        CNT_M=$((CNT_M + 1))
        ID_M=$(printf "%03d" "$CNT_M")
        DIR_L2="${DIR_L1}/${ID_M}_m${MASS}"
        STR_MASS=$(fortran_d "$MASS" "%.3e")

        while read -r P_DAYS; do
            P_SEC=$(echo "scale=8; $P_DAYS * 86400" | bc -l)
            OMEGA=$(echo "scale=8; 2 * 4*a(1) / $P_SEC" | bc -l)
            STR_OMEGA=$(fortran_d "$OMEGA" "%.3e")
            STR_PDAYS=$(fortran_d "$P_DAYS" "%.3e")
            DIR_L3="${DIR_L2}/m${STR_MASS}_p${STR_PDAYS}_w${STR_OMEGA}"

            [ ! -d "$DIR_L3" ] && continue
            TOTAL=$((TOTAL+1))

            INL1="$DIR_L3/inlist1"
            INLB="$DIR_L3/inlist_both"
            if [ ! -f "$INL1" ] || [ ! -f "$INLB" ]; then
                MISSING_INL=$((MISSING_INL+1))
                echo "MISSING_INLIST | $DIR_L3" >> "$OUT"
                continue
            fi

            ACT_Z=$(grep -E "^[[:space:]]*new_Z" "$INLB" | head -1 | awk -F= '{gsub(/[[:space:]]/,"",$2); print $2}')
            ACT_ZB=$(grep -E "^[[:space:]]*Zbase" "$INLB" | head -1 | awk -F= '{gsub(/[[:space:]]/,"",$2); print $2}')
            ACT_M=$(grep -E "^[[:space:]]*initial_mass" "$INL1" | head -1 | awk -F= '{gsub(/[[:space:]]/,"",$2); print $2}')
            ACT_W=$(grep -E "^[[:space:]]*new_omega" "$INL1" | head -1 | awk -F= '{sub(/!.*/,"",$2); gsub(/[[:space:]]/,"",$2); print $2}')

            row_ok=1
            if [ "$ACT_Z" != "$Z_EXPECTED" ] || [ "$ACT_ZB" != "$Z_EXPECTED" ]; then
                Z_BAD=$((Z_BAD+1)); row_ok=0
                echo "Z_MISMATCH    | exp=$Z_EXPECTED act=$ACT_Z zbase=$ACT_ZB | $DIR_L3" >> "$OUT"
            fi
            if [ "$ACT_M" != "$STR_MASS" ]; then
                M_BAD=$((M_BAD+1)); row_ok=0
                echo "M_MISMATCH    | exp=$STR_MASS act=$ACT_M | $DIR_L3" >> "$OUT"
            fi
            if [ "$ACT_W" != "$STR_OMEGA" ]; then
                W_BAD=$((W_BAD+1)); row_ok=0
                echo "W_MISMATCH    | exp=$STR_OMEGA act=$ACT_W | $DIR_L3" >> "$OUT"
            fi
            [ "$row_ok" -eq 1 ] && OK=$((OK+1))

            SEEN_Z_PER_ZDIR[$DIR_L1]+=" $ACT_Z"
            SEEN_M_PER_MDIR[$DIR_L2]+=" $ACT_M"
        done < "$INPUT_P_GRID"
    done < "$INPUT_M_GRID"
    CNT_Z=$((CNT_Z + 1))
done < "$INPUT_Z_GRID"

{
    echo "=========================================================="
    echo " GRID PHYSICS SCAN SUMMARY"
    echo "=========================================================="
    echo " Total L3 models inspected : $TOTAL"
    echo " OK                        : $OK"
    echo " Z mismatches              : $Z_BAD"
    echo " Mass mismatches           : $M_BAD"
    echo " Omega mismatches          : $W_BAD"
    echo " Missing inlist files      : $MISSING_INL"
    echo "----------------------------------------------------------"
    echo " UNIQUE new_Z values per Z folder"
    echo "----------------------------------------------------------"
    for z in $(printf "%s\n" "${!SEEN_Z_PER_ZDIR[@]}" | sort); do
        uniq_z=$(echo "${SEEN_Z_PER_ZDIR[$z]}" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')
        echo "  $z  ->  $uniq_z"
    done
    echo "----------------------------------------------------------"
    echo " UNIQUE initial_mass per M folder (only those with multiple values)"
    echo "----------------------------------------------------------"
    for m in $(printf "%s\n" "${!SEEN_M_PER_MDIR[@]}" | sort); do
        uniq_m=$(echo "${SEEN_M_PER_MDIR[$m]}" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')
        n=$(echo "$uniq_m" | wc -w)
        if [ "$n" -ne 1 ]; then
            echo "  $m  ->  $uniq_m"
        fi
    done
    echo "=========================================================="
} > "$SUM"

echo "Done. Per-model issues -> $OUT ; summary -> $SUM"

#!/bin/bash

OUTPUT_FILE="hillclimb_sweep_results.txt"

run_sweep() {
    local params=$1
    local leakage=$2
    local ir=$3

    while true; do
        local cmd="python3 hillclimb_mldsa.py --params ${params} --leakage ${leakage} --num-keys 10 --max-iter 100000 --inf-rels ${ir} --workers 16 --seed 42 --default-optimizations --non-verbose"
        echo "Running ML-DSA-${params}, leakage ${leakage}, inf-rels ${ir} ..."
        echo "# ${cmd}" >> "$OUTPUT_FILE"
        output=$(${cmd} 2>&1)
        echo "${output}" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

        if echo "${output}" | grep -q "Summary: 10/10 keys recovered"; then
            ir=$((ir - 500))
        else
            break
        fi
    done
}

# ML-DSA-44
run_sweep 44 6 6000
run_sweep 44 7 11500
run_sweep 44 8 14500
run_sweep 44 9 14500

# ML-DSA-87
run_sweep 87 6 5000
run_sweep 87 7 10000
run_sweep 87 8 18500
run_sweep 87 9 18500

# ML-DSA-65
run_sweep 65 6 6000
run_sweep 65 7 11500
run_sweep 65 8 23000
run_sweep 65 9 35000

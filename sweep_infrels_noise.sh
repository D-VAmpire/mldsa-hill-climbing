#!/bin/bash

OUTPUT_FILE="hillclimb_sweep_results_noisy_65.txt"

run_sweep() {
    local params=$1
    local leakage=$2
    local ir=$3

    while true; do
        local cmd="python3 hillclimb_mldsa_noise.py --noise-level 0.45 --params ${params} --leakage ${leakage} --num-keys 5 --patience 1000 --inf-rels ${ir} --workers 16 --seed 42 --default-optimizations"
        echo "Running ML-DSA-${params}, leakage ${leakage}, inf-rels ${ir} ..."
        echo "# ${cmd}" >> "$OUTPUT_FILE"
        output=$(${cmd} 2>&1)
        echo "${output}" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

            if echo "${output}" | grep -qE "Summary: [45]/5 keys recovered"; then
            ir=$((ir - 500000))
        else
            break
        fi
    done
}

# ML-DSA-44
#run_sweep 44 6 2500000
#run_sweep 44 7 4500000
#run_sweep 44 8 5000000
##run_sweep 44 9 5000000
#
## ML-DSA-87
#run_sweep 87 6 2500000
#run_sweep 87 7 4500000
#run_sweep 87 8 6500000
#run_sweep 87 9 5000000
#
## ML-DSA-65
#run_sweep 65 6 3000000
#run_sweep 65 7 5000000
run_sweep 65 8 10000000
run_sweep 65 9 13000000

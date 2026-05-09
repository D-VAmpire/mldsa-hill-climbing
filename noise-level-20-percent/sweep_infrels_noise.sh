#!/bin/bash

export PYTHONUNBUFFERED=1
OUTPUT_FILE="hillclimb_sweep_results_noisy.txt"

# Forward SIGINT/SIGTERM to the running python process, wait for graceful exit
CHILD_PID=0
trap 'if [ $CHILD_PID -ne 0 ]; then kill -INT $CHILD_PID; wait $CHILD_PID; fi; exit 1' INT TERM

run_sweep() {
    local params=$1
    local leakage=$2
    local ir=$3

    while true; do
        local cmd="python3 ../hillclimb_mldsa_noise.py --noise-level 0.20 --params ${params} --leakage ${leakage} --num-keys 5 --patience 1000 --inf-rels ${ir} --workers 16 --seed 42 --default-optimizations --adaptive-w-max 3"
        echo "Running ML-DSA-${params}, leakage ${leakage}, inf-rels ${ir} ..."
        echo "# ${cmd}" >> "$OUTPUT_FILE"

        # Stream output to both terminal and log in real time
        ${cmd} 2>&1 | tee -a "$OUTPUT_FILE" &
        CHILD_PID=$!
        wait $CHILD_PID
        local exit_code=$?
        CHILD_PID=0

        echo "" >> "$OUTPUT_FILE"

        # If killed by signal, stop the sweep
        if [ $exit_code -ge 128 ]; then
            echo "Child exited with signal (code ${exit_code}), stopping."
            exit $exit_code
        fi

        if tail -20 "$OUTPUT_FILE" | grep -qE "Summary: [45]/5 keys recovered"; then
            ir=$((ir - 10000))
        else
            break
        fi
    done
}

# ML-DSA-44
run_sweep 44 6 60000
run_sweep 44 7 120000
run_sweep 44 8 140000
#run_sweep 44 9 140000

# ML-DSA-87
run_sweep 87 6 60000
run_sweep 87 7 110000
run_sweep 87 8 200000
#run_sweep 87 9 200000

# ML-DSA-65
run_sweep 65 6 60000
run_sweep 65 7 120000
run_sweep 65 8 230000
run_sweep 65 9 370000

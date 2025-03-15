#!/usr/bin/env bash

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

COMMIT_HASH=$(git rev-parse --short HEAD)

mkdir -p benchmarks
mkdir -p flamegraphs

echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

DATASETS=("diabetes" "cancer" "housing")

for dataset in "${DATASETS[@]}"; do
    echo "Running benchmark for $dataset..."

    sudo perf record -F 99 -g --output=perf.data &
    PERF_PID=$!

    sleep 1  # Ensure perf starts before the benchmark

    ./genetic_benchmark "$dataset" | tee "benchmarks/${dataset}_${COMMIT_HASH}.txt" &
    BENCH_PID=$!

    echo "Benchmark ($dataset) started with PID: $BENCH_PID, perf PID: $PERF_PID"

    wait $BENCH_PID
    BENCH_STATUS=$?

    echo "Benchmark ($dataset) finished. Ensuring perf record flushes data..."

    sudo kill -INT $PERF_PID 2>/dev/null
    wait $PERF_PID 2>/dev/null

    if [[ ! -f perf.data || ! -s perf.data ]]; then
        echo "Error: perf.data was not created or is empty for $dataset. Skipping flamegraph generation."
        continue
    fi

    echo "Generating flamegraph for $dataset..."
    sudo perf script > perf_script.txt

    if [[ ! -s perf_script.txt ]]; then
        echo "Error: perf script output is empty. Skipping flamegraph for $dataset."
        continue
    fi

    sudo cat perf_script.txt | /run/current-system/sw/bin/stackcollapse-perf.pl | /run/current-system/sw/bin/flamegraph.pl > "flamegraphs/${dataset}_${COMMIT_HASH}.svg"

    echo "Flamegraph for $dataset saved to flamegraphs/${dataset}_${COMMIT_HASH}.svg"

    # Cleanup perf data
    sudo rm -f perf.data perf_script.txt
done

echo "All benchmarks and flamegraphs completed."


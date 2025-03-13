#!/usr/bin/env bash

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root"
  exit 1
fi

COMMIT_HASH=$(git rev-parse --short HEAD)

mkdir -p benchmarks
mkdir -p flamegraphs

echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
DATASETS=("diabetes" "cancer" "housing")

for dataset in "${DATASETS[@]}"; do
  echo "Running benchmark for $dataset"
  ./genetic_benchmark "$dataset" | tee "benchmarks/${dataset}_${COMMIT_HASH}.txt" &

  BENCH_PID=$!

  sudo perf record -F 99 -g -p $BENCH_PID -- sleep 10

  wait $BENCH_PID
  PERF_STATUS=$?

    # Check if perf encountered an error
  if [[ $PERF_STATUS -ne 0 ]]; then
      echo "Error: perf record failed for $dataset. Skipping flamegraph generation."
      continue
  fi

    # Ensure perf.data is created
  if [ ! -f perf.data ]; then
      echo "Error: perf.data was not created for $dataset. Skipping flamegraph generation."
      continue
  fi

  echo "Benchmark for $dataset done, saved to benchmarks/${dataset}_${COMMIT_HASH}.txt"

  echo "Generating flamegraph for $dataset"
  #sudo perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraphs/${dataset}_${COMMIT_HASH}.svg
  sudo perf script | /run/current-system/sw/bin/stackcollapse-perf.pl | /run/current-system/sw/bin/flamegraph.pl > "flamegraphs/${dataset}_${COMMIT_HASH}.svg"
  echo "Flamegraph for $dataset done, saved to flamegraphs/${dataset}_${COMMIT_HASH}.svg"

  sudo rm -f perf.data

  
  #./genetic_benchmark "$dataset" | tee benchmarks/${dataset}_${COMMIT_HASH}.txt

  #echo "Benchmark for $dataset done, saved to benchmarks/${dataset}_${COMMIT_HASH}.txt"
done

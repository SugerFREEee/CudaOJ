#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-16777216}"

nvcc -O3 -DSOLVE_FILE='"simple.cu"' \
    "${SCRIPT_DIR}/benchmark_reduction.cu" \
    -o "${SCRIPT_DIR}/benchmark_simple"

nvcc -O3 -DSOLVE_FILE='"optimal.cu"' \
    "${SCRIPT_DIR}/benchmark_reduction.cu" \
    -o "${SCRIPT_DIR}/benchmark_optimal"

printf '\n===== simple (single kernel + atomicAdd) =====\n'
"${SCRIPT_DIR}/benchmark_simple" "${N}"

printf '\n===== optimal (float4 + two-stage) =====\n'
"${SCRIPT_DIR}/benchmark_optimal" "${N}"

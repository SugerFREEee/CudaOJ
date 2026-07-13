#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-16777216}"

nvcc -O3 -DSOLVE_FILE='"solve_float4.cu"' \
    "${SCRIPT_DIR}/benchmark_reduction.cu" \
    -o "${SCRIPT_DIR}/benchmark_float4_atomic"

nvcc -O3 -DSOLVE_FILE='"solve_float4_2stage.cu"' \
    "${SCRIPT_DIR}/benchmark_reduction.cu" \
    -o "${SCRIPT_DIR}/benchmark_float4_2stage"

printf '\n===== float4 + atomicAdd =====\n'
"${SCRIPT_DIR}/benchmark_float4_atomic" "${N}"

printf '\n===== float4 + two-stage reduction =====\n'
"${SCRIPT_DIR}/benchmark_float4_2stage" "${N}"

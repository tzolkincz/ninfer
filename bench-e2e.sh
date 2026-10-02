#!/usr/bin/env bash
# E2E benchmark: checkout a commit, build, serve, measure prefill + decode throughput.
#
# Usage: ./bench-e2e.sh [commit] [runs] [kv-dtype]
#   commit:   git ref (default: HEAD)
#   runs:     generation runs per prompt length (default: 2)
#   kv-dtype: KV cache dtype (default: k8v4; use nvfp4 for baseline)
#
# Reports per prompt length:
#   - Prefill tok/s: prompt_tokens / TTFT
#   - Decode tok/s:  completion_tokens / (total - TTFT)
#   - TTFT (ms):     time to first token
#
# Stops llama-swap before GPU work, starts it after (trap-protected).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMIT="${1:-HEAD}"
RUNS="${2:-2}"
KV_DTYPE="${3:-k8v4}"
PORT=8080
MODEL="/home/v/models/qwen3_8_27b_quasar_nvfp4.ninfer"
BENCH_PY="$REPO_DIR/bench_stream.py"

# (label, prompt, max_tokens)
declare -a TESTS=(
    "short|128|Explain recursion in one sentence."
    "medium|256|Write a detailed technical article about the history of programming languages from Fortran to Rust, covering at least 10 languages with their design philosophies, key innovations, and influence on later languages. Include code examples where helpful. Discuss the tradeoffs between static and dynamic typing, compilation models, and memory management strategies that each language introduced."
    "long|256|Write an extremely detailed and comprehensive technical reference document about the design and implementation of a modern GPU tensor processing unit. Cover the following topics in exhaustive detail: 1) The complete memory hierarchy from global memory through L2 cache to shared memory and registers, including exact bandwidths in gigabytes per second, latencies in clock cycles, and capacity in kilobytes at each level, and how each level interacts with the others during a typical GEMM kernel execution. 2) The warp scheduler architecture, including work distribution strategies such as GigaThread, how the four warp schedulers per SM select from eligible warps each cycle, instruction issue slots, dual-issue behavior, and how independent thread scheduling since Volta affects register file allocation and performance when threads within a warp diverge. 3) The complete instruction set for matrix multiply-accumulate operations at FP16, BF16, TF32, FP8 E4M3, FP8 E5M2, and INT8 precisions, including the exact tile sizes of 16x16x16 or 16x8x32, the number of pipeline stages, how the accumulator reads and writes interleaved register banks, and the exact instruction encoding for the HMMA, QMMA, and OMMA families. 4) The memory coalescing and sector alignment requirements for optimal global memory throughput, including the 32-byte sector model, how misaligned or strided accesses are penalized with additional transactions, the L1 cache line size of 128 bytes, and the exact behavior when a warp accesses addresses that span multiple sectors. 5) The concurrent copy engine architecture and how it enables zero-copy PCIe transfers, including the BAR1 aperture sizing, the relationship between pinned host memory and device virtual address mapping, the number of copy engines and their bidirectional bandwidth, and how the unified addressing mode allows the CPU to read and write device memory directly. 6) The power management and thermal throttling behavior, including the exact temperature thresholds in degrees Celsius at which the GPU begins to reduce clock speeds, the power states P0 through P12, how the GPU firmware adjusts clock speeds under sustained load using the thermal diode readings, and the exact power limit in watts that triggers the first throttling event. 7) The error correction mechanisms at each level of the memory hierarchy, from ECC on GDDR7 or HBM3 to chipkill protection, the exact syndrome calculation polynomials used for SEC-DED error correction, how single-bit errors are corrected transparently while double-bit errors trigger a page retirement, and the exact number of redundant bits per codeword. 8) The virtual memory management unit and how it maps between process virtual addresses and physical device memory, including the four-level page table walk, the page table walk cache with its set-associativity and line size, the TLB hierarchy with its L1 and L2 partitions for system and GPU pages, and the exact cost of a TLB miss in terms of memory transactions. 9) The kernel launch and scheduling model, including how the hardware work distributor assigns thread blocks to processing elements, the exact ordering guarantees between blocks on different SMs versus blocks on the same SM, how the grid dimension limits of 2147483647 x 65535 x 65535 are enforced, and the behavior when the grid is larger than the total number of resident blocks. 10) The complete programming model for cooperative groups, including grid-wide synchronization with the exact barrier implementation, cluster-level shared memory with its 228 KB capacity on Hopper, and the exact hardware support for distributed shared memory access across SM pairs using the DSMEM instructions."
)

cleanup() {
    pkill -f "ninfer-serve" 2>/dev/null || true
    sleep 2
    systemctl start llama-swap 2>/dev/null || service llama-swap start 2>/dev/null || true
    echo "[bench] llama-swap restarted"
}
trap cleanup EXIT

echo "=== E2E Benchmark ==="
echo "commit:   $COMMIT"
echo "kv-dtype: $KV_DTYPE"
echo "runs:     $RUNS per prompt"
echo ""

systemctl stop llama-swap 2>/dev/null || service llama-swap stop 2>/dev/null || true
sleep 3

cd "$REPO_DIR"
CURRENT_HEAD="$(git rev-parse HEAD)"
git checkout "$COMMIT" --detach 2>/dev/null || {
    echo "[bench] FATAL: cannot checkout $COMMIT"
    exit 1
}
SHORT_HASH="$(git rev-parse --short HEAD)"
echo "[bench] checked out $SHORT_HASH"

echo "[bench] building..."
cmake --build build -j48 2>&1 | tail -3
echo "[bench] build done"

SERVE="$REPO_DIR/build/apps/ninfer-serve"
# Pre-flight: verify GPUs are free
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 1000 ]; then
    echo "[bench] FATAL: GPU0 still has ${GPU_MEM} MiB used. Kill lingering processes."
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
    exit 1
fi
echo "[bench] GPUs free"

echo "[bench] starting serve..."
"$SERVE" "$MODEL" \
    --tp 2 --devices 0,1 \
    --host 127.0.0.1 --port "$PORT" \
    --model-id quasar \
    --kv-dtype "$KV_DTYPE" \
    --max-context 170000 --kv-capacity 170000 \
    --device-state-slots 4 --max-concurrency 4 \
    --spec mtp --draft-tokens 5 \
    --vision --vision-device 0 --max-vision-tokens 4096 \
    --lm-head-draft \
    --default-thinking-budget 14000 --preserve-thinking \
    --no-tp-mailbox \
    --default-max-tokens 18000 --pending-timeout-ms 180000 \
    > /tmp/ninfer-bench-serve.log 2>&1 &
SERVE_PID=$!

READY=0
for i in $(seq 1 120); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then READY=1; break; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[bench] FATAL: serve died during startup"
        tail -30 /tmp/ninfer-bench-serve.log
        exit 1
    fi
    sleep 1
done
if [ "$READY" -ne 1 ]; then
    echo "[bench] FATAL: serve not ready after 120s"
    tail -30 /tmp/ninfer-bench-serve.log
    exit 1
fi
echo "[bench] serve ready (${i}s)"
echo ""

# Run benchmarks
printf "%-6s │ %-12s │ %-12s │ %-8s │ %-8s │ %-8s\n" "label" "prefill" "decode" "TTFT" "comp" "total"
printf "      │          tok/s │          tok/s │      ms │      tok │      ms\n"
printf "%s\n" "──────────┼────────────────┼────────────────┼──────────┼──────────┼──────────"

for test_entry in "${TESTS[@]}"; do
    IFS='|' read -r label maxtok prompt <<< "$test_entry"

    sum_decode=0
    sum_prefill=0
    sum_ttft=0
    n_ok=0

    for run in $(seq 1 "$RUNS"); do
        result=$(python3 "$BENCH_PY" "$PORT" "$prompt" "$maxtok" 2>/dev/null) || true

        if [[ "$result" == OK* ]]; then
            read -r _ p_tok c_tok ttft_ms total_ms d_tps p_tps <<< "$result"
            echo "  $label run $run: prefill=${p_tps} tok/s  decode=${d_tps} tok/s  ttft=${ttft_ms}ms  ${c_tok}tok/${total_ms}ms"
            sum_decode=$((sum_decode + $(python3 -c "print(int($d_tps * 10))")))
            sum_prefill=$((sum_prefill + $(python3 -c "print(int($p_tps * 10))")))
            sum_ttft=$((sum_ttft + ttft_ms))
            n_ok=$((n_ok + 1))
        else
            echo "  $label run $run: ${result:-FAILED}"
        fi
    done

    if [ "$n_ok" -gt 0 ]; then
        avg_d=$(python3 -c "print(f'{$sum_decode / $n_ok / 10:.1f}')")
        avg_p=$(python3 -c "print(f'{$sum_prefill / $n_ok / 10:.1f}')")
        avg_t=$(python3 -c "print(f'{$sum_ttft / $n_ok:.0f}')")
        printf "%-6s │ %11s │ %11s │ %7sms │ %7s │ %7s\n" \
            "$label" "$avg_p" "$avg_d" "$avg_t" "-" "-"
    fi
done

echo ""
echo "=== RESULT: $SHORT_HASH ($KV_DTYPE) ==="

kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true

git checkout "$CURRENT_HEAD" --detach 2>/dev/null || git checkout main
echo "[bench] done. Log: /tmp/ninfer-bench-serve.log"

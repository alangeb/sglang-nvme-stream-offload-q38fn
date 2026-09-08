# Mamba Host-Offload: Invariant, Crash-Fix, Benchmark

## The Invariant

sglang Mamba state cache assumes **slot-indexed** state: each concurrent
request owns exactly one state slot; the slot index is stable for the
request lifetime. This is an **upstream invariant**.

## The Crash (Before Fix)

Under high concurrency on sm_120 (RTX PRO 6000), the GPU Mamba state buffer
overflows. Pre-fix code assumed all state fits in HBM. When it does not:
- `cudaErrorIllegalAddress` on the state copy kernel
- Server crashes mid-batch; no graceful degradation

## The Fix (cache-01 + cache-02 + cache-04 + sizing)

1. **Observability** (cache-01): log host/GPU state residency per step.
2. **Whole-node offload** (cache-02): when HBM pressure exceeds threshold,
   evict entire node state to host RAM. Slot invariant preserved.
3. **Kernels** (cache-04): CUDA host<->GPU state transfer kernels. Required by cache-02.
4. **Sizing** (upstream/0001): decouple `mamba_host_size` from `mamba_gpu_size`.

These are **invariant-safe**: the slot-to-state mapping never changes.

## Per-Slot Offload (cache-03) - EXPERIMENTAL

Offloads individual slots rather than whole nodes. This **breaks the slot
invariant** (a slot state can be split across host/GPU mid-step).

- Gate: `SGLANG_MAMBA_PER_SLOT=1` (default `0` = baseline path, inert)
- Results are **workload-dependent**: helps long-context single-request,
  hurts high-concurrency short-request.
- NOT recommended for production without benchmarking your workload.

## cache-05 (fork-edit4-inert)

No-op patch retained for bisect history. Does not affect runtime.

## Benchmark Method

- Hardware: 1x RTX PRO 6000 (sm_120), 96GB GDDR7
- Model: Qwen3.8-Flash-Next-NVFP4-SSD-Stream
- Workloads: (a) 1x32k ctx, (b) 8x4k ctx, (c) 32x1k ctx
- Metric: tokens/sec, P99 latency, OOM events
- Reproduce: `compose/pro6000.yaml` + `sglang.bench_serving`

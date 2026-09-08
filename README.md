# sglang-nvme-stream-offload-q38fn

> **sglang** (the engine) · **NVMe streaming** (the `ssd-stream`/PLE plugin) · **KV + Mamba host offloading** · tuned for **Qwen3.8-Flash-Next** (the worked example)

[📄 License](LICENSE) · [📋 Disclaimer](DISCLAIMER.md) · [🔒 Security](SECURITY.md)

## ⚠️ Warning

This is an **experimental build kit** for a research GPU stack. It is provided **as-is, with no warranty** — it may crash, produce incorrect outputs, or corrupt running inference under load. The reference model it serves is a **third-party, unofficial quantization** (not affiliated with the SGLang project or Qwen/Alibaba). The pinned baseline is the head of an **open, unmerged PR** and can be force-pushed or closed at any time. The NVMe streaming reader is **third-party** (we only vendor + build it). Run in an isolated environment at your own risk.

See [DISCLAIMER.md](DISCLAIMER.md) for full terms and [SECURITY.md](SECURITY.md) for safe-usage guidelines.

---

## What Is This

A **build kit** (NOT a fork) that reproduces a **SGLang** runtime with two features stock SGLang lacks, validated on the **RTX PRO 6000 (sm_120 Blackwell)**:

1. **NVMe streaming** — the `ssd-stream` (PLE) plugin streams a model's large lookup table off **NVMe via io_uring** instead of pinning it in host RAM (third-party; we vendor + build it — see Credits).
2. **KV + Mamba host offloading** — spills Mamba/linear-state checkpoints to host RAM and restores them on agent rewind (whole-node, invariant-safe).

The kit is **tuned for Qwen3.8-Flash-Next** as its **reference model** (the worked example, not the headline). GPU-free build; GPU required at runtime only.

## What It Does

- Builds the `sglang-nvme-stream-offload-q38fn` image from a pinned upstream **sglang** tree + 9 flash-next patches + 5 cache-offload patches (3 applied by default; cache-03 experimental opt-in, cache-05 inert) + 1 upstream sizing patch.
- Builds the **NVMe `ssd-stream` (PLE) plugin** from vendored Rust source (`vendor/sglang-ssd-stream/`) — an io_uring reader that keeps the 48 GB lookup table on SSD, not in RAM.
- **Mamba host-spill + rewind-restore** (whole-node, invariant-safe), enabled in the reference compose via `SGLANG_MAMBA_HOST_ANCHOR=1` — verified in production.
- Observability counters (spill count, host-pool utilization, sanity-check violations) via `SGLANG_MAMBA_HOST_DEBUG=1`.
- Launch compose for pro6000 (`compose/pro6000.yaml`).

## What It Does NOT Do

- **NOT a fork** of SGLang; patches are applied at build time onto a pinned tree.
- **Does NOT bundle model weights** (third-party; see provenance below).
- **5090 / dense 27B NOT supported** — sglang corrupts on sm_120 fp8 full-attention.
- **Per-slot offload** (cache-03) is **EXPERIMENTAL**, breaks an upstream invariant; opt-in only via `SGLANG_MAMBA_PER_SLOT=1` (patch is not even applied by default).
- Offload is a **memory feature active under eviction pressure**; it does **NOT** speed raw single-shot warm prefill (stock-on-GPU is comparable-or-better there; GPU restore beats host restore). Decode is ~parity. Its win is **cache-hit under pressure / long multi-turn / rewind** — see Benchmarks.

## The NVMe Streaming Feature (`ssd-stream` / PLE)

**NOT ours.** The io_uring SSD streaming reader is pinned as a git submodule at `vendor/sglang-ssd-stream/` (tag `v0.2.0`, upstream `garnermccloud/sglang-ssd-stream`) and is built from source by the Dockerfile (maturin). Clone with `--recurse-submodules` to fetch it. It streams the model's ~48 GB lookup table from NVMe while the GPU works, freeing that RAM for context/cache. We merely **vendor and build** it; all attribution is in [Credits](#credits--acknowledgements) and [NOTICE](NOTICE).

## The Offload Feature (KV + Mamba host offloading)

**Problem:** Agent rewind leaves KV cache intact but drops the mamba checkpoint → forces re-prefill from token 0. **Solution:** Spill mamba checkpoints to host RAM; restore on rewind. The applied patches are env-gated: `SGLANG_MAMBA_HOST_ANCHOR=1` enables the whole-node host-tier anchor (safe, invariant-preserving; set in the reference compose). Per-slot = opt-in via `SGLANG_MAMBA_PER_SLOT=1` (**experimental**, breaks the upstream slot invariant; the cache-03 patch is not applied by default — see Dockerfile). With no env set, the patched tree behaves like stock. Details: [docs/mamba-host-offload.md](docs/mamba-host-offload.md).

## Model Provenance (the reference model)

> **THIRD-PARTY, unofficial.** Reference model = `garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream` @ snapshot `7b719225242aacd3dbd3f9407468c2ee9a9d2594` — an unofficial NVFP4 quant + custom SSD-Stream/PLE variant of the **Qwen** family. **We do NOT redistribute weights.** Pull it yourself with your own `HUGGING_FACE_HUB_TOKEN`. Verify its license yourself before use. The kit's features are engine-level; Qwen3.8-Flash-Next is the model it is tuned and benchmarked against.

## Baseline Provenance

> **Built on sglang @ `3df8e1e7dbc5807696622afe2929b6c33c185ca3`** = HEAD of **OPEN, UNMERGED PR [#36644](https://github.com/sgl-project/sglang/pull/36644)** ('[Qwen3.8] Fix FP8 KV cache support in QSA'), base `qwen4-main-squashed` (NOT main). This pin is **NOT in public sglang main**; vendored for reproducibility. Pinned to an open PR = **fragile** (author can force-push/close); will rebase onto main once #36644 lands. **Track PR #36644.** Before publishing/redistributing this kit, verify the vendored tree's publishability (see [DISCLAIMER.md](DISCLAIMER.md)).

## How to Use

> **Prerequisites:** an NVIDIA GPU (RTX PRO 6000 / sm_120 for the reference config) with driver + `nvidia-container-toolkit`; Docker with the Compose plugin; `git`; the Hugging Face CLI (`pip install -U huggingface_hub` → provides `hf`); and an HF token with access to the third-party model repo.

### 1. Clone (with submodules)
```bash
git clone --recurse-submodules <this-repo-url> sglang-nvme-stream-offload-q38fn
cd sglang-nvme-stream-offload-q38fn
# Already cloned without submodules? Recover with:
git submodule update --init --recursive
```
Verify: `ls vendor/sglang-ssd-stream/Cargo.toml` exists (the pinned `v0.2.0` submodule).

### 2. Authenticate + pull the model (third-party — bring your own token)
```bash
hf auth login            # or: export HF_TOKEN=***
hf download garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594
```
This lands in the default cache `~/.cache/huggingface/hub`, which `compose/pro6000.yaml` already mounts into the container — no extra path wiring needed. Verify: `hf scan-cache` lists the snapshot.

### 3. Build the image (GPU-free)
```bash
docker build -t sglang-nvme-stream-offload-q38fn:latest .
# or: ./build.sh
```
The default target is `flash-next` (the serving image). A context-only sanity stage also exists: `docker build --target context-check .` (fast, no base pull, no GPU — verifies every COPY source exists).

### 4. Launch (GPU required at runtime)
```bash
export HUGGING_FACE_HUB_TOKEN=***        # forwarded into the container
docker compose -f compose/pro6000.yaml up
```
Verify: `curl -s localhost:8001/health` returns `200` once the model finishes loading (~3-4 min cold start).

> **The full launch flags live in `compose/pro6000.yaml` (single source of truth).** To change anything — memory fraction, host-offload sizing, speculative decode — edit that file. `SGLANG_MAMBA_HOST_ANCHOR=1` enables the KV+Mamba host offload. If you don't want speculative decode, drop the `--speculative-*` flags; if the draft (MTP) weights live in an `mtp/` subdir of your snapshot, point `--speculative-draft-model-path` at that subdir.

## Benchmarks

Verified in production under real eviction pressure (stress test against the offload endpoint):

| Signal | Value | Meaning |
|--------|-------|---------|
| `gpu_cache_usage` | 99.3% | genuine eviction pressure |
| external_prefix_cache hit rate | 82.6% (1,835,200 / 2,221,846) | host tier serving hits |
| rewind `cached_tokens` | 232,000 | rewinds hit the HOST tier, not re-prefill |
| preemptions / request errors | 0 / 0 | clean under pressure |

**Verdict: the offload is confirmed working in production.**

What to expect:

- Offload pays off on **cache-hit under pressure / long multi-turn / rewind** (avoids re-prefill, sustains
  concurrency when GPU KV is evicted).
- It does **not** win raw single-shot warm prefill — there, stock-on-GPU is comparable-or-better
  (GPU restore beats host restore).
- **Decode: ~parity.**
- Per-slot mode (cache-03) is opt-in for **invariant reasons**, not proven perf reasons.
- No benchmarks for 27B/5090 — out of this kit's scope (see compose note).

## Credits & Acknowledgements

> **We didn't build most of this.** This kit is glue around other people's work. Credit where it's due:

1. **Baseline engine — [sgl-project/sglang](https://github.com/sgl-project/sglang)** (Apache-2.0). The inference engine we build on. Our pinned baseline `3df8e1e7dbc5807696622afe2929b6c33c185ca3` is the **HEAD of OPEN PR [#36644](https://github.com/sgl-project/sglang/pull/36644)** ('[Qwen3.8] Fix FP8 KV cache support in QSA'), base branch `qwen4-main-squashed`, authored by **LingZ315**. **We build on their unmerged work**; this pin is **fragile** (can be force-pushed/closed) and is not in public main.
2. **NVMe streaming — [Garner McCloud / sglang-ssd-stream](https://github.com/garnermccloud/sglang-ssd-stream)** (Apache-2.0). The io_uring NVMe streaming reader / PLE plugin vendored at `vendor/sglang-ssd-stream/` is **entirely third-party — NOT ours**. We merely **vendor and build** it (maturin). All credit for the SSD-streaming design and implementation goes to Garner McCloud and the sglang-ssd-stream project.
3. **Model — the [Qwen](https://github.com/QwenLM) model family (Alibaba Qwen team)**, and the specific third-party quant/streaming artifact [`garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream`](https://huggingface.co/garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream) @ snapshot `7b719225242aacd3dbd3f9407468c2ee9a9d2594`. **We do NOT redistribute weights** — users pull it themselves.
4. **NVIDIA** (CUDA / cuDNN / TensorRT — proprietary, NVIDIA EULA) and **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** (used via `flashinfer-python`/`flashinfer-cubin`/`flashinfer-jit-cache` in the Dockerfile).

## License

Apache-2.0 (upstream sglang is Apache-2.0; our patches inherit). See [LICENSE](LICENSE), [NOTICE](NOTICE), and [DISCLAIMER.md](DISCLAIMER.md) (incl. the vendored-baseline publishability caveat).

## Layout

```
Dockerfile
build.sh
compose/pro6000.yaml
patches/flash-next/       (9 patches)
patches/cache-offload/    (5 patches; 3 applied by default, cache-03 opt-in, cache-05 inert)
patches/upstream/         (1 sizing patch)
vendor/sglang-ssd-stream/ (PLE Rust/io_uring NVMe reader — THIRD-PARTY, Garner McCloud)
vendor/sglang-baseline/   (pinned sglang tree @ PR #36644 HEAD)
docs/
LICENSE  NOTICE  DISCLAIMER.md  SECURITY.md
```
- **FlashAttention CUTE kernel** (BSD-3-Clause) — vendored third-party attention kernel; contributors per its AUTHORS file; CuTe DSL from NVIDIA CUTLASS.

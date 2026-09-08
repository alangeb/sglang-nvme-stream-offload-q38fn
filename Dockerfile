# SGLang NVMe-Stream + Host-Offload Build Kit — pro6000 (sm_120)
# Clean-room rebuild of the sglang NVMe-stream/offload stack from upstream + our patches.
# GPU-FREE build (no --gpus, never launches a model). Run with GPU only at launch.
#
# BASE: lmsysorg/sglang:v0.5.18 (CUDA, sm_120-capable). Apache-2.0.
# SOURCE: upstream sglang @ pin 3df8e1e7dbc5807696622afe2929b6c33c185ca3 + our patches.
# VENDOR: sglang-ssd-stream (Rust/pyo3 PLE reader) built from vendor/ via maturin.
#
# Build:  docker build -t sglang-nvme-stream-offload-q38fn:latest .
# Launch: docker compose -f compose/pro6000.yaml up -d   (GPU at runtime only)

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.18

# ---- context-check: prove every COPY source exists in THIS kit's context ----
# NOT the default target (flash-next is). Run explicitly:
#   docker build --target context-check .   (no base pull, no GPU)
FROM scratch AS context-check
COPY vendor/sglang-baseline /baseline
COPY vendor/sglang-ssd-stream /ssdstream
COPY patches/flash-next /patches/flash-next
COPY patches/cache-offload /patches/cache-offload
COPY patches/upstream /patches/upstream

# ---- Stage 1: build the SSD-Stream plugin wheel (Rust/pyo3 io_uring reader) ----
FROM ${BASE_IMAGE} AS ssd-builder
COPY vendor/sglang-ssd-stream /opt/src/sglang-ssd-stream
RUN pip install --break-system-packages --no-cache-dir "maturin>=1.9,<2" \
 && cd /opt/src/sglang-ssd-stream \
 && maturin build --release --out /wheels --interpreter python3 \
 && ls -l /wheels

# ---- Stage 2: runtime image — upstream sglang + patch series ----
FROM ${BASE_IMAGE} AS flash-next

ARG SGLANG_COMMIT=3df8e1e7dbc5807696622afe2929b6c33c185ca3
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ENV SGLANG_BUILD_RUST_EXTS=none
ENV SGLANG_PKG=/usr/local/lib/python3.12/dist-packages

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Vendored clean upstream tree @ pin 3df8e1e7 (no tarball URLs, no network clone —
# the pin is not on public main). Patches are applied on top below.
COPY vendor/sglang-baseline /opt/sglang-src
RUN git -C /opt/sglang-src init -q && \
    git -C /opt/sglang-src add -A && \
    git -C /opt/sglang-src -c user.email=k@kit -c user.name=kit commit -qm "baseline 3df8e1e7"

# Apply the 9 flash-next model/quant patches in order.
COPY patches/flash-next/ /opt/patches/flash-next/
RUN for p in $(ls /opt/patches/flash-next/*.patch | sort); do \
      echo "applying $(basename $p)"; git -C /opt/sglang-src apply "$p"; done

# Apply cache-offload DEFAULT set (invariant-safe): 01 observability + 02 whole-node
# + 04 kernels + sizing. These are env-gated; the offload path is inert unless
# SGLANG_MAMBA_HOST_OFFLOAD / SGLANG_MAMBA_PER_SLOT are set at runtime.
COPY patches/cache-offload/ /opt/patches/cache-offload/
COPY patches/upstream/ /opt/patches/upstream/
RUN git -C /opt/sglang-src apply /opt/patches/cache-offload/cache-01-observability.patch && \
    git -C /opt/sglang-src apply /opt/patches/cache-offload/cache-02-whole-node.patch && \
    git -C /opt/sglang-src apply /opt/patches/cache-offload/cache-04-kernels.patch && \
    git -C /opt/sglang-src apply /opt/patches/upstream/0001-decouple-mamba-host-size.patch

# OPTIONAL (EXPERIMENTAL, opt-in): per-slot offload breaks the upstream slot-indexed
# invariant. NOT applied by default. To enable, uncomment AND set SGLANG_MAMBA_PER_SLOT=1:
# RUN git -C /opt/sglang-src apply /opt/patches/cache-offload/cache-03-per-slot.patch
# OPTIONAL inert (bisect-history only, no runtime effect):
# RUN git -C /opt/sglang-src apply /opt/patches/cache-offload/cache-05-fork-edit4-inert.patch

# Install sglang from the patched tree + vendor-pinned deps (mirrors shipped image).
RUN pip install --break-system-packages --no-cache-dir \
      "/opt/sglang-src/python" \
      "flashinfer-python==0.6.17" \
      "nvidia-modelopt==0.45.0" \
      "nvidia-cuda-cccl==13.0.85" \
      "nvidia-cuda-crt==13.0.88" \
      "nvidia-cuda-nvcc==13.0.88" \
      "nvidia-cuda-nvrtc==13.0.88" \
      "nvidia-cuda-runtime==13.0.96" \
      "nvidia-nvjitlink==13.0.88" \
      "nvidia-nvvm==13.0.88" \
 && pip install --break-system-packages --no-cache-dir --no-deps \
      --index https://flashinfer.ai/whl "flashinfer-cubin==0.6.17" \
 && pip install --break-system-packages --no-cache-dir --no-deps \
      --index https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.17+cu130" \
 && python3 -c "import sglang, os; p=os.path.dirname(sglang.__file__); print('sglang', sglang.__version__, p); assert p.startswith('/usr/local/lib/python3.12'), p"

# Install the SSD-Stream wheel built in stage 1.
COPY --from=ssd-builder /wheels/ /tmp/ssd-wheels/
RUN pip install --break-system-packages --no-cache-dir --no-deps \
      /tmp/ssd-wheels/sglang_ssd_stream-*.whl "huggingface-hub>=0.36,<2" \
 && rm -rf /tmp/ssd-wheels

# Runtime defaults: per-slot OFF (invariant-safe baseline path). Host-tier knobs
# (read by the applied patches): SGLANG_MAMBA_HOST_SIZE_GB, SGLANG_MAMBA_HOST_ANCHOR,
# SGLANG_MAMBA_SPILL_LOW_WATER, SGLANG_MAMBA_HOST_DEBUG. See docs/mamba-host-offload.md.
ENV SGLANG_MAMBA_PER_SLOT=0
ENV PYTHONUNBUFFERED=1

EXPOSE 30000
CMD ["sglang", "serve", "--help"]
#!/usr/bin/env bash
# Thin wrapper: build the kit image. GPU-free.
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-sglang-nvme-stream-offload-q38fn:latest}"
echo "Building $TAG from kit context..."
docker build --target flash-next -t "$TAG" .
echo "Done: $TAG"

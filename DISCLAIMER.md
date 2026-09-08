# Disclaimer

This software is provided **"AS IS"**, without warranty of any kind, express or implied, including without limitation the warranties of merchantability, fitness for a particular purpose, and non-infringement (see the Apache-2.0 *Disclaimer of Warranty*, section 7, in [LICENSE](LICENSE)).

The authors are not responsible for any damages, data loss, security breaches, or system failures resulting from the use of this software.

## Experimental — Use At Your Own Risk

This kit is a research-grade GPU stack. It:
- may crash or corrupt in-flight inference under load (the offload path exists precisely because stock behavior crashed under HBM pressure)
- may produce incorrect or unsafe outputs
- is not guaranteed to be safe, accurate, or reliable
- includes an experimental per-slot offload patch (`cache-03`) that **breaks an upstream invariant** and is opt-in for that reason

## Third-Party Model — Not Bundled, Not Affiliated

- The model `garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream` is a **third-party, unofficial** NVFP4 quantization. We do **not** bundle or redistribute its weights; you pull it yourself with your own token and must verify its license yourself.
- This project is **not affiliated with, endorsed by, or sponsored by** the sgl-project/sglang maintainers or Qwen/Alibaba.

## Vendored Baseline — Publishability Caveat

`vendor/sglang-baseline/` pins sglang @ `3df8e1e7` = HEAD of **open, unmerged PR [#36644](https://github.com/sgl-project/sglang/pull/36644)** (base `qwen4-main-squashed`), which is **NOT in public sglang main**. Consequences:
- **Fragile pin:** the PR author can force-push or close the PR; this exact tree only survives in the vendored copy.
- **Publishability:** before publishing or redistributing this kit, **you must verify** that redistributing this vendored tree is acceptable under sglang's Apache-2.0 terms and the PR's current status. Apache-2.0 permits redistribution with attribution (see [NOTICE](NOTICE)), but this is unreviewed upstream code — treat it as such.

## User Responsibility

Users are solely responsible for validating outputs, securing their environment, compliance with applicable laws and the licenses of all components (sglang, the model, and NVIDIA CUDA/cuDNN/TensorRT — proprietary, subject to the NVIDIA EULA), and reviewing all code before execution.

## No Liability

Under no circumstances shall the authors be liable for any claim, damages, or other liability arising from use of the software.

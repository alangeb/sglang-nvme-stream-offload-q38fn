# Security Policy

## Supported Versions

This is experimental software pinned to an open upstream PR. No versions are guaranteed secure or stable.

## Reporting Vulnerabilities

If you discover a security vulnerability in this kit or its patches, report it privately to the repository owner. Do **NOT** publicly disclose vulnerabilities without coordination. Upstream sglang issues should follow the [sgl-project/sglang security policy](https://github.com/sgl-project/sglang).

## Safe Usage Guidelines

- Run in isolated environments (Docker / VM). The reference compose publishes port 30000 — do **not** expose it to the public internet; put auth/reverse-proxy in front.
- The container receives your `HUGGING_FACE_HUB_TOKEN`; restrict what the container can reach.
- Do not provide production credentials to the container.
- Restrict filesystem and network access; the HF cache is mounted read-only in the reference compose — keep it that way.
- The experimental per-slot offload (`SGLANG_MAMBA_PER_SLOT=1`) breaks an upstream invariant: assume it can corrupt cache state.

## Scope

This project is not designed for production deployment or handling sensitive data without your own hardening and review.

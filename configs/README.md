# `configs/` — DeepSpeed Configs

This directory contains configuration files for DeepSpeed distributed training.

## Configuration Files

- **`ds_zero2.json`**: DeepSpeed ZeRO-Stage 2 configuration with optimizer offloading to CPU enabled to minimize GPU VRAM consumption.
- **`ds_zero2_no_offload.json`**: DeepSpeed ZeRO-Stage 2 configuration without CPU offloading, used when GPU VRAM is sufficient (e.g., on H100 nodes) to avoid CPU-GPU transfer bottlenecks.

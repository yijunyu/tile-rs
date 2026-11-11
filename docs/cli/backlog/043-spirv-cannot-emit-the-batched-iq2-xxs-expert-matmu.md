id: 43
area: lowering
title: spirv cannot emit the batched IQ2_XXS expert matmul that LLM prefill needs
opened: 2026-11-04

## wanted
A batched expert matvec/matmul over IQ2_XXS (and IQ1_M) weights for SPIR-V, so a
llama.cpp Vulkan prefill path could use a tile-rs kernel. Shape: 2304 rows x 5120 K per
expert matrix, 6 of 192 experts per token, 16-64 token tile, dequantize each weight once
and reuse it across the tile.

## got
$ tile --list-ops | awk 'NR==1 || /matvec|matmul|dequant|quant/'
op                       linalg   rvv      pto      msl      gpu      musa     spirv ...
matvec                   yes      yes      yes      -        -        -        -     ...
matmul                   yes      yes      yes      yes      yes      yes      -     ...
matmul_transposed        -        -        yes      yes      yes      yes      yes   ...

So spirv has matmul_transposed only, and there is no quant/dequant op on any target, so
the dequant+dot cannot be expressed in the tile DSL at all (not just for spirv).

## workaround
Hand-written GLSL IQ2_XXS dequant+dot (ggml 66-byte block layout, real lane map), timed
on the Strix Halo iGPU with this repo's Vulkan harness (crates/tile_cli/assets/harness/
vulkan.c):
  - single-token 2304x5120 matvec: 69 us per matrix (3.04 MB read, 44 GB/s)
  - batched 16-token, weights dequantized once: 224 us = 14 us/token (1.68 TFLOP/s)
The batch reuse is a 4.9x win per token, but 1.68 TFLOP/s is below the ~4 TFLOP/s
llama.cpp's Vulkan kernels already sustain, so no hook was wired. The remaining lever
there is a cooperative-matrix / INT8-dot formulation, not the naive loop.

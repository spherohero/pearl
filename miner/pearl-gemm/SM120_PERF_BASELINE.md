# sm_120 Performance Baseline (RTX 5090)

Date: 2026-06-23
Branch: `perf/sm120-baseline`
Device: NVIDIA GeForce RTX 5090 (`sm_120`)
Docker image: `vllm_miner_5090:latest`
Torch/CUDA: `torch 2.11.0+cu130`

## Benchmark script

`bench_sm120_baseline.py` times `noisy_gemm` with CUDA events and supports filters:

```bash
python3 bench_sm120_baseline.py \
  --shapes 1024x1024x512,8192x6144x4096 \
  --tiles 64x128x64,128x128x64 \
  --rs 64,128 \
  --stages 2 \
  --modes full,no_denoise \
  --dtypes fp16,int32
```

## Caveats

- CUDA event timing only; no Nsight/CUPTI counters yet.
- TOPS is **main GEMM equivalent** only: `2*M*N*K / time`. The full path also includes noising, denoise, scaling, host-signal/reduction plumbing, etc.
- Small shapes are dominated by launch/overhead and should not drive optimization choices.

## Quick sweep command

```bash
python3 bench_sm120_baseline.py \
  --warmup 1 \
  --iters 3 \
  --shapes 128x128x256,1024x1024x512,1025x1032x512
```

Results CSV: `_sm120_perf_baseline.csv`

### Best full-path result per shape/dtype

| Shape | dtype | Best tile | R | stages | median ms | main-equiv TOPS |
|---|---:|---|---:|---:|---:|---:|
| 128x128x256 | fp16 | 64x64x64 | 64 | 2 | 0.0654 | 0.13 |
| 128x128x256 | int32 | 64x128x64 | 128 | 2 | 0.0639 | 0.13 |
| 1024x1024x512 | fp16 | 64x128x64 | 64 | 2 | 0.0883 | 12.16 |
| 1024x1024x512 | int32 | 64x128x64 | 64 | 2 | 0.0839 | 12.79 |
| 1025x1032x512 | fp16 | 64x128x64 | 64 | 2 | 0.1000 | 10.84 |
| 1025x1032x512 | int32 | 64x128x64 | 64 | 2 | 0.0939 | 11.53 |

## Targeted big-shape command

```bash
python3 bench_sm120_baseline.py \
  --warmup 2 \
  --iters 10 \
  --shapes 8192x6144x4096 \
  --tiles 64x128x64,128x128x64 \
  --rs 64,128 \
  --stages 2 \
  --modes full,no_denoise \
  --dtypes fp16,int32 \
  --out /work/_sm120_perf_bigshape.csv
```

Results CSV: `_sm120_perf_bigshape.csv`

### Big-shape full-path ranking

| dtype | tile | R | stages | median ms | mean ms | min ms | main-equiv TOPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| int32 | 64x128x64 | 128 | 2 | 9.4780 | 9.9256 | 9.4417 | 43.50 |
| int32 | 64x128x64 | 64 | 2 | 9.4832 | 9.7744 | 9.3479 | 43.48 |
| fp16 | 64x128x64 | 64 | 2 | 9.4843 | 9.7741 | 9.3239 | 43.47 |
| fp16 | 64x128x64 | 128 | 2 | 9.5174 | 9.9880 | 9.4146 | 43.32 |
| fp16 | 128x128x64 | 64 | 2 | 10.8932 | 11.1503 | 10.2967 | 37.85 |
| fp16 | 128x128x64 | 128 | 2 | 10.9378 | 11.3174 | 10.6148 | 37.70 |
| int32 | 128x128x64 | 64 | 2 | 12.4108 | 17.3413 | 11.4410 | 33.22 |
| int32 | 128x128x64 | 128 | 2 | 12.9938 | 15.4978 | 11.3180 | 31.73 |

### Big-shape no-denoise ranking

| dtype | tile | R | stages | median ms | mean ms | min ms | main-equiv TOPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| fp16 | 64x128x64 | 128 | 2 | 9.4486 | 9.9826 | 9.3378 | 43.64 |
| fp16 | 64x128x64 | 64 | 2 | 9.4622 | 9.6619 | 9.3925 | 43.58 |
| int32 | 64x128x64 | 128 | 2 | 9.4755 | 9.9441 | 9.3653 | 43.51 |
| int32 | 64x128x64 | 64 | 2 | 9.5132 | 9.9464 | 9.4146 | 43.34 |
| fp16 | 128x128x64 | 64 | 2 | 11.4759 | 13.5643 | 10.9168 | 35.93 |
| fp16 | 128x128x64 | 128 | 2 | 11.8620 | 13.3020 | 10.7673 | 34.76 |
| int32 | 128x128x64 | 128 | 2 | 12.3762 | 17.3392 | 11.4182 | 33.32 |
| int32 | 128x128x64 | 64 | 2 | 13.0988 | 17.4000 | 11.3311 | 31.48 |

## Interpretation

1. `64x128x64 stages=2` is decisively best for the large mining-sized shape.
2. `R=64` and `R=128` are effectively tied for `64x128x64`; differences are below run-to-run jitter.
3. `fp16` and `int32` denoise/noising factor modes are effectively tied for the best tile.
4. `128x128x64 stages=2` is 15-37% slower on the large shape and should not be the default performance target.
5. Full vs no-denoise is almost identical for `64x128x64` at the big shape, so denoise overhead is not the dominant bottleneck at this scale.
6. Best measured big-shape full-path throughput is about `43.5` main-equivalent TOPS.

## Current default candidate

```text
mode:   full
dtype:  int32 or fp16 (tie)
tile:   64x128x64
R:      64 or 128 (tie)
stages: 2
shape:  8192x6144x4096
median: ~9.48 ms
TOPS:   ~43.5 main-equivalent
```

## Recommended next profiling step

Use Nsight Compute on the best `64x128x64 stages=2` config for the big shape and inspect:

- tensor pipe utilization,
- shared-memory/load-store pressure,
- register pressure/occupancy,
- barrier/synchronization overhead,
- global-store efficiency from the direct register-to-gmem epilogue.

If we want a code optimization before Nsight, the most promising low-risk experiment is to **disable or deprioritize the slower `128x128x64` configs for sm_120 performance selection** and keep `64x128x64 stages=2` as the performance default.

"""Focused SM120 runtime-swift benchmark on RTX 5090.

Sweeps runtime knobs only against the best-known matmul config:
  - 64x128x64 R64/R128 stages=2
  - swizzle: None / 0 / 4 / 8 / 16 / 32 / 64 / 128
  - swizzle_n_maj: True / False
  - skip_denoising: False / True
  - denoise dtype: fp16 / int32

Run inside vllm_miner_5090:sm120-opt
  python3 bench_sm120_runtime_swift.py --shape 8192x6144x4096 --iters 20
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, "/work/src")
os.environ.setdefault("PEARL_GEMM_DISABLE_R32", "TRUE")

import torch  # noqa: E402
from pearl_gemm import noisy_gemm  # noqa: E402
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator  # noqa: E402
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import (  # noqa: E402
    KERNEL_CONFIGS,
)


def parse_shape(s):
    m, n, k = s.split("x")
    return int(m), int(n), int(k)


def run_once(tg, gp, skip_denoising=False):
    noisy_gemm(
        A=tg.A,
        B=tg.B,
        EAL=tg.EAL,
        EAL_fp16=tg.EAL_fp16,
        EAR_R_major=tg.EAR_R_major,
        EBL_R_major=tg.EBL_R_major,
        EAR_K_major=tg.EAR_K_major,
        EBL_K_major=tg.EBL_K_major,
        EBR=tg.EBR,
        EBR_fp16=tg.EBR_fp16,
        AxEBL_fp16=tg.AxEBL_fp16,
        EARxBpEB_fp16=tg.EARxBpEB_fp16,
        ApEA=tg.ApEA,
        BpEB=tg.BpEB,
        A_scales=tg.A_scales,
        B_scales=tg.B_scales,
        C=tg.C,
        host_signal_header_pinned=tg.host_signal_header_pinned,
        host_signal_sync=tg.host_signal_sync,
        AxEBL_int32=tg.AxEBL_int32,
        EARxBpEB_int32=tg.EARxBpEB_int32,
        tile_size_m=gp.tile_size_m,
        tile_size_n=gp.tile_size_n,
        tile_size_k=gp.tile_size_k,
        pipeline_stages=gp.pipeline_stages,
        cluster_size_m=gp.cluster_size_m,
        cluster_size_n=gp.cluster_size_n,
        swizzle=gp.swizzle,
        swizzle_n_maj=gp.swizzle_n_maj,
        tile_size_m_noising_A=gp.tile_size_m_noising_A,
        tile_size_n_noising_B=gp.tile_size_n_noising_B,
        tile_size_k_noising_A=gp.tile_size_k_noising_A,
        tile_size_k_noising_B=gp.tile_size_k_noising_B,
        k_blocks_per_split_noising_A=gp.k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B=gp.k_blocks_per_split_noising_B,
        run_noising_A=True,
        run_noising_B=True,
        skip_reduction=gp.skip_reduction,
        skip_denoising=skip_denoising,
        pow_target=tg.pow_target,
        pow_key=tg.pow_key,
    )


def bench_one(m, n, k, mc, warmup, iters, swizzle, swizzle_n_maj,
              denoise_dtype, skip_denoising, pipeline_stages):
    gp = GEMMParam(
        m,
        n,
        k,
        skip_noising_a=False,
        skip_noising_b=False,
        EARxBpEB_type_noising=denoise_dtype,
        AxEBL_type_noising=denoise_dtype,
        matmul_config=mc,
        use_variable_scales=False,
    )
    tg = GemmTensorGenerator(gp)
    tg.generate()

    # warmup
    for _ in range(warmup):
        run_once(tg, gp, skip_denoising=skip_denoising)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once(tg, gp, skip_denoising=skip_denoising)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    tops = (2.0 * m * n * k) / (statistics.median(times_ms) / 1000.0) / 1e12
    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "tops": tops,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="8192x6144x4096")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="/work/_sm120_runtime_swift.csv")
    args = ap.parse_args()

    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    m, n, k = parse_shape(args.shape)
    print(f"Shape: {m}x{n}x{k}, warmup={args.warmup}, iters={args.iters}\n")

    mc = next(c for c in KERNEL_CONFIGS.matmul_kernels
              if c.tile_size_m == 64 and c.tile_size_n == 128 and c.tile_size_k == 64 and c.R == 128 and c.pipeline_stages == 2)

    rows = []
    for swizzle in [None, 0, 4, 8, 16, 32, 64, 128]:
        for swizzle_n_maj in [True, False]:
            for denoise_dtype in ["fp16", "int32"]:
                for skip_denoising in [False, True]:
                    for pipeline_stages in [None, 1, 2]:
                        tag = (
                            f"sw={swizzle} nmaj={swizzle_n_maj} dtype={denoise_dtype} "
                            f"skipd={skip_denoising} stages={pipeline_stages}"
                        )
                        try:
                            r = bench_one(
                                m, n, k, mc, args.warmup, args.iters,
                                swizzle=swizzle,
                                swizzle_n_maj=swizzle_n_maj,
                                denoise_dtype=denoise_dtype,
                                skip_denoising=skip_denoising,
                                pipeline_stages=pipeline_stages,
                            )
                            print(
                                f"{tag}: median={r['median_ms']:.4f} ms  "
                                f"min={r['min_ms']:.4f} ms  TOPS={r['tops']:.2f}"
                            )
                            rows.append({
                                "swizzle": swizzle,
                                "swizzle_n_maj": swizzle_n_maj,
                                "denoise_dtype": denoise_dtype,
                                "skip_denoising": skip_denoising,
                                "pipeline_stages": pipeline_stages,
                                **r,
                            })
                        except Exception as e:
                            print(f"{tag}: ERROR {type(e).__name__}: {str(e)[:80]}")

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")
        rows.sort(key=lambda x: (x["median_ms"], -x["tops"]))
        print(f"\nBest:")
        for r in rows[:10]:
            print(
                f"  sw={r['swizzle']} nmaj={r['swizzle_n_maj']} dtype={r['denoise_dtype']} "
                f"skipd={r['skip_denoising']} stages={r['pipeline_stages']} -> "
                f"{r['median_ms']:.4f} ms / {r['tops']:.2f} TOPS"
            )


if __name__ == "__main__":
    main()

"""SM120 GEMM benchmark on RTX 5090.

Compares the existing 64x128x64 / 128x128x64 tiles against the new wider-N
tiles (64x256x64, 128x256x64) added in perf/sm120-optimization.

Run inside the vllm_miner_5090:sm120-opt container:
  python3 bench_sm120_opt.py --shape 8192x6144x4096 --iters 20
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


def bench_one(m, n, k, mc, warmup, iters, denoise_type="int32"):
    gp = GEMMParam(
        m,
        n,
        k,
        skip_noising_a=False,
        skip_noising_b=False,
        EARxBpEB_type_noising=denoise_type,
        AxEBL_type_noising=denoise_type,
        matmul_config=mc,
        use_variable_scales=False,
    )
    tg = GemmTensorGenerator(gp)
    tg.generate()

    # warmup
    for _ in range(warmup):
        run_once(tg, gp)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once(tg, gp)
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
    ap.add_argument("--out", default="/work/_sm120_opt_results.csv")
    args = ap.parse_args()

    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    m, n, k = parse_shape(args.shape)
    print(f"Shape: {m}x{n}x{k}, warmup={args.warmup}, iters={args.iters}\n")

    rows = []
    print(
        f"{'tile':>18} {'R':>4} {'stages':>6}  "
        f"{'median ms':>10} {'min ms':>10} {'TOPS':>8}"
    )
    print("-" * 70)
    for mc in sorted(
        KERNEL_CONFIGS.matmul_kernels,
        key=lambda c: (c.tile_size_m, c.tile_size_n, c.tile_size_k, c.R, c.pipeline_stages),
    ):
        tag = f"{mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k}"
        try:
            r = bench_one(m, n, k, mc, args.warmup, args.iters)
            print(
                f"{tag:>18} {mc.R:>4} {mc.pipeline_stages:>6}  "
                f"{r['median_ms']:>10.4f} {r['min_ms']:>10.4f} {r['tops']:>8.2f}"
            )
            rows.append({
                "tile_m": mc.tile_size_m, "tile_n": mc.tile_size_n, "tile_k": mc.tile_size_k,
                "R": mc.R, "stages": mc.pipeline_stages, **r,
            })
        except Exception as e:
            print(f"{tag:>18} {mc.R:>4} {mc.pipeline_stages:>6}  ERROR: {type(e).__name__}: {str(e)[:60]}")

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")
        rows.sort(key=lambda x: x["median_ms"])
        print(f"\nBest: {rows[0]['tile_m']}x{rows[0]['tile_n']}x{rows[0]['tile_k']} "
              f"R{rows[0]['R']} st{rows[0]['stages']} -> "
              f"{rows[0]['median_ms']:.4f} ms / {rows[0]['tops']:.2f} TOPS")


if __name__ == "__main__":
    main()

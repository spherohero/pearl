"""Focused benchmark: noising B bN=64 vs bN=128 on same gemm config.
Runs timed launches of pearl_gemm.noisy_gemm with skip_denoising=True
to isolate the matmul + noising throughput (no denoise epilogue cost).
"""
import time, statistics
import torch
from pearl_gemm import noisy_gemm

BATCH = 30
WARMUP = 3

def bench_case(label, tile_size_n_noising_B):
    device = "cuda"
    m, n, k = 8192, 6144, 4096
    r = 64  # Match the R=64 variant we compiled for bN=128
    
    torch.cuda.synchronize()
    A = torch.randint(-64, 64, (m, k), dtype=torch.int8, device=device)
    B = torch.randint(-64, 64, (n, k), dtype=torch.int8, device=device)
    A_scales = torch.ones(m, dtype=torch.float32, device=device)
    B_scales = torch.ones(n, dtype=torch.float32, device=device)
    C = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    
    # Noise matrices
    EAL = torch.randint(-64, 64, (m, r), dtype=torch.int8, device=device)
    EBR = torch.randint(-64, 64, (n, r), dtype=torch.int8, device=device)
    EAR_R_major = torch.randint(-1, 1, (k, r), dtype=torch.int8, device=device)
    EBL_R_major = torch.randint(-1, 1, (k, r), dtype=torch.int8, device=device)
    EAR_K_major = EAR_R_major.t().contiguous()
    EBL_K_major = EBL_R_major.t().contiguous()
    
    # Denoising outputs
    AxEBL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EARxBpEB_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
    
    # Host signal (pinned memory for PoW extraction) — 640 bytes required
    host_signal_header_pinned = torch.empty(640, dtype=torch.int8, device="cpu", pin_memory=True)
    host_signal_sync = torch.zeros(8, dtype=torch.int8, device="cpu")
    
    # PoW parameters
    pow_target = torch.zeros(8, dtype=torch.uint32, device=device)
    pow_key = torch.zeros(8, dtype=torch.uint32, device=device)
    
    # Noised outputs (scratch)
    ApEA = torch.empty((m, k), dtype=torch.int8, device=device)
    BpEB = torch.empty((n, k), dtype=torch.int8, device=device)
    
    def run():
        noisy_gemm(
            A, B,
            EAL, AxEBL_fp16,  # EAL, EAL_fp16
            EBR, EARxBpEB_fp16,  # EBR, EBR_fp16
            EAR_R_major, EBL_R_major,
            EAR_K_major, EBL_K_major,
            AxEBL_fp16, EARxBpEB_fp16,
            ApEA, BpEB,
            A_scales, B_scales, C,
            host_signal_header_pinned, host_signal_sync,
            pow_target, pow_key,
            tile_size_m=64,
            tile_size_n=128,
            tile_size_k=64,
            pipeline_stages=2,
            tile_size_m_noising_A=64,
            tile_size_n_noising_B=tile_size_n_noising_B,
            tile_size_k_noising_A=64,
            tile_size_k_noising_B=64,
            pipeline_stages_noising_A=2,
            pipeline_stages_noising_B=2,
            run_noising_A=False,  # Skip noising A for now (not being retiled)
            run_noising_B=True,
            skip_denoising=True,  # Skip denoise for throughput measurement
        )
    
    # Warmup
    for _ in range(WARMUP):
        try:
            run()
        except Exception as e:
            return {"label": label, "error": str(e)[:150]}
    
    torch.cuda.synchronize()
    times = []
    
    # Timed runs
    for _ in range(BATCH):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            run()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        except Exception as e:
            return {"label": label, "error": str(e)[:150]}
    
    med = statistics.median(times)
    avg = statistics.mean(times)
    mn = min(times)
    mx = max(times)
    
    # TOPS: 2 ops per int8 matmul element
    tops = (2 * m * n * k * 1e-9) / med
    
    return {
        "label": label,
        "median_ms": med * 1000,
        "avg_ms": avg * 1000,
        "min_ms": mn * 1000,
        "max_ms": mx * 1000,
        "tops": tops,
    }

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Noising B Retile Benchmark: bN=64 vs bN=128                  ║")
    print("║  Shape: 8192×6144×4096 matmul, 64×128×64 tile, R=64          ║")
    print("║  Config: skip_denoising=True (measure matmul + noising only)  ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    res64 = bench_case("noisingB_bN=64", tile_size_n_noising_B=64)
    res128 = bench_case("noisingB_bN=128_R64", tile_size_n_noising_B=128)
    
    print("\n📊 Results:")
    print("─" * 80)
    print(f"{'Config':30s} {'Median (ms)':>12s} {'Avg (ms)':>12s} {'Range (ms)':>15s} {'TOPS':>10s}")
    print("─" * 80)
    
    for r in [res64, res128]:
        if "error" in r:
            print(f"{r['label']:30s} ERROR: {r['error']}")
        else:
            rng = f"{r['min_ms']:.2f}–{r['max_ms']:.2f}"
            print(f"{r['label']:30s} {r['median_ms']:12.2f} {r['avg_ms']:12.2f} {rng:>15s} {r['tops']:10.1f}")
    
    print("─" * 80)
    
    # Summary
    if "error" not in res64 and "error" not in res128:
        delta = ((res128["tops"] - res64["tops"]) / res64["tops"]) * 100
        winner = "bN=128" if delta > 0 else "bN=64"
        print(f"\n✓ Winner: {winner:10s} (+{abs(delta):.1f}% vs baseline)")
    else:
        print("\n✗ Benchmark incomplete due to errors")

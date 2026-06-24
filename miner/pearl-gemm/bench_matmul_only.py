"""Simpler test: matmul only with skip_denoising=True."""
import time, statistics
import torch
from pearl_gemm import noisy_gemm

BATCH = 30
WARMUP = 3

device = "cuda"
m, n, k = 8192, 6144, 4096
r = 64

torch.cuda.synchronize()
A = torch.randint(-64, 64, (m, k), dtype=torch.int8, device=device)
B = torch.randint(-64, 64, (n, k), dtype=torch.int8, device=device)
A_scales = torch.ones(m, dtype=torch.float32, device=device)
B_scales = torch.ones(n, dtype=torch.float32, device=device)
C = torch.empty((m, n), dtype=torch.bfloat16, device=device)

# Noise matrices (dummy)
EAL = torch.randint(-64, 64, (m, r), dtype=torch.int8, device=device)
EBR = torch.randint(-64, 64, (n, r), dtype=torch.int8, device=device)
EAR_R_major = torch.randint(-1, 1, (k, r), dtype=torch.int8, device=device)
EBL_R_major = torch.randint(-1, 1, (k, r), dtype=torch.int8, device=device)
EAR_K_major = EAR_R_major.t().contiguous()
EBL_K_major = EBL_R_major.t().contiguous()

# Denoising outputs
AxEBL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
EARxBpEB_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)

# Host signal
host_signal_header_pinned = torch.empty(640, dtype=torch.int8, device="cpu", pin_memory=True)
host_signal_sync = torch.zeros(8, dtype=torch.int8, device="cpu")

# PoW
pow_target = torch.zeros(8, dtype=torch.uint32, device=device)
pow_key = torch.zeros(8, dtype=torch.uint32, device=device)

# Noised outputs (scratch)
ApEA = torch.empty((m, k), dtype=torch.int8, device=device)
BpEB = torch.empty((n, k), dtype=torch.int8, device=device)

def run():
    noisy_gemm(
        A, B,
        EAL, AxEBL_fp16,
        EBR, EARxBpEB_fp16,
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
        run_noising_A=False,
        run_noising_B=False,  # Skip both noising kernels
        skip_denoising=True,
    )

# Warmup
for _ in range(WARMUP):
    run()

torch.cuda.synchronize()
times = []

# Timed runs
for _ in range(BATCH):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

med = statistics.median(times)
avg = statistics.mean(times)
mn = min(times)
mx = max(times)
tops = (2 * m * n * k * 1e-9) / med

print(f"Matmul only (no noising): {med*1000:.2f} ms median, {tops:.1f} TOPS")
print(f"  Range: {mn*1000:.2f}–{mx*1000:.2f} ms")

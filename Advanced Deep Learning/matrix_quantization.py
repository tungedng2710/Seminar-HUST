import torch
import time

def test_matmul(x_fp32, y_fp32, dtype, device='cuda'):
    """
    Converts x_fp32, y_fp32 to the specified dtype and device,
    performs matmul, and returns the result along with elapsed time.
    """
    x_cast = x_fp32.to(dtype=dtype, device=device)
    y_cast = y_fp32.to(dtype=dtype, device=device)
    
    torch.cuda.synchronize(device) if device.startswith('cuda') else None
    start = time.time()
    # Some hardware supports direct matmul in half/bfloat16; int8 is trickier.
    if dtype == torch.int8:
        # Naive usage: cast back to float for the multiply.
        # (Real INT8 workflows are more complex and typically use quantized ops.)
        out = torch.matmul(x_cast.float(), y_cast.float())
    else:
        out = torch.matmul(x_cast, y_cast)
    torch.cuda.synchronize(device) if device.startswith('cuda') else None
    elapsed = time.time() - start
    
    return out, elapsed

# Choose your device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create random FP32 matrices
size = 1024  # change size if you want a bigger/smaller test
x_fp32 = torch.randn(size, size, dtype=torch.float32)
y_fp32 = torch.randn(size, size, dtype=torch.float32)

# Move FP32 data to device once (for fairer comparisons)
x_fp32 = x_fp32.to(device)
y_fp32 = y_fp32.to(device)

# Baseline: FP32
res_fp32, time_fp32 = test_matmul(x_fp32, y_fp32, torch.float32, device)
print(f"[FP32 ] time: {time_fp32:.6f} s")

# FP16
res_fp16, time_fp16 = test_matmul(x_fp32, y_fp32, torch.float16, device)
err_fp16 = (res_fp32 - res_fp16.float()).abs().mean().item()
print(f"[FP16 ] time: {time_fp16:.6f} s | mean abs diff vs FP32: {err_fp16}")

# BF16
res_bf16, time_bf16 = test_matmul(x_fp32, y_fp32, torch.bfloat16, device)
err_bf16 = (res_fp32 - res_bf16.float()).abs().mean().item()
print(f"[BF16 ] time: {time_bf16:.6f} s | mean abs diff vs FP32: {err_bf16}")

# INT8 (naively cast to int8, then multiply in float)openmmlab
res_int8, time_int8 = test_matmul(x_fp32, y_fp32, torch.int8, device)
err_int8 = (res_fp32 - res_int8).abs().mean().item()
print(f"[INT8 ] time: {time_int8:.6f} s | mean abs diff vs FP32: {err_int8}")
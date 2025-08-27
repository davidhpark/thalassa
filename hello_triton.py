import time
import torch, triton
import triton.language as tl

@triton.jit
def square_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = x * x
    tl.store(y_ptr + offs, y, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    square_kernel[grid](x, y, n, BLOCK_SIZE=1024)
    return y

def bench(fn, x, iters=200, warmup=50):
    for _ in range(warmup):
        fn(x); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms

if __name__ == "__main__":
    x = torch.randn(1_000_000, device='cuda', dtype=torch.float32)
    ms_torch = bench(lambda t: t*t, x)
    ms_triton = bench(square, x)
    print(f"torch mul: {ms_torch:.3f} ms")
    print(f"triton sq: {ms_triton:.3f} ms")

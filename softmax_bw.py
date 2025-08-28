import time
import torch, triton
import triton.language as tl

from torch.autograd import Function


def softmax_bw_ref(y, g, dim=-1):
    # x, g: [..., N]
    d = (g * y).sum(dim=dim, keepdim=True)
    return (g - d) * y


@triton.jit
def softmax_bw_kernel(
    y_ptr, g_ptr, dx_ptr,
    M, N,
    sym, syn, sgm, sgn, sdxm, sdxn,
    BLOCK_N: tl.constexpr
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    row_y = y_ptr + m * sym
    row_g = g_ptr + m * sgm
    row_dx = dx_ptr + m * sdxm

    dot = tl.zeros([1], tl.float32)
    for n in range(0, N, BLOCK_N):
        idx = n + offs
        mask = idx < N
        y = tl.load(row_y + idx * syn, mask=mask, other=0.0)
        g = tl.load(row_g + idx * sgn, mask=mask, other=0.0)
        dot += tl.sum(g * y, axis=0)

    for n in range(0, N, BLOCK_N):
        idx = n + offs
        mask = idx < N
        y = tl.load(row_y + idx * syn, mask=mask, other=0.0)
        g = tl.load(row_g + idx * sgn, mask=mask, other=0.0)
        dx = (g - dot) * y
        tl.store(row_dx + idx * sdxn, dx, mask=mask)


def softmax_bw_triton(y, g, BLOCK_N=1024):
    assert y.is_cuda and g.is_cuda and y.ndim == 2 and g.shape == y.shape
    M, N = y.shape
    dx = torch.empty_like(y)
    grid = lambda meta: (M,)
    softmax_bw_kernel[grid](
        y, g, dx,
        M, N,
        y.stride(0), y.stride(1),
        g.stride(0), g.stride(1),
        dx.stride(0), dx.stride(1),
        BLOCK_N=BLOCK_N
    )
    return dx


def softmax_bw_test():
    torch.manual_seed(0)
    for M, N in [(2, 7), (4, 16), (32, 257), (128, 4096)]:
        y = torch.randn(M, N, device="cuda", dtype=torch.float32)
        g = torch.randn(M, N, device="cuda", dtype=torch.float32)
        dx_ref = softmax_bw_ref(y, g)
        dx_triton = softmax_bw_triton(y, g)
        err = (dx_ref - dx_triton).abs().max().item()
        print(f"[{M}x{N}] max |Δ| = {err:.3e}")


def bench_bw(fn, x, g, iter=200, warmup=50):
    for _ in range(warmup):
        fn(x, g); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iter):
        fn(x, g)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iter * 1e3


if __name__ == "__main__":
    softmax_test()

    M, N = 1024, 4096
    y = torch.randn(M, N, device="cuda", dtype=torch.float32)
    g = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ms_ref = bench(softmax_bw_ref, y, g)
    ms_tri = bench(softmax_bw_triton, y, g)

    print(f"torch ref bw:     {ms_ref:.3f} ms")
    print(f"triton softmaxbw: {ms_tri:.3f} ms")

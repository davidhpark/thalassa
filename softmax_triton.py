import time
import torch, triton
import triton.language as tl


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    # x: [M, N]
    x_max = x.max(dim=1, keepdim=True).values
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=1, keepdim=True)


@triton.jit
def softmax_kernel(
        x_ptr, y_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        BLOCK_N: tl.constexpr
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_x = x_ptr + m * stride_xm
    row_y = y_ptr + m * stride_ym

    row_max = float("-inf")
    for n in range(0, N, BLOCK_N):
        idx = n + offs
        mask = idx < N
        x = tl.load(row_x + idx * stride_xn, mask=mask, other=float("-inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = 0.0
    for n in range(0, N, BLOCK_N):
        idx = n + offs
        mask = idx < N
        x = tl.load(row_x + idx * stride_xn, mask=mask, other=float("-inf"))
        ex = tl.exp(x - row_max)
        row_sum += tl.sum(ex, axis=0)

    for n in range(0, N, BLOCK_N):
        idx = n + offs
        mask = idx < N
        x = tl.load(row_x + idx * stride_xn, mask=mask, other=float("-inf"))
        ex = tl.exp(x - row_max)
        y = ex / row_sum
        tl.store(row_y + idx * stride_yn, y, mask=mask)


def softmax_triton(x: torch.Tensor, BLOCK_N=1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    grid = lambda meta: (M,)
    softmax_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=BLOCK_N
    )
    return y


def softmax_test():
    torch.manual_seed(0)
    M, N = 4, 16
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y_ref = softmax_ref(x)
    y_tri = softmax_triton(x)
    max_abs = (y_ref - y_tri).abs().max().item()
    print("max |Δ|:", max_abs)


def bench(fn, x, iter=200, warmup=50):
    for _ in range(warmup):
        fn(x); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iter):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iter * 1e3


if __name__ == "__main__":
    M, N = 1024, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ms_torch = bench(lambda t: torch.nn.functional.softmax(t, dim=-1), x)
    ms_tri = bench(lambda t: softmax_triton(t), x)
    ms_cpu = bench(softmax_ref, x)
    print(f"ms torch: {ms_torch:.3f} ms")
    print(f"ms tri: {ms_tri:.3f} ms")
    print(f"ms cpu: {ms_cpu:.3f} ms")

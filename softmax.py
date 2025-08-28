import time
import torch, triton

from torch.autograd import Function

from softmax_bw_triton import softmax_bw_triton
from softmax_fw_triton import softmax_fw_triton


class SoftmaxTriton(Function):
    @staticmethod
    def forward(ctx, x):
        y = softmax_fw_triton(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        dx = softmax_bw_triton(y, grad_out)
        return dx


def softmax_test():
    torch.manual_seed(0)
    M, N = 1024, 4096

    x0 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    x1 = x0.clone().requires_grad_(True)
    x2 = x0.clone().requires_grad_(True)

    y_triton = SoftmaxTriton.apply(x1)
    y_ref = torch.softmax(x2, dim=-1)

    f_err = (y_ref - y_triton).abs().max().item()
    print("forward max |Δ|:", f_err)

    y_triton.sum().backward()
    y_ref.sum().backward()

    b_err = (x1.grad - x2.grad).abs().max().item()
    print("backward max |Δ|:", b_err)


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3


def e2e_ref(x0):
    def run():
        x = x0.clone().requires_grad_(True)
        y = torch.softmax(x, dim=-1)
        y.sum().backward()
    return run


def e2e_triton(x0):
    def run():
        x = x0.clone().requires_grad_(True)
        y = SoftmaxTriton.apply(x)
        y.sum().backward()
    return run


if __name__ == "__main__":
    softmax_test()

    torch.manual_seed(0)
    shapes = [(256, 1024), (512, 2048), (1024, 4096), (2048, 4096)]
    for M, N in shapes:
        x0 = torch.randn(M, N, device="cuda", dtype=torch.float32)

        ms_ref = bench(e2e_ref(x0))
        ms_triton = bench(e2e_triton(x0))

        print(f"[{M}x{N}] ref: {ms_ref:.3f} ms | triton: {ms_triton:.3f} ms")


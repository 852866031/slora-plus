import torch

import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)


def rmsnorm_forward(x, weight, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # print("BLOCK_SIZE:", BLOCK_SIZE)
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # print(BLOCK_SIZE, num_warps, "block_size, numwarps")
    BLOCK_SIZE = 128 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    num_warps = 8
    # enqueue kernel
    _rms_norm_fwd_fused[(M,)](x_arg, y, weight,
                              x_arg.stride(0), N, eps,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y

def rmsnorm_backward(x: torch.Tensor, grad_out: torch.Tensor, weight: torch.Tensor, eps: float):
    D = x.shape[-1]

    # Compute stable norm
    norm_sq = x.pow(2).sum(dim=-1, keepdim=True) / D  # [N, 1]
    norm = torch.sqrt(norm_sq + eps)  # [N, 1]

    # Normalize x
    x_hat = x / norm  # [N, D]

    # Gradient of the output of RMSNorm (elementwise weight applied after norm)
    grad_x_hat = grad_out * weight  # [N, D]

    # Compute projection term
    x_dot_grad = (x_hat * grad_x_hat).sum(dim=-1, keepdim=True)  # [N, 1]

    # Backprop through normalization
    grad_x = (grad_x_hat - x_hat * x_dot_grad / D) / norm  # [N, D]

    return grad_x



def torch_rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def test_rms_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda')
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    # forward pass
    y_tri = rmsnorm_forward(x, weight, eps)
    y_ref = torch_rms_norm(x.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    return

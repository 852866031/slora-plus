import time
import torch
import triton
import triton.language as tl

from slora._kernels import dispatch_bgmv


@triton.jit
def triton_batch_lora_B(
    output,
    x,
    w,
    a_start,
    a_len,
    a_loc,
    batch_req_bins,
    a_scaling,
    qkvo_offset: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_LORA_RANK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    return


def batch_lora_forward_B(
    output,
    x,
    w,
    a_start,
    a_len,
    a_loc,
    batch_req_bins,
    qkvo_offset,
    a_scaling,
):
    #print("B", output.shape, x.shape, w.shape, a_start.shape, a_len.shape, a_loc.shape,
    #      batch_req_bins.shape, qkvo_offset, a_scaling.shape)
    NUM_TOKENS, MAX_LORA_RANK = x.shape
    NUM_TOKENS, HIDDEN = output.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(NUM_TOKENS, BLOCK_SIZE_M), triton.cdiv(HIDDEN, BLOCK_SIZE_N))
    triton_batch_lora_B[grid](output, x,
                              w,
                              a_start, a_len, 
                              a_loc, batch_req_bins, a_scaling, qkvo_offset,
                              NUM_TOKENS, HIDDEN, MAX_LORA_RANK,
                              BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

def compare_tensors(name, a, b):
    diff = (a - b).float()
    print(f"[{name}] shape={list(a.shape)}")
    print(f"  L2 error : {diff.norm():.4f}")
    print(f"  Mean |Δ|  : {diff.abs().mean().item():.6f}")
    print(f"  Max  |Δ|  : {diff.abs().max().item():.6f}")
    print()

def test_bgmv():
    H = 4096
    R = 128
    N = 128
    num_adapters = 2
    num_head = 32
    part = "A"

    if part == "A":
        x = torch.randn((N, H), dtype=torch.float16, device="cuda")
        delta_qA = torch.zeros((len(x), R), dtype=torch.float16, device="cuda")
        forward_func = dispatch_bgmv
    else:
        x = torch.randn((N, R), dtype=torch.float16, device="cuda")
        delta_qA = torch.zeros((len(x), H), dtype=torch.float16, device="cuda")
        forward_func = dispatch_bgmv

    key_buffer = torch.randn((R * 4 * num_adapters, num_head, H // num_head), dtype=torch.float16, device="cuda")
    a_len = torch.tensor([R * 4] * num_adapters, dtype=torch.long, device="cuda")
    a_start = torch.zeros_like(a_len)
    a_start[1:] = torch.cumsum(a_len[:-1], dim=0)
    a_loc = torch.arange(R * 4 * num_adapters, dtype=torch.long, device="cuda")
    a_scaling = torch.tensor([1] * num_adapters, dtype=torch.float16, device="cuda")
    batch_req_bins = torch.concat([
        torch.tensor([i] * ((N  + num_adapters - 1) // num_adapters), dtype=torch.long, device="cuda")
        for i in range(num_adapters)])
    batch_req_bins = batch_req_bins[:len(x)]

    qkvo = 1
    results = []
    for i in range(N):
        a_id = batch_req_bins[i]
        a_w = key_buffer[a_start[a_id] + qkvo * R: a_start[a_id] + (qkvo + 1) * R]
        if part == "A":
            a_w = a_w.reshape(R, H).T
        else:
            a_w = a_w.reshape(H, R).T
        results.append(x[i:i+1, :] @ a_w)
    ref = delta_qA + torch.concat(results)

    forward_func(delta_qA, x,
                 key_buffer,
                 a_start, a_len,
                 a_loc, batch_req_bins, qkvo, a_scaling)

    print("max delta:", torch.max(torch.abs(delta_qA - ref)))
    compare_tensors(part, delta_qA, ref)

    def to_test():
        #batch_lora_forward_B(delta_qA, x,
        #                     key_buffer,
        #                     a_start, a_len, 
        #                     a_loc, batch_req_bins, 0, a_scaling)

        dispatch_bgmv(delta_qA, x,
                      key_buffer,
                      a_start, a_len, 
                      a_loc, batch_req_bins, 0, a_scaling)
        #ref = x @ key_buffer[:R].reshape(-1, H).T

    # Warm up
    for _ in range(10):
        to_test()
    run_iter = 500
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        to_test()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time cost {((t2 - t1) / run_iter) * 1000:.2f} ms")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_bgmv()

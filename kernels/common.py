"""Shared Triton helpers for the Qwen3.5 INT8 engine."""
import triton
import triton.language as tl

# Activation quantisation granularity.  Every GEMM block along K must equal this
# so a single scale applies to each int8 dot product.
GROUP = 64
# Same value as a Triton constexpr (globals inside @jit must be constexpr).
GROUP_C = tl.constexpr(GROUP)


@triton.jit
def swizzle(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    """Grouped tile order: consecutive programs share rows, improving L2 reuse."""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def round_i8(v):
    """Round-to-nearest, symmetric int8 clamp of a float value."""
    r = (v + tl.where(v >= 0.0, 0.5, -0.5)).to(tl.int32)
    r = tl.minimum(tl.maximum(r, -127), 127)
    return r.to(tl.int8)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))


@triton.jit
def grid_barrier(CNT, target):
    """Grid-wide barrier. CNT is a monotonic int32 counter, target = G * (i + 1)."""
    tl.debug_barrier()
    tl.atomic_add(CNT, 1, sem="release")
    while tl.atomic_add(CNT, 0, sem="acquire") < target:
        pass
    tl.debug_barrier()


@triton.jit
def load_row_i8(P, offs, mask):
    return tl.load(P + offs, mask=mask, other=0).to(tl.int8)


def next_pow2(n):
    return 1 << max(0, (n - 1)).bit_length()

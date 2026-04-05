import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


def solve(x: torch.Tensor, y: torch.Tensor):
    assert x.device == DEVICE and y.device == DEVICE, f"Tensors must be on CUDA device, got {x.device} and {y.device}"
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

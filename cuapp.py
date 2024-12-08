from numba import cuda
import numpy as np


@cuda.jit
def add_arrays(a, b, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = a[i] + b[i]


# Set up array dimensions
N = 1000000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)
result = np.zeros(N, dtype=np.float32)

# Set up CUDA kernel launch configuration
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Launch the kernel
add_arrays[blocks_per_grid, threads_per_block](a, b, result)

# Verify the result
# np.allclose(a + b, result),
print(result, a + b, sep="\n")

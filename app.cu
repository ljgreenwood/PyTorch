#include <stdio.h>

// CUDA kernel function
__global__ void add(int *d_array, int size)
{
    // Get the global index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (idx < size)
    {
        // Add 1 to the element at this index
        d_array[idx] += 1;
        // o_array[idx] = d_array[idx] + o_array[idx];
    }
}

int main()
{
    const int SIZE = 10;
    int h_array[SIZE]; // Host array
    int *d_array;      // Device array
    // int i_array[SIZE]; // Second Host array
    // int *e_array;      // Second Device array

    // Initialize the host array
    for (int i = 0; i < SIZE; i++)
    {
        h_array[i] = i;
        // i_array[i] = 256 - i;
        
    }

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_array, SIZE * sizeof(int));

    // Copy the host array to the GPU
    cudaMemcpy(d_array, h_array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel (1 block, SIZE threads)
    add<<<1, SIZE>>>(d_array, SIZE);

    // Copy the result back to the host
    cudaMemcpy(h_array, d_array, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Results:\n");
    for (int i = 0; i < SIZE; i++)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free GPU memory
    cudaFree(d_array);

    return 0;
}

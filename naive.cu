#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel for parallel scan (prefix sum)
__global__ void scan_kernel(int* input, int* output, int n) {
    __shared__ int temp[THREADS_PER_BLOCK];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    // Load input into shared memory
    if (index < n) {
        temp[tx] = input[index];
    } else {
        temp[tx] = 0;
    }
    __syncthreads();

    // Perform scan on shared memory
    for (int stride = 1; stride <= tx; stride *= 2) {
        int temp_val = 0;
        if (tx >= stride) {
            temp_val = temp[tx - stride];
        }
        __syncthreads();
        temp[tx] += temp_val;
        __syncthreads();
    }

    // Write the result to output array
    if (index < n) {
        output[index] = temp[tx];
    }
}

// Function to initialize the input array
void initialize_array(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1;
    }
}

// Host function to call the CUDA kernel
void gpu_scan(int* input, int* output, int n) {
    int *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, sizeof(int) * n);
    cudaMalloc((void**)&d_output, sizeof(int) * n);

    // Copy data from host to device
    cudaMemcpy(d_input, input, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the scan kernel
    scan_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, n);

    // Copy results from device to host
    cudaMemcpy(output, d_output, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Array of different sizes to test
    int sizes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000};

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Loop over each size and measure the time taken for the scan
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];

        // Allocate memory for input and output arrays
        int* input = (int*)malloc(sizeof(int) * size);
        int* output = (int*)malloc(sizeof(int) * size);

        // Initialize input array
        initialize_array(input, size);

        // Measure the execution time
        clock_t start, end;
        start = clock();

        // Perform the scan on GPU
        gpu_scan(input, output, size);

        end = clock();

        // Calculate and print the execution time in milliseconds
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
        printf("Time taken for GPU scan of size %d: %f ms\n", size, time_taken);

        // Free host memory
        free(input);
        free(output);
    }

    return 0;
}


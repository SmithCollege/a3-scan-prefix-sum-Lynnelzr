#include <stdio.h>
#include <cuda.h>

__global__ void naivePrefixScan(int* input, int* output, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= size) return;  // Prevent out-of-bounds access

    int value = 0;

    // Accumulate all previous values
    for (int j = 0; j <= index; j++) {
        value += input[j];
    }

    output[index] = value;
}

__global__ void recursivePrefixScan(int* input, int* output, int size) {
    extern __shared__ int temp[];  // Shared memory declaration
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= size) return;

    // Load input into shared memory
    temp[threadIdx.x] = input[index];
    __syncthreads();

    // Recursive doubling step
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int tempVal = 0;
        if (threadIdx.x >= stride) {
            tempVal = temp[threadIdx.x - stride];
        }
        __syncthreads();
        temp[threadIdx.x] += tempVal;
        __syncthreads();
    }

    // Write the result back to global memory
    output[index] = temp[threadIdx.x];
}

void perform_scan(int size, bool use_naive) {
    int* input;
    int* output;

    cudaMallocManaged(&input, size * sizeof(int));
    cudaMallocManaged(&output, size * sizeof(int));

    // Initialize input array
    for (int i = 0; i < size; i++) {
        input[i] = 1;
    }

    // Adjust threadsPerBlock and blocksPerGrid based on size
    int threadsPerBlock = (size < 256) ? size : 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Use CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start timing

    // Launch the appropriate kernel
    if (use_naive) {
        naivePrefixScan<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    } else {
        recursivePrefixScan<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(input, output, size);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);  // Stop timing
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  // Calculate elapsed time

    // Output the first 10 results to verify correctness
    printf("First 10 results for size %d: ", size);
    for (int i = 0; i < 10; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("Time taken for size %d (GPU): %f ms\n", size, milliseconds);

    // Free memory
    cudaFree(input);
    cudaFree(output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Array of different sizes to test
    int sizes[] = {100, 1000, 10000, 100000, 1000000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Loop over each size and perform the scan
    for (int i = 0; i < num_sizes; i++) {
        printf("Using naive prefix scan for size %d:\n", sizes[i]);
        perform_scan(sizes[i], true);  // true for naive prefix scan

        printf("Using recursive doubling prefix scan for size %d:\n", sizes[i]);
        perform_scan(sizes[i], false);  // false for recursive prefix scan
    }

    return 0;
}


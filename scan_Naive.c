#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RADIUS 2  // Example radius

// Naive stencil implementation
void stencil_naive(int* in, int* out, int size) {
    for (int i = 0; i < size; ++i) {
        int result = 0;

        // Apply the stencil
        for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
            int index = i + offset;
            
            // Check if the index is within bounds of the array
            if (index >= 0 && index < size) {
                result += in[index];
            }
        }

        // Store the result in the output array
        out[i] = result;
    }
}

int main() {
    // Array sizes to test for scaling analysis
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000}; // Different input sizes
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);

    // Loop through different array sizes for performance measurement
    for (int test = 0; test < num_tests; ++test) {
        int size = sizes[test];

        // Allocate memory for the input and output arrays
        int* input = (int*)malloc(size * sizeof(int));
        int* output = (int*)malloc(size * sizeof(int));

        // Initialize the input array with all 1s
        for (int i = 0; i < size; ++i) {
            input[i] = 1;
        }

        // Timing the stencil operation
        clock_t start = clock();
        stencil_naive(input, output, size);  // Perform the stencil computation
        clock_t end = clock();

        // Calculate the time taken in seconds
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Size: %d, Time taken: %f seconds\n", size, time_taken);

        // Free allocated memory
        free(input);
        free(output);
    }

    return 0;
}


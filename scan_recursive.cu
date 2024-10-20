#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RADIUS 2  // Example radius

// Recursive doubling stencil function
void stencil_recursive_doubling(int* in, int* out, int size) {
    int* temp = (int*)malloc(size * sizeof(int));
    
    // Copy input to temp array
    for (int i = 0; i < size; ++i) {
        temp[i] = in[i];
    }

    // Perform recursive doubling
    for (int step = 1; step <= RADIUS; step *= 2) {
        for (int i = 0; i < size; ++i) {
            int result = temp[i];
            // Combine values at distance `step` from both sides
            if (i - step >= 0) {
                result += temp[i - step];
            }
            if (i + step < size) {
                result += temp[i + step];
            }
            out[i] = result;
        }
        
        // Copy output to temp for the next iteration
        for (int i = 0; i < size; ++i) {
            temp[i] = out[i];
        }
    }

    // Free allocated memory
    free(temp);
}

int main() {
    // Array sizes to test for scaling analysis
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000}; // Different input sizes
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);

    for (int test = 0; test < num_tests; ++test) {
        int size = sizes[test];

        int* input = (int*)malloc(size * sizeof(int));
        int* output = (int*)malloc(size * sizeof(int));

        // Initialize the input array with all 1s
        for (int i = 0; i < size; ++i) {
            input[i] = 1;
        }

        // Timing the stencil operation
        clock_t start = clock();
        stencil_recursive_doubling(input, output, size);
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


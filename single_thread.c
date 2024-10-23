#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void perform_scan(int size) {
    // Allocate memory
    int* input = malloc(sizeof(int) * size);
    int* output = malloc(sizeof(int) * size);

    // Check if memory allocation was successful
    if (input == NULL || output == NULL) {
        printf("Memory allocation failed for size %d\n", size);
        return;
    }

    // Initialize inputs
    for (int i = 0; i < size; i++) {
        input[i] = 1;
    }

    // Timing start
    clock_t start = clock();

    // Scan
    output[0] = input[0];
    for (int i = 1; i < size; i++) {
        output[i] = output[i - 1] + input[i];
    }

    // Timing end
    clock_t end = clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds

    printf("Time taken for size %d: %f ms\n", size, total_time);

    // Free memory
    free(input);
    free(output);
}

int main() {
    // Array of different sizes to test
    int sizes[] = { 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Loop over each size and perform the scan
    for (int i = 0; i < num_sizes; i++) {
        perform_scan(sizes[i]);
    }

    return 0;
}


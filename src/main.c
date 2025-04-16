#include <stdio.h>
#include "conv2d.h"

int main() {
    // Input: 3x3 matrix
    float input[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Kernel: 2x2
    float kernel[4] = {
        1, 0,
        0, -1
    };

    int in_width = 3, in_height = 3;
    int kernel_width = 2, kernel_height = 2;

    // Output size: (3 - 2 + 1) x (3 - 2 + 1) = 2 x 2
    float output[4];

    conv2d(input, in_width, in_height, kernel, kernel_width, kernel_height, output);

    printf("Output:\n");
    for (int i = 0; i < in_height - kernel_height + 1; ++i) {
        for (int j = 0; j < in_width - kernel_width + 1; ++j) {
            printf("%f ", output[i * (in_width - kernel_width + 1) + j]);
        }
        printf("\n");
    }

    return 0;
}


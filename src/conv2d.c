#include <string.h>
#include <math.h>
//#include <fftw3-mpi.h>
#include <fftw3.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "conv_check.h" 

int method_used = -1; // global variable to track the method used

#define MAX_BLOCK_SIZE 64  

// calculate the next power of 2
#define NEXT_POWER_OF_2(x) ({ \
    int p = 1; \
    while (p < (x)) p <<= 1; \
    p; \
})

#define IS_POWER_OF_2(n) ((n > 0) && ((n & (n - 1)) == 0))

// print a matrix
void print_matrix(float* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

// simplest form (Naive direct)
void naive_conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output) {
    int out_width = in_width - kernel_width + 1;
    int out_height = in_height - kernel_height + 1;

    #pragma omp parallel for collapse(2)  
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_height; ++ki) {
                for (int kj = 0; kj < kernel_width; ++kj) {
                    sum += in[(i + ki) * in_width + (j + kj)] * kernel[ki * kernel_width + kj];
                }
            }
            output[i*out_width+j] = sum;
        
    }
}
}

// block-wise convolution with optimizations
void block_conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output) {
    int out_width = in_width - kernel_width + 1;
    int out_height = in_height - kernel_height + 1;

    // define block size
    int block_size = NEXT_POWER_OF_2(fmin(64, fmax(16, in_width/8)));

    // Reduce number of threads for very small kernels
    if (kernel_width <= 4 && kernel_height <= 4) {
        omp_set_num_threads(1);
    }

    #pragma omp parallel for collapse(2)  // Parallelize over blocks
    for (int i = 0; i < out_height; i += block_size) {
        for (int j = 0; j < out_width; j += block_size) {
            int i_end = fmin(i + block_size, out_height);
            int j_end = fmin(j + block_size, out_width);

            // process each block
            for (int bi = i; bi < i_end; ++bi) {
                for (int bj = j; bj < j_end; ++bj) {
                    float sum = 0.0f;

                    
                    #pragma omp simd
                    for (int ki = 0; ki < kernel_height; ki++) {
                        for (int kj = 0; kj < kernel_width; kj++) {
                            int ii = bi + ki;
                            int jj = bj + kj;
                            sum += in[ii * in_width + jj] * kernel[ki * kernel_width + kj];
                        }
                    }
                    output[bi * out_width + bj] = sum;
                }
            }
        }
    }

    // reset thread number
    omp_set_num_threads(omp_get_max_threads());
}

// 2D convolution using FFT 
void fft_conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output) {
    int out_width = in_width - kernel_width +1;
    int out_height = in_height - kernel_height +1;

    int fft_width = in_width+kernel_width-1;
    int fft_height = in_height+kernel_height-1;
    
    // allocate FFTW input, kernel, and output 
    fftwf_complex* fftw_in = fftwf_alloc_complex(fft_height * (fft_width / 2 + 1));
    fftwf_complex* fftw_kernel = fftwf_alloc_complex(fft_height * (fft_width / 2 + 1));
    fftwf_complex* fftw_out = fftwf_alloc_complex(fft_height * (fft_width / 2 + 1));
    float* padded_in = (float*)calloc(fft_width * fft_height, sizeof(float));
    float* padded_kernel = (float*)calloc(fft_width * fft_height, sizeof(float));
    float* output_array = (float*)malloc(fft_width * fft_height * sizeof(float));

    // check for memory allocation 
    if (!fftw_in || !fftw_kernel || !fftw_out || !padded_in || !padded_kernel || !output_array) {
        fprintf(stderr, "memory allocation failed.\n");
        exit(1);
    }

    // reset the arrays
    memset(padded_in, 0, fft_width*fft_height* sizeof(float));
    memset(padded_kernel, 0, fft_width*fft_height * sizeof(float));

    // copy input into padded input array
    #pragma omp parallel for
    for (int i = 0; i < in_height; i++) {
        for (int j = 0; j < in_width; j++) {
            padded_in[i * fft_width + j] = in[i * in_width + j];
        }
    }

    // flip kernel and copy into padded kernel array
    #pragma omp parallel for
    for (int i = 0; i < kernel_height; ++i) {
        for (int j = 0; j < kernel_width; ++j) {
            padded_kernel[i*fft_width+j] = kernel[(kernel_height - 1 - i)*kernel_width + (kernel_width - 1 - j)];
        }
    }

    // create FFTW plans
    fftwf_plan plan_in = fftwf_plan_dft_r2c_2d(fft_height, fft_width, padded_in, fftw_in, FFTW_ESTIMATE);
    fftwf_plan plan_kernel = fftwf_plan_dft_r2c_2d(fft_height, fft_width, padded_kernel, fftw_kernel, FFTW_ESTIMATE);

    // execute FFT on input and kernel
    fftwf_execute(plan_in);
    fftwf_execute(plan_kernel);

    // elements multiplication in frequency domain 
    #pragma omp parallel for
    for (int i = 0; i < fft_height*(fft_width / 2 + 1); i++) {
        float real_part = fftw_in[i][0]*fftw_kernel[i][0] - fftw_in[i][1] * fftw_kernel[i][1];
        float imag_part = fftw_in[i][0]*fftw_kernel[i][1] + fftw_in[i][1] * fftw_kernel[i][0];
        fftw_out[i][0] = real_part;
        fftw_out[i][1] = imag_part;
    }

    // execute inverse FFT
    fftwf_plan plan_out = fftwf_plan_dft_c2r_2d(fft_height, fft_width, fftw_out, output_array, FFTW_ESTIMATE);
    fftwf_execute(plan_out);

    // normalize the output 
    float normalization_factor = 1.0f / (fft_width*fft_height);
    #pragma omp parallel for
    for (int i = 0; i < fft_width * fft_height; ++i) {
        output_array[i] *= normalization_factor;
    }

    // extract the suitable elemnts of the output
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < out_height; ++i) {
        for (int j = 0; j < out_width; ++j) {
            output[i*out_width +j] = output_array[(i + kernel_height - 1)*fft_width+(j+ 			 kernel_width - 1)];
        }
    }

    // free alocated memory
    fftwf_destroy_plan(plan_in);
    fftwf_destroy_plan(plan_kernel);
    fftwf_destroy_plan(plan_out);
    fftwf_free(fftw_in);
    fftwf_free(fftw_kernel);
    fftwf_free(fftw_out);
    free(padded_in);
    free(padded_kernel);
    free(output_array);
}


// main function to select method 

void conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output) {
    int input_size = in_width*in_height;
    int kernel_size = kernel_width*kernel_height;

    double C_naive = 0.1;    // naive constant
    double C_block = 0.025;   // block-wise constant
    double C_fft = 0.15;     // fft constant
    double O_block = 0.001;  // block overhead

    // calculate time estimates for each method
    double T_naive = C_naive*input_size* kernel_size;
    double T_block = C_block*input_size*kernel_size + O_block* input_size;
    double T_fft = C_fft*input_size*log2(input_size);




    // if FFT is power-of-2 
    if (IS_POWER_OF_2(in_width) && IS_POWER_OF_2(in_height) && kernel_size >= 0.1 * input_size) {
        printf("FFT chosen due to power-of-2 input size.\n");
        fft_conv2d(in, in_width, in_height, kernel, kernel_width, kernel_height, output);
        method_used = 2; // FFT
    }

    //  choose between one of 3 methods 
    else if ((T_naive < T_block && T_naive < T_fft) || kernel_size < 7 * 7) {
        printf("naive.\n");
        naive_conv2d(in, in_width, in_height, kernel, kernel_width, kernel_height, output);
        method_used = 0; // naive method
    } else if (T_block < T_fft) {
        printf("block.\n");
        block_conv2d(in, in_width, in_height, kernel, kernel_width, kernel_height, output);
        method_used = 1; // block-wise method
    } else {
        printf("fft.\n");
        fft_conv2d(in, in_width, in_height, kernel, kernel_width, kernel_height, output);
        method_used = 2; // Ffft method
    }
}






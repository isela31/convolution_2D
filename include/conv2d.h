#ifndef CONV_CHECK_H
#define CONV_CHECK_H
extern int method_used;

void print_matrix(float* matrix, int width, int height);
void block_conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output);
void fft_conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output);
void conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output);


void naive_conv2d(float* in, int in_width, int in_height, float* kernel, int kernel_width, int kernel_height, float* output) ; 

#endif  //CONV_CHECK_H



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void SimpleSumReductionKernel(float* input, float* output) {
	unsigned int i = 2 * threadIdx.x;
	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		if (threadIdx.x % stride == 0) {
			input[i] += input[i + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		*output = input[0];
	}
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
	unsigned int i = threadIdx.x;
	for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
		if (threadIdx.x < stride) {
			input[i] += input[i + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		*output = input[0];
	}
}

__global__ void ConvergentSumReductionReverseKernel(float* input, float* output) {
	unsigned int i = 2*blockDim.x - threadIdx.x - 1;
	for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
		if (threadIdx.x < stride) {
			input[i] += input[i - stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == blockDim.x-1) {
		*output = input[2*blockDim.x-1];
	}
}

#define BLOCK_DIM 16
#define COARSE_FACTOR 4

__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;
	float sum = input[i];
	for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
		sum += input[i + tile * BLOCK_DIM];
	}
	input_s[t] = sum;
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			input_s[t] += input_s[t + stride];
		}
	}
	if (t == 0) {
		atomicAdd(output, input_s[0]);
	}
}

__global__ void CoarsenedSumReduction2Kernel(float* input, float* output, int N) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;
	float sum = input[i];
	for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
		if ((i + tile * BLOCK_DIM) < N) {
			sum += input[i + tile * BLOCK_DIM];
		}
		}
	input_s[t] = sum;
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			input_s[t] += input_s[t + stride];
		}
	}
	if (t == 0) {
		atomicAdd(output, input_s[0]);
	}
}

__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;
	float max = input[i];
	for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
		if (max < input[i + tile * BLOCK_DIM]) {
			max = input[i + tile * BLOCK_DIM];
		}
	}
	input_s[t] = max;
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			if (input_s[t] < input_s[t + stride]) {
				input_s[t] = input_s[t+stride];
			}
		}
	}
	if (t == 0) {
		*output = input_s[0];
	}
}

void CoarsenedSumReduction() {
	float* input_d, * output_d;
	float input[32];
	for (int i = 0; i < 32; i++) {
		input[i] = (float)i;
	}
	float output;
	int size = 32 * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(16, 1, 1);

	cudaMalloc((void**)&input_d, size);
	cudaMalloc((void**)&output_d, sizeof(float));

	cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

	CoarsenedSumReductionKernel << <dimGrid, dimBlock >> > (input_d, output_d);

	cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);

	printf("%.2f", output);
}

void CoarsenedSumReduction2() {
	float* input_d, * output_d;
	float input[35];
	int N = 35;
	for (int i = 0; i < 35; i++) {
		input[i] = (float)i;
	}
	float output;
	int size = 35 * sizeof(float);

	dim3 dimGrid(2, 1, 1);
	dim3 dimBlock(16, 1, 1);

	cudaMalloc((void**)&input_d, size);
	cudaMalloc((void**)&output_d, sizeof(float));

	cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

	CoarsenedSumReduction2Kernel << <dimGrid, dimBlock >> > (input_d, output_d, N);

	cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);

	printf("%.2f", output);
}

void CoarsenedMaxReduction() {
	float* input_d, * output_d;
	float input[32];
	for (int i = 0; i < 32; i++) {
		input[i] = (float)i;
	}
	float output;
	int size = 32 * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(16, 1, 1);

	cudaMalloc((void**)&input_d, size);
	cudaMalloc((void**)&output_d, sizeof(float));

	cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

	CoarsenedMaxReductionKernel << <dimGrid, dimBlock >> > (input_d, output_d);

	cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);

	printf("%.2f", output);
}


void ConvergentSumReduction() {
	float* input_d, * output_d;
	float input[32];
	for (int i = 0; i < 32; i++) {
		input[i] = 1.0;
	}
	float output;
	int size = 32 * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(16, 1, 1);

	cudaMalloc((void**)&input_d, size);
	cudaMalloc((void**) & output_d, sizeof(float));

	cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

	ConvergentSumReductionReverseKernel << <dimGrid, dimBlock >> > (input_d, output_d);

	cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);

	printf("%.2f", output);
}

int main() {
	CoarsenedSumReduction2();
	return 1;
}
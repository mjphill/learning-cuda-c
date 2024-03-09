
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32
#define GRID_DIM 4

__device__ void exclusiveScan(unsigned int* bits, unsigned int N);

__global__ void radix_sort_iter_kernel(unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int N, unsigned int iter) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int key, bit;
	if (i < N) {
		key = input[i];
		bit = (key >> iter) & 1;
		bits[i] = bit;
	}
	//__syncthreads();
	exclusiveScan(bits, N);
	//__syncthreads();
	if (i < N) {
		unsigned int numOnesBefore = bits[i];
		unsigned int numOnesTotal = bits[N-1];
		unsigned int dst = (bit == 0) ? (i - numOnesBefore) : (N - numOnesTotal + numOnesBefore);
		output[dst] = key;
	}
}

__global__ void radix_sort_iter_tiling_kernel(unsigned int* input, unsigned int* output, unsigned int* buckets, unsigned int N, unsigned int iter) {
	__shared__ unsigned int out_tile[TILE_SIZE];
	__shared__ unsigned int bits[TILE_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int key, bit;
	//Identify bit value for specified iteration. Key is extracted from input to local memory
	if (i < N) {
		key = input[i];
		bit = (key >> iter) & 1;
		bits[t] = bit;
	}
	__syncthreads();
	exclusiveScan(bits, TILE_SIZE);
	__syncthreads();
	//Sort key to proper place in shared tile memoery
	if (i < N) {
		unsigned int numOnesBefore = bits[t];
		unsigned int numOnesTotal = bits[TILE_SIZE - 1];
		unsigned int dst = (bit == 0) ? (i - numOnesBefore) : (TILE_SIZE - numOnesTotal + numOnesBefore);
		out_tile[dst] = key;
		//Save bit counts to global memory, only on thread 0 for efficency with global memory access
		buckets[blockIdx.x] = TILE_SIZE - numOnesTotal;
		buckets[blockIdx.x + gridDim.x] = numOnesTotal;
	} else {
		out_tile[t] = 0;
	}
	//Exclusive scan on bit count array
	__syncthreads();
	exclusiveScan(buckets, GRID_DIM * 2);
	__syncthreads();
	//write results to global memory output
	if (i < N) {
		/*bit = (out_tile[t] >> iter) & 1;
		unsigned int numOnesBefore = bits[t];
		unsigned int numOnesTotal = bits[TILE_SIZE - 1];
		unsigned int dst = (bit == 0) ? buckets[blockIdx.x] + t : buckets[blockIdx.x + gridDim.x] + (t - (TILE_SIZE - numOnesTotal));
		output[dst + blockIdx.x * blockDim.x] = out_tile[t];
		*/
		output[i] = out_tile[t];
	}

}

__device__ void exclusiveScan(unsigned int* bits, unsigned int N) {
	unsigned int temp = bits[0];
	bits[0] = 0;
	for (unsigned int i = 1; i < N; ++i) {
		unsigned int current = bits[i];
		bits[i] = temp;
		temp += current;
	}
}

void radix_sort_iter() {
	unsigned int N = 4096;
	unsigned int input[4096];
	unsigned int output[4096];
	for (unsigned int i = 0; i < N; ++i) {
		input[i] = rand() % 10;
	}
	unsigned int *input_d, *output_d, *bits_d;
	int iter = 5;

	cudaMalloc((void**)&input_d, N * sizeof(unsigned int));
	cudaMalloc((void**)&output_d, N * sizeof(unsigned int));
	cudaMalloc((void**)&bits_d, N * sizeof(unsigned int));

	dim3 dimGrid(4, 1, 1);
	dim3 dimBlock(1024, 1, 1);

	cudaMemcpy(input_d, input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

	for (unsigned int i = 0; i < iter; ++i) {
		radix_sort_iter_kernel << <dimGrid, dimBlock >> > (input_d, output_d, bits_d, N, i);
		//cudaDeviceSynchronize();
		//cudaMemcpy(input_d, output_d, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		unsigned int* temp = input_d;
		input_d = output_d;
		output_d = temp;
	}

	cudaMemcpy(output, input_d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);
	cudaFree(bits_d);
	printf("In:\n");
	for (unsigned int i = 0; i < N; ++i) {
		printf("%d ", input[i]);
	}
	printf("\n\nOut:\n");
	for (unsigned int i = 0; i < N; ++i) {
		printf("%d ", output[i]);
	}

}

void radix_sort_iter_tiling() {
	unsigned int N = 128;
	unsigned int input[128];// = { 9,8, 7, 6, 5, 4,3, 2, 1, 0 };
	unsigned int output[128];
	unsigned int buckets[GRID_DIM * 2];
	for (unsigned int i = 0; i < N; ++i) {
		input[i] = rand() % 10;
	}
	unsigned int* input_d, * output_d, * buckets_d;
	int iter = 5;

	cudaMalloc((void**)&input_d, N * sizeof(unsigned int));
	cudaMalloc((void**)&output_d, N * sizeof(unsigned int));
	cudaMalloc((void**)&buckets_d, GRID_DIM * 2 * sizeof(unsigned int));

	dim3 dimGrid(4, 1, 1);
	dim3 dimBlock(32, 1, 1);

	cudaMemcpy(input_d, input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

	for (unsigned int i = 0; i < iter; ++i) {
		radix_sort_iter_tiling_kernel << <dimGrid, dimBlock >> > (input_d, output_d, buckets_d, N, i);
		//cudaDeviceSynchronize();
		//cudaMemcpy(input_d, output_d, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		unsigned int* temp = input_d;
		input_d = output_d;
		output_d = temp;
	}

	cudaMemcpy(output, output_d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(buckets, buckets_d, GRID_DIM * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);
	cudaFree(buckets_d);
	printf("In:\n");
	for (unsigned int i = 0; i < N; ++i) {
		if (i % 32 == 0) {
			printf("\nThread %d marker\n", i);
		}
		printf("%d ", input[i]);
	}
	printf("\n\nOut:\n");
	for (unsigned int i = 0; i < N; ++i) {
		if (i % 32 == 0) {
			printf("\nThread %d marker\n", i);
		}
		printf("%d ", output[i]);
	}
	printf("\n\nBuckets:\n");
	for (unsigned int i = 0; i < (GRID_DIM * 2); ++i) {
		printf("%d ", buckets[i]);
	}

}

int main() {
	radix_sort_iter();
	return 1;
}
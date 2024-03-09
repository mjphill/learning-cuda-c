
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define SECTION_SIZE 32

__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N) {
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		XY[threadIdx.x] = X[i];
	}
	else {
		XY[threadIdx.x] = 0.0f;
	}
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		float temp;
		if (threadIdx.x >= stride)
			temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
		__syncthreads();
		if (threadIdx.x >= stride)
			XY[threadIdx.x] = temp;
	}
	if (i < N) {
		Y[i] = XY[threadIdx.x];
	}
}

__global__ void Kogge_Stone_scan_double_buffer_kernel(float* X, float* Y, unsigned int N) {
	__shared__ float XY1[SECTION_SIZE], XY2[SECTION_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		XY1[threadIdx.x] = X[i];
	}
	else {
		XY1[threadIdx.x] = 0.0f;
	}
	if (threadIdx.x == 0)
		XY2[threadIdx.x] = XY1[threadIdx.x];
	unsigned int buffer_flag = 0;
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (threadIdx.x >= stride) {
			if (buffer_flag % 2 == 0) {
				XY2[threadIdx.x] = XY1[threadIdx.x] + XY1[threadIdx.x - stride];
			}
			else {
				XY1[threadIdx.x] = XY2[threadIdx.x] + XY2[threadIdx.x - stride];
			}
			buffer_flag++;
		}
	}
	if (i < N) {
		if (buffer_flag % 2 == 0) {
			Y[i] = XY1[threadIdx.x];
		}
		else {
			Y[i] = XY2[threadIdx.x];
		}
	}
}

__global__ void Kogge_Stone_exclusive_scan_double_buffer_kernel(float* X, float* Y, unsigned int N) {
	__shared__ float XY1[SECTION_SIZE], XY2[SECTION_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N && threadIdx.x != 0) {
		XY1[threadIdx.x] = X[i-1];
	}
	else {
		XY1[threadIdx.x] = 0.0f;
	}
	if (threadIdx.x == 0)
		XY2[threadIdx.x] = XY1[threadIdx.x];
	unsigned int buffer_flag = 0;
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (threadIdx.x >= stride) {
			if (buffer_flag % 2 == 0) {
				XY2[threadIdx.x] = XY1[threadIdx.x] + XY1[threadIdx.x - stride];
			}
			else {
				XY1[threadIdx.x] = XY2[threadIdx.x] + XY2[threadIdx.x - stride];
			}
			buffer_flag++;
		}
	}
	if (i < N) {
		if (buffer_flag % 2 == 0) {
			Y[i] = XY1[threadIdx.x];
		}
		else {
			Y[i] = XY2[threadIdx.x];
		}
	}
}

__global__ void segmented_scan_phase1_kernel(float* X, float* Y, float* S, unsigned int N) {
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		XY[threadIdx.x] = X[i];
	}
	else {
		XY[threadIdx.x] = 0.0f;
	}
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		float temp;
		if (threadIdx.x >= stride)
			temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
		__syncthreads();
		if (threadIdx.x >= stride)
			XY[threadIdx.x] = temp;
	}
	if (i < N) {
		Y[i] = XY[threadIdx.x];
		if (threadIdx.x == blockDim.x - 1 || i == N - 1) {
			S[blockIdx.x] = Y[i];
		}
	}
}

__global__ void segmented_scan_phase3_kernel(float* Y, float* S) {
	Y[(blockIdx.x + 1)*blockDim.x + threadIdx.x] += S[blockIdx.x];
}

void segmented_scan_phase1() {
	float* X_d, * Y_d, *S_d, *S2_d;
	float X[128], Y[128], S[4];
	unsigned int N = 128;
	for (unsigned int i = 0; i < N; ++i) {
		X[i] = (float)(i + 1);
	}
	int size = N * sizeof(float);

	dim3 dimGrid(4, 1, 1);
	dim3 dimBlock(32, 1, 1);

	cudaMalloc((void**)&X_d, size);
	cudaMalloc((void**)&Y_d, size);
	cudaMalloc((void**)&S_d, 4 * sizeof(float));
	cudaMalloc((void**)&S2_d, 4 * sizeof(float));

	cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);

	segmented_scan_phase1_kernel << <dimGrid, dimBlock >> > (X_d, Y_d, S_d, N);
	cudaDeviceSynchronize();
	Kogge_Stone_scan_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (S_d, S2_d, 4);
	cudaDeviceSynchronize();
	segmented_scan_phase3_kernel << <dim3(3, 1, 1), dim3(32, 1, 1) >> > (Y_d, S2_d);


	cudaMemcpy(&Y, Y_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&S, S2_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);
	cudaFree(S_d);
	cudaFree(S2_d);

	for (unsigned int i = 0; i < N; ++i) {
		printf("%.2f ", Y[i]);
		if ((i + 1) % 32 == 0) {
			printf("\n\n\n");
		}
	}
	printf("\n\n\nS:\n");
	for (unsigned int i = 0; i < 4; ++i) {
		printf("%.2f ", S[i]);
	}
}

void Kogge_Stone_exclusive_scan_double_buffer() {
	float* X_d, * Y_d;
	float X[32], Y[32];
	unsigned int N = 32;
	for (unsigned int i = 0; i < N; ++i) {
		X[i] = (float)(i+1);
	}
	int size = N * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(32, 1, 1);

	cudaMalloc((void**)&X_d, size);
	cudaMalloc((void**)&Y_d, size);

	cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);

	Kogge_Stone_exclusive_scan_double_buffer_kernel << <dimGrid, dimBlock >> > (X_d, Y_d, N);

	cudaMemcpy(&Y, Y_d, size, cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);

	for (unsigned int i = 0; i < N; ++i) {
		printf("%.2f ", Y[i]);
	}
}

void Kogge_Stone_scan_double_buffer() {
	float* X_d, * Y_d;
	float X[32], Y[32];
	unsigned int N = 32;
	for (unsigned int i = 0; i < N; ++i) {
		X[i] = (float)(i+1);
	}
	int size = N * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(32, 1, 1);

	cudaMalloc((void**)&X_d, size);
	cudaMalloc((void**)&Y_d, size);

	cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);

	Kogge_Stone_scan_double_buffer_kernel << <dimGrid, dimBlock >> > (X_d, Y_d, N);

	cudaMemcpy(&Y, Y_d, size, cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);

	for (unsigned int i = 0; i < N; ++i) {
		printf("%.2f ", Y[i]);
	}
}

void Kogge_Stone_scan() {
	float* X_d, * Y_d;
	float X[128], Y[128];
	unsigned int N = 128;
	for (unsigned int i = 0; i < N; ++i) {
		X[i] = (float)(i+1);
	}
	int size = N * sizeof(float);

	dim3 dimGrid(4, 1, 1);
	dim3 dimBlock(32, 1, 1);

	cudaMalloc((void**) & X_d, size);
	cudaMalloc((void**) & Y_d, size);

	cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);

	Kogge_Stone_scan_kernel << <dimGrid, dimBlock >> > (X_d, Y_d, N);

	cudaMemcpy(&Y, Y_d, size, cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);

	for (unsigned int i = 0; i < N; ++i) {
		printf("%.2f ", Y[i]);
	}
}

int main() {
	segmented_scan_phase1();
	return 1;
}

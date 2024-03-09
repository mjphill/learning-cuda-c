
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define TILE_WIDTH 2

__global__ void matrixMultKernel (float *A, float *B, float *C, int Width){
	__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	// Identify row and column of C element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	// Loop over A and B tiles to compute C
	float Cvalue = 0;
	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {

		// Collaborative loading of A and B tiles into shared memory
		Ads[ty][tx] = A[Row * Width + ph * TILE_WIDTH + tx];
		//Bds[ty][tx] = B[Col * Width + ph*TILE_WIDTH + ty];
		Bds[ty][tx] = B[Col * Width +ph * TILE_WIDTH + ty];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k) {
			Cvalue += Ads[ty][k] * Bds[k][tx];
			//Cvalue += Ads[ty][k] * Bds[ty][k];
		}
		__syncthreads();
	}
	C[Row * Width + Col] = Cvalue;
	
}

void MatrixMul() {
	float* A_d, * B_d, * P_d;
	float A[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0, 2.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	float B[4][4] = {
		{2.0, 1.0, 1.0, 1.0},
		{2.0, 1.0, 1.0, 1.0},
		{2.0, 1.0, 1.0, 1.0},
		{2.0, 1.0, 1.0, 1.0}
	};
	//float* P_d[4][4];
	float P[4][4];
	int size = 16 * sizeof(float);

	dim3 dimGrid(2, 2, 1);
	dim3 dimBlock(2, 2, 1);

	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&P_d, size);

	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	matrixMultKernel << <dimGrid, dimBlock >> > (A_d, B_d, P_d, 4);

	cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(P_d);

	int row, columns;
	for (row = 0; row < 4; row++)
	{
		for (columns = 0; columns < 4; columns++)
		{
			printf("%.2f     ", P[row][columns]);
		}
		printf("\n");
	}
}

__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float a_s[256];
	__shared__ float bc_s[4 * 256];
	a_s[threadIdx.x] = a[i];
	for (unsigned int j = 0; j < 4; ++j) {
		bc_s[j * 256 + threadIdx.x] = b[j * blockDim.x * gridDim.x + i] + c[i * 4 + j];
	}
	__syncthreads();
	d[i + 8] = a_s[threadIdx.x];
	e[i * 8] = bc_s[threadIdx.x * 4];
}

void main() {
	MatrixMul();
}

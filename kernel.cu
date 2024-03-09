
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void convolution_3D_basic_kernel(float* N, float* F, float* P, int r, int width, int height, int depth) {
	int inRow; int inCol; int inDepth;
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outDepth = blockIdx.z * blockDim.z + threadIdx.z;
	float Pvalue = 0.0f;
	for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
		for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
			for (int fDepth = 0; fDepth < 2 * r + 1; fDepth++) {
				inRow = outRow - r + fRow;
				inCol = outCol - r + fCol;
				inDepth = outDepth - r + fDepth;
				if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width && inDepth >= 0 && inDepth < depth) {
					Pvalue += F[fDepth*height*width + fRow*width + fCol] * N[inDepth*height*width + inRow * width + inCol];
				}
			}
		}
	}
}

#define FILTER_RADIUS 2
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_3D_const_mem_kernel(float* N, float* P, int r, int width, int height, int depth) {
	int inRow, inCol, inDepth;
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.x * blockDim.x + threadIdx.y;
	int outDepth = blockIdx.z * blockDim.z + threadIdx.z;
	float Pvalue = 0.0f;
	for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
		for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
			for (int fDepth = 0; fDepth < 2 * r + 1; fDepth++) {
				inRow = outRow - r + fRow;
				inCol = outCol - r + fCol;
				inDepth = outDepth - r + fDepth;
				if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width && inDepth >= 0 && inDepth < depth) {
					Pvalue += F[fDepth][fRow][fCol] * N[inDepth*depth*width + inRow * width + inCol];
				}
			}
		}
	}
}

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
//__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_tiled_2D_const_mem_kernel(float* N, float* P, int width, int height) {
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
	//loading input tile
	__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
	if (row >= 0 && row < height && col >= 0 && col < width) {
		N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
	}
	else {
		N_s[threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();
	//Calculating output elements
	int tileCol = threadIdx.x - FILTER_RADIUS;
	int tileRow = threadIdx.y - FILTER_RADIUS;
	//turning off the threads at the edges of the block
	if (col >= 0 && col < width && row >= 0 && row < height) {
		if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
			float Pvalue = 0.0f;
			for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
				for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
					Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
				}
			}
			P[row * width + col] = Pvalue;
		}
	}
}

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];
__global__ void convolution_tiled_3D_const_mem_kernel(float* N, float* P, int width, int height, int depth) {
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
	int dep = blockIdx.z * OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
	//loading input tile
	__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
	if (row >= 0 && row < height && col >= 0 && col < width && dep >= 0 && dep < depth) {
		N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[dep*depth*width + row * width + col];
	}
	else {
		N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();
	//Calculating output elements
	int tileCol = threadIdx.x - FILTER_RADIUS;
	int tileRow = threadIdx.y - FILTER_RADIUS;
	int tileDep = threadIdx.z - FILTER_RADIUS;
	//turning off the threads at the edges of the block
	if (col >= 0 && col < width && row >= 0 && row < height && dep >= 0 && dep < depth) {
		if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM && tileDep >= 0 && tileDep < OUT_TILE_DIM) {
			float Pvalue = 0.0f;
			for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
				for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
					for (int fDep = 0; fDep < 2 * FILTER_RADIUS + 1; fDep++) {
						Pvalue += F_c[fDep*width*depth + fRow*width + fCol] * N_s[tileDep + fDep][tileRow + fRow][tileCol + fCol];
					}
				}
			}
			P[dep*depth*width + row * width + col] = Pvalue;
		}
	}
}

int main() {
	return 1;
}
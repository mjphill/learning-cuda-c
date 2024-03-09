
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define min(a,b) (a < b ? a : b)

__device__ void merge_sequential(int* A, int m, int* B, int n, int* C);
__device__ int co_rank(int k, int* A, int m, int* B, int n);

int compareIntegers(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}

__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size) {
	/* shared memory allocation*/
	extern __shared__ int shareAB[];
	int* A_S = &shareAB[0];														//shareA is the first half of shareAB
	int* B_S = &shareAB[tile_size];												//shareB is the second half of shareAB
	int C_curr = blockIdx.x * ceilf((m + n) / gridDim.x);						//start point of block's C subarray
	int C_next = min((blockIdx.x + 1) * ceilf((m + n) / gridDim.x), (m + n));	//ending point

	if (threadIdx.x == 0) {
		A_S[0] = co_rank(C_curr, A, m, B, n);	//Make block-level co-rank values visible
		A_S[1] = co_rank(C_next, A, m, B, n);	//to other threads in the block
	}

	__syncthreads();
	int A_curr = A_S[0];
	int A_next = A_S[1];
	int B_curr = C_curr - A_curr;
	int B_next = C_next - A_next;
	__syncthreads();
	int counter = 0;											//iteration counter
	int C_length = C_next - C_curr;
	int A_length = A_next - A_curr;
	int B_length = B_next - B_curr;
	int total_iteration = ceilf((C_length) / tile_size);			//total iteration
	int C_completed = 0;
	int A_consumed = 0;
	int B_consumed = 0;
	while (counter < total_iteration) {
		/* loading tile-size A and B elements into shared memory */
		for (int i = 0; i < tile_size; i += blockDim.x) {
			if (i + threadIdx.x < A_length - A_consumed) {
				A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
			}
		}
		for (int i = 0; i < tile_size; i += blockDim.x) {
			if (i + threadIdx.x < B_length - B_consumed) {
				B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
			}
		}
		__syncthreads();
		int c_curr = threadIdx.x * (tile_size / blockDim.x);
		int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
		c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
		c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
		/* find co-rank for c_curr and c_next*/
		int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
		int b_curr = c_curr - a_curr;
		int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
		int b_next = c_next - a_next;

		/* All threads call the sequential merge function */
		merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr);
		/* Update the number of A and B elemets that have been consumed thus far */
		counter++;
		C_completed += tile_size;
		A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
		B_consumed = C_completed - A_consumed;
		__syncthreads();
	}
}

__global__ void merge_tiled_kernel2(int* A, int m, int* B, int n, int* C, int tile_size) {
	/* shared memory allocation*/
	extern __shared__ int shareAB[];
	int* A_S = &shareAB[0];														//shareA is the first half of shareAB
	int* B_S = &shareAB[tile_size];												//shareB is the second half of shareAB
	int C_curr = blockIdx.x * ceilf((m + n) / gridDim.x);						//start point of block's C subarray
	int C_next = min((blockIdx.x + 1) * ceilf((m + n) / gridDim.x), (m + n));	//ending point

	if (threadIdx.x == 0) {
		A_S[0] = co_rank(C_curr, A, m, B, n);	//Make block-level co-rank values visible
		A_S[1] = co_rank(C_next, A, m, B, n);	//to other threads in the block
	}

	__syncthreads();
	int A_curr = A_S[0];
	int A_next = A_S[1];
	int B_curr = C_curr - A_curr;
	int B_next = C_next - A_next;
	__syncthreads();
	int counter = 0;											//iteration counter
	int C_length = C_next - C_curr;
	int A_length = A_next - A_curr;
	int B_length = B_next - B_curr;
	int total_iteration = ceilf((C_length) / tile_size);			//total iteration
	int C_completed = 0;
	int A_consumed = 0;
	int B_consumed = 0;
	while (counter < total_iteration) {
		/* loading tile-size A and B elements into shared memory */
		//for (int i = 0; i < tile_size; i += blockDim.x) {
			if (threadIdx.x < A_length - A_consumed) {
				A_S[threadIdx.x] = A[A_curr + A_consumed + threadIdx.x];
			}
		//}
		//for (int i = 0; i < tile_size; i += blockDim.x) {
			if (threadIdx.x < B_length - B_consumed) {
				B_S[threadIdx.x] = B[B_curr + B_consumed + threadIdx.x];
			}
		//}
		__syncthreads();
		int c_curr = threadIdx.x * (tile_size / blockDim.x);
		int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
		c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
		c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
		/* find co-rank for c_curr and c_next*/
		int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
		int b_curr = c_curr - a_curr;
		int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
		int b_next = c_next - a_next;

		/* All threads call the sequential merge function */
		merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr);
		/* Update the number of A and B elemets that have been consumed thus far */
		counter++;
		C_completed += tile_size;
		A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
		B_consumed = C_completed - A_consumed;
		__syncthreads();
	}
}

__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
	int i = 0;	//Index into A
	int j = 0;	//Index into B
	int k = 0;	//Index into C
	while ((i < m) && (j < n)) {
		if (A[i] <= B[j]) {
			C[k++] = A[i++];
		}
		else {
			C[k++] = B[j++];
		}
	}
	if (i == m) {				//Done with A[], handle remaining B[]
		while (j < n) {
			C[k++] = B[j++];
		}
	}
	else {
		while (i < m) {			//Done with B[], handle remaining A[]
			C[k++] = A[i++];
		}
	}
}

__device__ int co_rank(int k, int* A, int m, int* B, int n) {
	int i = min(k, m);
	int j = k - i;
	int i_low = 0 > (k - n) ? 0 : k - n;
	int j_low = 0 > (k - m) ? 0 : k - m;
	int delta;
	bool active = true;
	while (active) {
		if (i > 0 && j < n && A[i - 1] > B[j]) {
			delta = ((i - i_low + 1) >> 1);
			j_low = j;
			j = j + delta;
			i = i - delta;
		}
		else if (j > 0 && i < m && B[j - 1] >= A[i]) {
			delta = ((j - j_low + 1) >> 1);
			i_low = i;
			i = i + delta;
			j = j - delta;
		}
		else {
			active = false;
		}
	}
	return i;
}

void merge_tiled() {
	int* A_d, * B_d, * C_d;
	int A[2048]; 
	int B[2048];
	int C[4096];
	for (int i = 0; i < 2048; ++i) {
		A[i] = rand() % 100;
	}
	size_t size = sizeof(A) / sizeof(A[0]);
	qsort(A, size, sizeof(int), compareIntegers);
	for (int i = 0; i < 2048; ++i) {
		B[i] = rand() % 100;
	}
	size = sizeof(B) / sizeof(B[0]);
	qsort(B, size, sizeof(int), compareIntegers);

	dim3 dimGrid(16, 1, 1);
	dim3 dimBlock(256, 1, 1);
	
	cudaMalloc((void**)&A_d, 2048 * sizeof(int));
	cudaMalloc((void**)&B_d, 2048 * sizeof(int));
	cudaMalloc((void**)&C_d, 4096 * sizeof(int));

	cudaMemcpy(A_d, A, 2048 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, 2048 * sizeof(int), cudaMemcpyHostToDevice);

	merge_tiled_kernel2 << <dimGrid, dimBlock, 512*sizeof(int) >> > (A_d, 2048, B_d, 2048, C_d, 256);

	cudaMemcpy(C, C_d, 4096 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	/*printf("A:\n");
	for (int i = 0; i < 64; ++i) {
		printf("%d ",A[i]);
	}
	printf("\n\nB:\n");
	for (int i = 0; i < 58; ++i) {
		printf("%d ", B[i]);
	}
	printf("\n\n");
	*/
	for (int i = 0; i < 4096; ++i) {
		printf("%d: %d \n", i, C[i]);
	}

}

int main() {
	merge_tiled();
	return 1;
}
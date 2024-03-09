#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < Width) && (col < Width)) {
		float Pvalue = 0;
		for (int k = 0; k < Width; ++k) {
			Pvalue += M[row * Width + k] * N[k * Width + col];
		}
		

		P[row * Width + col] = Pvalue;
		printf("p address: %d		pvalue: %f\n", row * Width + col, P[row * Width + col]);
		printf("thread id y:%d        thread id x:%d      p address:%d        pvalue:%f\n", threadIdx.y, threadIdx.x, row * Width + col, P[row * Width + col]);

	}

}

__global__
void MatrixMulRowKernel(float* M, float* N, float* P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;  //this is effectively 0 if the execution configuration parameters are appropriate. 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float Pvalue = 0;
	if ((row < Width) && (col < Width)) {
		for (int j = 0; j < Width; ++j) {
			Pvalue = 0;
			for (int k = 0; k < Width; ++k) {
				Pvalue += M[row * Width + k] * N[k * Width + col];
			}
			P[row * Width + col] = Pvalue;
			++row;
		}

		
		printf("p address: %d		pvalue: %f\n", row * Width + col, P[row * Width + col]);
		printf("thread id y:%d        thread id x:%d      p address:%d        pvalue:%f\n", threadIdx.y, threadIdx.x, row * Width + col, P[row * Width + col]);
		printf("blockdimY %d\n", blockDim.y);

	}
}

__global__
void MatrixMulColKernel(float* M, float* N, float* P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;   
	int col = blockIdx.x * blockDim.x + threadIdx.x; //this is effectively 0 if the execution configuration parameters are appropriate.
	float Pvalue = 0;
	if ((row < Width) && (col < Width)) {
		for (int j = 0; j < Width; ++j) {
			col = blockDim.x * j;
			Pvalue = 0;
			for (int k = 0; k < Width; ++k) {
				Pvalue += M[row * Width + k] * N[k * Width + col];
			}
			P[row * Width + col] = Pvalue;
		}


		printf("p address: %d		pvalue: %f\n", row * Width + col, P[row * Width + col]);
		printf("thread id y:%d        thread id x:%d      p address:%d        pvalue:%f\n", threadIdx.y, threadIdx.x, row * Width + col, P[row * Width + col]);
		printf("blockdimX %d\n", blockDim.x);

	}
}

void MatrixMul() {
	float *A_d, *B_d, *P_d;
	float A[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0, 2.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	float B[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	//float* P_d[4][4];
	float P[4][4];
	int size = 16 * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(4, 4, 1);

	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&P_d, size);

	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	MatrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, P_d, 4);

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

void MatrixMulRow() {
	float* A_d, * B_d, * P_d;
	float A[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0, 2.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	float B[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	//float* P_d[4][4];
	float P[4][4];
	int size = 16 * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(4, 1, 1);

	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&P_d, size);

	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	MatrixMulRowKernel << <dimGrid, dimBlock >> > (A_d, B_d, P_d, 4);

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

void MatrixMulCol() {
	float* A_d, * B_d, * P_d;
	float A[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0, 2.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	float B[4][4] = {
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	//float* P_d[4][4];
	float P[4][4];
	int size = 16 * sizeof(float);

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(1, 4, 1);

	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&P_d, size);

	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	MatrixMulColKernel << <dimGrid, dimBlock >> > (A_d, B_d, P_d, 4);

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

__global__
void MatVecMultKernel(float* Out, float *M, float* V, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x; //this is effectively 0 if the execution configuration parameters are appropriate.

	float output = 0;
	printf("row: %d col: %d  n:%d\n", row, col,n);

	if ((row < n) && (col < n)) {
		for (int i = 0; i < n; ++i) {
			output += (M[row*n + i] + V[i]);
			printf("M: %f   V: %f    Output: %f\n", M[i*n+row], V[row], output);
		}
		
		Out[row] = output;
	}	
}

void MatVecMult() {

	int n = 4;

	float* Out_d, * M_d, * V_d;

	float Out[4];
	float M[4][4] = {
		{1.0, 1.0, 1.0, 5.0},
		{2.0, 2.0, 2.0, 2.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0}
	};
	float V[4] = { 1.0, 2.0, 3.0, 4.0 };

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(1, n, 1);

	int size = n * sizeof(float);

	cudaMalloc((void**) &M_d, 16*sizeof(float));
	cudaMalloc((void**) &V_d, size);
	cudaMalloc((void**) &Out_d, size);

	cudaMemcpy(M_d, M, 16*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(V_d, V, size, cudaMemcpyHostToDevice);

	MatVecMultKernel<<<dimGrid, dimBlock >>> (Out_d, M_d, V_d, n);

	cudaMemcpy(Out, Out_d, size, cudaMemcpyDeviceToHost);

	cudaFree(M_d);
	cudaFree(V_d);
	cudaFree(Out_d);

	for (int i = 0; i < n; ++i) {
		printf("%f ", Out[i]);
	}
} 

int main() {

	MatVecMult();
	//MatrixMulCol();
	return 5;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define M 1024
#define N 2048
#define K 1024
#define RANGE 100
#define BLOCK_SIZE 32


// Mat1[M][N]
// Mat2[N][K]
__global__
void MatMul(int* Mat1, int* Mat2, int* Res) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < M) && (col < K)) {
		int Value = 0;
		for (int k = 0; k < N; ++k) {
			Value += Mat1[row * N + k] * Mat2[k * K + col];
		}
		Res[row * K + col] = Value;
	}
}


int main(void) {

	srand(time(NULL));

	int* Mat1 = (int*)malloc(sizeof(int) * M * N);
	int* Mat2 = (int*)malloc(sizeof(int) * N * K);
	for (int i = 0; i < M * N; ++i) {
		Mat1[i] = rand() % RANGE + 1; // 1 - 100
	}
	for(int i = 0; i< N * K; ++i) {
		Mat2[i] = rand() % RANGE + 1;
	}
	int* Mat_d1;
	int* Mat_d2;
	int* Res_d;
	cudaMalloc((void**)&Mat_d1, sizeof(int) * M * N);
	cudaMalloc((void**)&Mat_d2, sizeof(int) * N * K);
	cudaMalloc((void**)&Res_d, sizeof(int) * M * K);
	cudaMemcpy(Mat_d1, Mat1, sizeof(int) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(Mat_d2, Mat2, sizeof(int) * N * K, cudaMemcpyHostToDevice);

	cudaEvent_t Start, Stop;
	float Milliseconds = 0.0f;
	cudaEventCreate(&Start);
	cudaEventCreate(&Stop);

	dim3 GridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
	dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	cudaEventRecord(Start);

	MatMul<<<GridDim, BlockDim >>>(Mat_d1, Mat_d2, Res_d);

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time: %f ms\n", Milliseconds);
	cudaEventDestroy(Start);
	cudaEventDestroy(Stop);
	cudaFree(Mat_d1);
	cudaFree(Mat_d2);
	cudaFree(Res_d);

	free(Mat1);
	free(Mat2);
	return 0;
}
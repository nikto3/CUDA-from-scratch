#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define M 1024
#define N 4096
#define K 1024
#define RANGE 100
#define BLOCK_SIZE 32


// Mat1[M][N]
// Mat2[N][K]
// Works even when Mat1 and Mat2 dims are not divisible by BLOCK_SIZE
__global__
void MatMul(float* Mat1, float* Mat2, float* Res) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;
	__shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

	float Value = 0.0f;
	for(int Phase = 0; Phase < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++Phase) {
		if(Row < M && (Phase*BLOCK_SIZE + Tx) < N) {
			Mds[Ty][Tx] = Mat1[Row*N + Phase*BLOCK_SIZE + Tx];
		}
		else {
			Mds[Ty][Tx] = 0.0f;
		}

		if((Phase*BLOCK_SIZE + Ty) < N && Col < K) {
			Nds[Ty][Tx] = Mat2[(Phase*BLOCK_SIZE + Ty)*K + Col];
		}
		else {
			Nds[Ty][Tx] = 0.0f;
		}

		__syncthreads(); // wait for other threads in the block to finish loading

		for(int i = 0; i < BLOCK_SIZE; ++i) {
			Value += Mds[Ty][i] * Nds[i][Tx];
		}

		__syncthreads(); // wait for other threads in the block before proceeding to load new tiles
	}

	if(Row < M && Col < K) {
		Res[Row*K + Col] = Value;
	}
	
}


int main(void) {

	srand(time(NULL));

	float* Mat1 = (float*)malloc(sizeof(float) * M * N);
	float* Mat2 = (float*)malloc(sizeof(float) * N * K);
	for (int i = 0; i < M * N; ++i) {
		Mat1[i] = rand() % RANGE + 1; // 1 - 100
	}
	for(int i = 0; i< N * K; ++i) {
		Mat2[i] = rand() % RANGE + 1;
	}
	float* Mat_d1;
	float* Mat_d2;
	float* Res_d;
	cudaMalloc((void**)&Mat_d1, sizeof(float) * M * N);
	cudaMalloc((void**)&Mat_d2, sizeof(float) * N * K);
	cudaMalloc((void**)&Res_d, sizeof(float) * M * K);
	cudaMemcpy(Mat_d1, Mat1, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(Mat_d2, Mat2, sizeof(float) * N * K, cudaMemcpyHostToDevice);

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
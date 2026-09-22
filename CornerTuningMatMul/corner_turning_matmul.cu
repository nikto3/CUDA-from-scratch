#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define M 1024			// 1024
#define N 2048			// 4096
#define K 1024			// 1024
#define RANGE 100
#define BLOCK_SIZE 32
#define COARSENING_FACTOR 4

#define CHECK_CUDA_ERROR(Val) Check((Val), #Val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() CheckLast(__FILE__, __LINE__) 


inline 
void Compare(float* Mat1, float* Mat2) {
	for(int i = 0; i<M*K; ++i){
		if(fabs(Mat1[i] - Mat2[i]) >= 1e-3f) {
			printf("Error while multiplying matrices.\n");
			exit(-1);
		}
	}
}


void Check(cudaError_t Err, const char* Func, const char* File, int Line) {
	if(Err != cudaSuccess) {
		printf("Cuda Runtime Error At %s: %d\n", File, Line);
		printf("%s %s\n", cudaGetErrorString(Err), Func);
	}
}  

void CheckLast(const char* File, int Line) {
	cudaError_t Err = cudaGetLastError();
	if(Err != cudaSuccess) {
		printf("Cuda Runtime Error At %s: %d\n", File, Line);
		printf("%s\n", cudaGetErrorString(Err));
	}
}


float* MatMulCpu(float* Mat1, float* Mat2) {
	float* Res = (float*)malloc(sizeof(float) * M * K);
	for(int i = 0; i < M * K; ++i) {
		Res[i] = 0.0f;
	}
	for(int Row = 0; Row<M; ++Row) {
		for(int s = 0; s<N; ++s) {
			for(int Col = 0; Col<K; ++Col) {
				Res[Row*K + Col] += Mat1[Row*N + s] * Mat2[s*K + Col];
			}
		}
	}
	return Res;
}


// Mat1[M][N]
// Mat2[N][K]
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
			Nds[Ty][Tx] = Mat2[(Phase*BLOCK_SIZE + Ty) * K + Col];
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


// With thread coarsening
__global__
void MatMulCoars(float* Mat1, float* Mat2, float* Res) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int ColStart = blockIdx.x * blockDim.x * COARSENING_FACTOR + threadIdx.x;
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;
	__shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

	float Value[COARSENING_FACTOR];
	for(int i = 0; i<COARSENING_FACTOR; ++i) {
		Value[i] = 0.0f;
	}
	for(int Phase = 0; Phase < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++Phase) {
		if(Row < M && (Phase*BLOCK_SIZE + Tx) < N) {
			Mds[Ty][Tx] = Mat1[Row*N + Phase*BLOCK_SIZE + Tx];
		}
		else {
			Mds[Ty][Tx] = 0.0f;
		}

		for(int c = 0; c < COARSENING_FACTOR; ++c) {
			int Col = ColStart + c*BLOCK_SIZE;

			if((Phase*BLOCK_SIZE + Ty) < N && Col < K) {
				Nds[Ty][Tx] = Mat2[(Phase * BLOCK_SIZE + Ty) * K + Col];
			}
			else {
				Nds[Ty][Tx] = 0.0f;
			}
			__syncthreads(); // wait for other threads in the block to finish loading

			for(int i = 0; i < BLOCK_SIZE; ++i) {
				Value[c] += Mds[Ty][i] * Nds[i][Tx];
			}

			__syncthreads(); // wait for other threads in the block before proceeding to load new tiles
		}
	}

	for(int c = 0; c < COARSENING_FACTOR; ++c) {
		int Col = ColStart + c * BLOCK_SIZE;
		if(Row < M && Col < K) {
			Res[Row*K + Col] = Value[c];
		}
	}
	
}


int main(void) {

	srand(time(NULL));

	float* Mat1 = (float*)malloc(sizeof(float) * M * N);
	float* Mat2 = (float*)malloc(sizeof(float) * N * K);
	float* Res = (float*)malloc(sizeof(float) * M * K);
	for (int i = 0; i < M * N; ++i) {
		Mat1[i] = rand() % RANGE + 1; // 1 - 100
	}
	for(int i = 0; i< N * K; ++i) {
		Mat2[i] = rand() % RANGE + 1;
	}
	float* MatMulRes = MatMulCpu(Mat1, Mat2);
	float* Mat_d1;
	float* Mat_d2;
	float* Res_d;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Mat_d1, sizeof(float) * M * N));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Mat_d2, sizeof(float) * N * K));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Res_d, sizeof(float) * M * K));
	CHECK_CUDA_ERROR(cudaMemcpy(Mat_d1, Mat1, sizeof(float) * M * N, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(Mat_d2, Mat2, sizeof(float) * N * K, cudaMemcpyHostToDevice));

	cudaEvent_t Start, Stop;
	float Milliseconds = 0.0f;
	cudaEventCreate(&Start);
	cudaEventCreate(&Stop);

	dim3 GridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
	dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	cudaEventRecord(Start);

	MatMul<<<GridDim, BlockDim>>>(Mat_d1, Mat_d2, Res_d);

	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 
	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Tiled): %f ms\n", Milliseconds);

	CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
	Compare(Res, MatMulRes);

	GridDim.x = (K / COARSENING_FACTOR + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cudaEventRecord(Start);
	MatMulCoars<<<GridDim, BlockDim>>>(Mat_d1, Mat_d2, Res_d);
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 
	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Coarsened): %f ms\n", Milliseconds);

	CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
	Compare(Res, MatMulRes);

	cudaEventDestroy(Start);
	cudaEventDestroy(Stop);
	cudaFree(Mat_d1);
	cudaFree(Mat_d2);
	cudaFree(Res_d);

	free(Mat1);
	free(Mat2);
	free(MatMulRes);
	free(Res);
	return 0;
}
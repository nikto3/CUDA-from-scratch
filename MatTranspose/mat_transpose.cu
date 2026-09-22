#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define M 1024
#define N 4096
#define RANGE 100
#define BLOCK_SIZE 32


#define CHECK_CUDA_ERROR(Val) Check((Val), #Val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() CheckLast(__FILE__, __LINE__) 


inline 
void Compare(float* Mat1, float* Mat2) {
	for(int i = 0; i<M*N; ++i){
		if(fabs(Mat1[i] - Mat2[i]) >= 1e-6) {
			printf("Error while transposing matrix.\n");
			exit(-1);
		}
	}
}


float* MatTransposeCpu(float* Mat) {
	float* Res = (float*)malloc(sizeof(float) * M * N);
	for(int i = 0; i<M; ++i) {
		for(int j = 0; j<N; ++j) {
			Res[j*M + i] = Mat[i*N + j];
		}
	}
	return Res;
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


// Mat[M][N]
__global__
void MatTransposeNaive(float* Mat, float* Res) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(Row < M && Col < N) {
		// Res(Col, Row) = Mat(Row, Col)
		Res[Col*M + Row] = Mat[Row*N + Col];
	}
}

__global__
void MatTransposeConflict(float* Mat, float* Res) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;

	__shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];

	if(Row < M && Col < N) {
		Mds[Ty][Tx] = Mat[Row*N + Col];
	}
	__syncthreads();

	if(Row < M && Col < N) {
		Res[(blockIdx.x*blockDim.x + threadIdx.y)*M + (blockIdx.y*blockDim.y + threadIdx.x)] = Mds[Tx][Ty];
	}

}

__global__
void MatTransposeNoConflict(float* Mat, float* Res) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;

	// added +1 to avoid bank conflicts while accessing the memory
	// now each thread reads from different bank, allowing for parallel reading
	__shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE+1];

	if(Row < M && Col < N) {
		Mds[Ty][Tx] = Mat[Row*N + Col];
	}
	__syncthreads();

	if(Row < M && Col < N) {
		Res[(blockIdx.x*blockDim.x + threadIdx.y)*M + (blockIdx.y*blockDim.y + threadIdx.x)] = Mds[Tx][Ty];
	}

}


int main(void) {

	srand(time(NULL));

	float* Mat1 = (float*)malloc(sizeof(float) * M * N);
	for (int i = 0; i < M * N; ++i) {
		Mat1[i] = rand() % RANGE + 1; // 1 - 100
	}
	float* Mat1Transpose = MatTransposeCpu(Mat1);
	float* Mat_d;
	float* Res_d;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Mat_d, sizeof(float) * M * N));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Res_d, sizeof(float) * M * N));
	CHECK_CUDA_ERROR(cudaMemcpy(Mat_d, Mat1, sizeof(float) * M * N, cudaMemcpyHostToDevice));

	cudaEvent_t Start, Stop;
	float Milliseconds = 0.0f;
	cudaEventCreate(&Start);
	cudaEventCreate(&Stop);

	dim3 GridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
	dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	cudaEventRecord(Start);

	MatTransposeNaive<<<GridDim, BlockDim >>>(Mat_d, Res_d);

	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Naive): %f ms\n", Milliseconds);

	CHECK_CUDA_ERROR(cudaMemcpy(Mat1, Res_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

	Compare(Mat1, Mat1Transpose);


	cudaEventRecord(Start);
	MatTransposeConflict<<<GridDim, BlockDim>>>(Mat_d, Res_d);
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Shared Memory - Conflict): %f ms\n", Milliseconds);

	CHECK_CUDA_ERROR(cudaMemcpy(Mat1, Res_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
	Compare(Mat1, Mat1Transpose);

	cudaEventRecord(Start);
	MatTransposeNoConflict<<<GridDim, BlockDim>>>(Mat_d, Res_d);
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Shared Memory - No Conflict): %f ms\n", Milliseconds);

	CHECK_CUDA_ERROR(cudaMemcpy(Mat1, Res_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
	Compare(Mat1, Mat1Transpose);

	printf("Transpose works!\n");

	cudaEventDestroy(Start);
	cudaEventDestroy(Stop);
	cudaFree(Mat_d);
	cudaFree(Res_d);

	free(Mat1);
	free(Mat1Transpose);
	return 0;
}
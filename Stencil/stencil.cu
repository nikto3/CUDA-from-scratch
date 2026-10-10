#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "..\cuda_check.cuh"

#define C 96
#define M 384
#define N 384
#define RANGE 100
#define IN_TILE_X 16
#define IN_TILE_Y 8
#define IN_TILE_Z 8
#define OUT_TILE_X (IN_TILE_X - 2)
#define OUT_TILE_Y (IN_TILE_Y - 2)
#define OUT_TILE_Z (IN_TILE_Z - 2)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define ERROR_MESSAGE "Error while applying 3d stencil to matrix."

// +- 1 for x axis
// +- 1 for y axis
// +- 1 for z axis
// value itself
// 2 + 2 + 2 + 1 = 7
typedef union {
    struct {float Values[7];};
    struct {
        float C0;
        float C1;
        float C2;
        float C3;
        float C4;
        float C5;
        float C6;
    };
} StencilValues;


float* StencilCpu(float* Input, StencilValues* SValues) {
    float* Output = (float*)malloc(sizeof(float) * C * M * N);
    for(int Channel = 1; Channel < C - 1; ++Channel) {
        for(int Row = 1; Row < M - 1; ++Row) {
            for(int Col = 1; Col < N - 1; ++Col) {
                Output[Channel*(M*N) + Row*N + Col] = SValues->C0 * Input[Channel*(M*N) + Row*N + Col]
                                                    + SValues->C1* Input[Channel*(M*N) + Row*N + Col-1]
                                                    + SValues->C2 * Input[Channel*(M*N) + Row*N + Col+1]
                                                    + SValues->C3 * Input[Channel*(M*N) + (Row-1)*N + Col]
                                                    + SValues->C4 * Input[Channel*(M*N) + (Row+1)*N + Col]
                                                    + SValues->C5 * Input[(Channel-1)*(M*N) + Row*N + Col]
                                                    + SValues->C6 * Input[(Channel+1)*(M*N) + Row*N + Col];
            }
        }
    }

    int BorderChannels[] = {0, C-1};
    int BorderRows[] = {0, M-1};
    int BorderCols[] = {0, N-1};

    for(int i = 0; i<2; ++i){
        int Channel = BorderChannels[i];
        for(int Row = 0; Row<M; ++Row){
            for(int Col = 0; Col<N; ++Col){
                Output[Channel*(M*N) + Row*N + Col] = Input[Channel*(M*N) + Row*N + Col];
            }
        }
    }
    for(int i = 0; i<2; ++i){
        int Row = BorderRows[i];
        for(int Channel = 0; Channel<C; ++Channel){
            for(int Col = 0; Col<N; ++Col){
                Output[Channel*(M*N) + Row*N + Col] = Input[Channel*(M*N) + Row*N + Col];
            }
        }
    }
    for(int i = 0; i<2; ++i){
        int Col = BorderCols[i];
        for(int Channel = 0; Channel<C; ++Channel){
            for(int Row = 0; Row<M; ++Row){
                Output[Channel*(M*N) + Row*N + Col] = Input[Channel*(M*N) + Row*N + Col];
            }
        }
    }
    return Output;
}


__constant__ StencilValues SValues_d;

__global__
void NaiveStencil(float* Input, float* Output) {
    int Channel = blockIdx.z;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Channel >= 1 && Channel < C - 1 && Row >= 1 && Row < M - 1 && Col >= 1 && Col < N - 1) {
        Output[Channel*(M*N) + Row*N + Col] = SValues_d.C0 * Input[Channel*(M*N) + Row*N + Col] 
                                            + SValues_d.C1 * Input[Channel*(M*N) + Row*N + (Col - 1)]
                                            + SValues_d.C2 * Input[Channel*(M*N) + Row*N + (Col + 1)]
                                            + SValues_d.C3 * Input[Channel*(M*N) + (Row - 1)*N + Col]
                                            + SValues_d.C4 * Input[Channel*(M*N) + (Row + 1)*N + Col]
                                            + SValues_d.C5 * Input[(Channel - 1)*(M*N) + Row*N + Col]
                                            + SValues_d.C6 * Input[(Channel + 1)*(M*N) + Row*N + Col];
    }
}


// Shared memory variant
__global__
void SharedStencil(float* Input, float* Output) {
    int Channel = blockIdx.z*OUT_TILE_Z + threadIdx.z - 1;
    int Row = blockIdx.y*OUT_TILE_Y + threadIdx.y - 1;
    int Col = blockIdx.x*OUT_TILE_X + threadIdx.x - 1;
    __shared__ float Mds[IN_TILE_Z][IN_TILE_Y][IN_TILE_X];

    if(Row >= 0 && Row < M && Col >= 0 && Col < N && Channel >= 0 && Channel < C) {
        Mds[threadIdx.z][threadIdx.y][threadIdx.x] = Input[Channel*(M*N) + Row*N + Col];
    }

    __syncthreads();

    if(Row >= 1 && Row < M - 1 && Col >= 1 && Col < N - 1 && Channel >= 1 && Channel < C - 1) {
        if(threadIdx.x >= 1 && threadIdx.x < IN_TILE_X - 1 && threadIdx.y >= 1 && threadIdx.y < IN_TILE_Y - 1
            && threadIdx.z >= 1 && threadIdx.z < IN_TILE_Z - 1) {
            Output[Channel*(M*N) + Row*N + Col] = SValues_d.C0 * Mds[threadIdx.z][threadIdx.y][threadIdx.x] 
                                            + SValues_d.C1 * Mds[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                            + SValues_d.C2 * Mds[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                                            + SValues_d.C3 * Mds[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                            + SValues_d.C4 * Mds[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                            + SValues_d.C5 * Mds[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                            + SValues_d.C6 * Mds[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }

}

__global__
void CoarsStencil(float* Input, float* Output) {
    int ChannelStart = blockIdx.z * OUT_TILE_DIM;
    int Row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int Col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float Prev_S[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float Curr_S[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float Next_S[IN_TILE_DIM][IN_TILE_DIM];

    if(ChannelStart - 1 >= 0 && ChannelStart - 1 < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
        Prev_S[threadIdx.y][threadIdx.x] = Input[(ChannelStart-1)*(M*N) + Row*N + Col];
    }
    if(ChannelStart >= 0 && ChannelStart < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
        Curr_S[threadIdx.y][threadIdx.x] = Input[ChannelStart*(M*N) + Row*N + Col];
    }
    for(int Channel = ChannelStart; Channel < ChannelStart + OUT_TILE_DIM; ++Channel) {
        if(Channel + 1 >= 0 && Channel + 1 < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
            Next_S[threadIdx.y][threadIdx.x] = Input[(Channel+1)*(M*N) + Row*N + Col];
        }
        __syncthreads();

        if(Channel >= 1 && Channel < C - 1 && Row >= 1 && Row < M-1 && Col >= 1 && Col < N-1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                Output[Channel*(M*N) + Row*N + Col] = SValues_d.C0 * Curr_S[threadIdx.y][threadIdx.x]
                                                    + SValues_d.C1 * Curr_S[threadIdx.y][threadIdx.x-1]
                                                    + SValues_d.C2 * Curr_S[threadIdx.y][threadIdx.x+1]
                                                    + SValues_d.C3 * Curr_S[threadIdx.y-1][threadIdx.x]
                                                    + SValues_d.C4 * Curr_S[threadIdx.y+1][threadIdx.x]
                                                    + SValues_d.C5 * Prev_S[threadIdx.y][threadIdx.x]
                                                    + SValues_d.C6 * Next_S[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        Prev_S[threadIdx.y][threadIdx.x] = Curr_S[threadIdx.y][threadIdx.x];
        Curr_S[threadIdx.y][threadIdx.x] = Next_S[threadIdx.y][threadIdx.x];
    }
}

__global__
void RegTileStencil(float* Input, float* Output) {
    int ChannelStart = blockIdx.z * OUT_TILE_DIM;
    int Row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int Col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float Prev = 0.0f;
    float Curr = 0.0f;
    __shared__ float Curr_S[IN_TILE_DIM][IN_TILE_DIM];
    float Next = 0.0f;

    if(ChannelStart - 1 >= 0 && ChannelStart - 1 < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
        Prev = Input[(ChannelStart-1)*(M*N) + Row*N + Col];
    }
    if(ChannelStart >= 0 && ChannelStart < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
        Curr_S[threadIdx.y][threadIdx.x] = Input[ChannelStart*(M*N) + Row*N + Col];
        Curr = Curr_S[threadIdx.y][threadIdx.x];
    }
    for(int Channel = ChannelStart; Channel < ChannelStart + OUT_TILE_DIM; ++Channel) {
        if(Channel + 1 >= 0 && Channel + 1 < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
            Next = Input[(Channel+1)*(M*N) + Row*N + Col];
        }
        __syncthreads();

        if(Channel >= 1 && Channel < C - 1 && Row >= 1 && Row < M-1 && Col >= 1 && Col < N-1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1) {
                Output[Channel*(M*N) + Row*N + Col] = SValues_d.C0 * Curr
                                                    + SValues_d.C1 * Curr_S[threadIdx.y][threadIdx.x-1]
                                                    + SValues_d.C2 * Curr_S[threadIdx.y][threadIdx.x+1]
                                                    + SValues_d.C3 * Curr_S[threadIdx.y-1][threadIdx.x]
                                                    + SValues_d.C4 * Curr_S[threadIdx.y+1][threadIdx.x]
                                                    + SValues_d.C5 * Prev
                                                    + SValues_d.C6 * Next;
            }
        }
        __syncthreads();
        Prev = Curr;
        Curr = Next;
        Curr_S[threadIdx.y][threadIdx.x] = Next;
    }
}


int main(void) {

	srand(time(NULL));

	float* Mat1 = (float*)malloc(sizeof(float) * C * M * N);
    float* Res = (float*)malloc(sizeof(float) * C * M * N);
    StencilValues SValues;
	for (int i = 0; i < C * M * N; ++i) {
		Mat1[i] = rand() % RANGE + 1; 
	}
    for(int i = 0; i<7; ++i) {
        SValues.Values[i] = 1.0f / 7.0f; // dummy initialization
    }
    float* StencilResultCpu = StencilCpu(Mat1, &SValues);
	float* Mat_d1;
	float* Res_d;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Mat_d1, sizeof(float) * C * M * N));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Res_d, sizeof(float) * C * M * N));
	CHECK_CUDA_ERROR(cudaMemcpy(Mat_d1, Mat1, sizeof(float) * C * M * N, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(Res_d, Mat1, sizeof(float) * C * M * N, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(SValues_d, &SValues, sizeof(StencilValues)));

	cudaEvent_t Start, Stop;
	float Milliseconds = 0.0f;
	cudaEventCreate(&Start);
	cudaEventCreate(&Stop);

    // Naive
	dim3 GridDim((N + IN_TILE_DIM - 1) / IN_TILE_DIM, (M + IN_TILE_DIM - 1) / IN_TILE_DIM, C);
	dim3 BlockDim(IN_TILE_DIM, IN_TILE_DIM, 1);

	cudaEventRecord(Start);

    NaiveStencil<<<GridDim, BlockDim>>>(Mat_d1, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

    cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Naive): %f ms\n", Milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * C * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, StencilResultCpu, C*M*N, ERROR_MESSAGE);

    // Shared memory
    GridDim = {(N + OUT_TILE_X - 1) / OUT_TILE_X, (M + OUT_TILE_Y - 1) / OUT_TILE_Y, (C + OUT_TILE_Z - 1) / OUT_TILE_Z};
    BlockDim = {IN_TILE_X, IN_TILE_Y, IN_TILE_Z};
    CHECK_CUDA_ERROR(cudaMemcpy(Res_d, Mat1, sizeof(float) * C * M * N, cudaMemcpyHostToDevice));
    cudaEventRecord(Start);

    SharedStencil<<<GridDim, BlockDim>>>(Mat_d1, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Shared memory): %f ms\n", Milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * C * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, StencilResultCpu, C*M*N, ERROR_MESSAGE);

    // Thread coarsening
    GridDim = {(N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
            (M + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
            (C + OUT_TILE_DIM - 1) / OUT_TILE_DIM};
    BlockDim = {IN_TILE_DIM, IN_TILE_DIM, 1};
    CHECK_CUDA_ERROR(cudaMemcpy(Res_d, Mat1, sizeof(float) * C * M * N, cudaMemcpyHostToDevice));
    cudaEventRecord(Start);

    CoarsStencil<<<GridDim, BlockDim>>>(Mat_d1, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Thread coarsening): %f ms\n", Milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * C * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, StencilResultCpu, C*M*N, ERROR_MESSAGE);

    // Register tiling
    CHECK_CUDA_ERROR(cudaMemcpy(Res_d, Mat1, sizeof(float) * C * M * N, cudaMemcpyHostToDevice));
    cudaEventRecord(Start);

    RegTileStencil<<<GridDim, BlockDim>>>(Mat_d1, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

	cudaEventRecord(Stop);
	cudaEventSynchronize(Stop); 

	cudaEventElapsedTime(&Milliseconds, Start, Stop); 
	printf("Execution time (Register tiling): %f ms\n", Milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * C * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, StencilResultCpu, C*M*N, ERROR_MESSAGE);

	cudaEventDestroy(Start);
	cudaEventDestroy(Stop);
	cudaFree(Mat_d1);
	cudaFree(Res_d);

	free(Mat1);
    free(StencilResultCpu);
    free(Res);
	return 0;
}
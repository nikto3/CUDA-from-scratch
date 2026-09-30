#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "..\cuda_check.cuh"

#define FILTER_SIZE 5
#define BLOCK_SIZE 1024
#define OUT_TILE (BLOCK_SIZE - (2*FILTER_SIZE))
#define N (1 << 26)
#define RANGE 100


__constant__ float Kernel[2*FILTER_SIZE + 1];

float* Conv1DCpu(float* Vec, float* Kernel) {
    float* Res = (float*)malloc(sizeof(float) * N);
    for(int i = 0; i<N; ++i) {
        float Value = 0.0f;
        for(int j = -FILTER_SIZE; j < FILTER_SIZE + 1; ++j) {
            int Index = i + j;
            if(Index >= 0 && Index < N) {
                Value += Vec[Index]*Kernel[j+FILTER_SIZE];
            }
        }
        Res[i] = Value;
    }
    return Res;
}

__global__
void Conv1D(float* Vec, float* Res) {
    int Index = blockIdx.x * blockDim.x + threadIdx.x;

    if(Index < N) {
        float Value = 0.0f;
        for(int i = -FILTER_SIZE; i < FILTER_SIZE + 1; ++i) {
            int VecIndex = Index + i;
            if(VecIndex >= 0 && VecIndex < N) {
                Value += Vec[VecIndex]*Kernel[i+FILTER_SIZE]; 
            }
        }
        Res[Index] = Value;
    }
}

__global__
void Conv1DTiled(float* Vec, float* Res) {
    int OutIndex = blockIdx.x * OUT_TILE + threadIdx.x;
    int InIndex = OutIndex - FILTER_SIZE;

    __shared__ float Mds[BLOCK_SIZE];


    if(InIndex >= 0 && InIndex < N) {
        Mds[threadIdx.x] = Vec[InIndex];
    }
    else {
        Mds[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if(OutIndex < N && threadIdx.x < OUT_TILE) {
        float Value = 0.0f;
        for (int FilterIndex = 0; FilterIndex < 2*FILTER_SIZE + 1; ++FilterIndex) {
            Value += Kernel[FilterIndex]*Mds[threadIdx.x + FilterIndex];      
        }
        Res[OutIndex] = Value;
    }
   
}

__global__
void Conv1DTiledCache(float* Vec, float* Res) {
    int Index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float Mds[BLOCK_SIZE];

    if(Index < N) {
        Mds[threadIdx.x] = Vec[Index];
    }
    else {
        Mds[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if(Index < N) {
        float Value = 0.0f;
        for(int FilterIndex = 0; FilterIndex < 2*FILTER_SIZE + 1; ++FilterIndex) {
            if((int)threadIdx.x - FILTER_SIZE + FilterIndex >= 0 && (int)threadIdx.x - FILTER_SIZE + FilterIndex < BLOCK_SIZE) {
                Value += Mds[threadIdx.x - FILTER_SIZE +FilterIndex] * Kernel[FilterIndex];
            }
            else if(Index - FILTER_SIZE + FilterIndex >= 0 && Index - FILTER_SIZE + FilterIndex < N) {
                Value += Vec[Index - FILTER_SIZE + FilterIndex] * Kernel[FilterIndex];
            }
        }
        Res[Index] = Value;
    }
}

int main() {
    srand(time(NULL));

    float* Vec = (float*)malloc(sizeof(float) * N);
    float* Res = (float*)malloc(sizeof(float) * N);
    for(int i = 0; i<N; ++i) {
        Vec[i] = rand() % RANGE;
    }
    float Kernel_h[2*FILTER_SIZE + 1];
    for(int i = 0; i<2*FILTER_SIZE + 1; ++i) {
        Kernel_h[i] = 0.1f * i;
    }
    float* FeatureVec = Conv1DCpu(Vec, Kernel_h);
    float* Vec_d;
    float* Res_d;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&Vec_d, sizeof(float) * N));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Kernel, Kernel_h, sizeof(float) * (2*FILTER_SIZE + 1)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&Res_d, sizeof(float) * N));
    CHECK_CUDA_ERROR(cudaMemcpy(Vec_d, Vec, sizeof(float) * N, cudaMemcpyHostToDevice));

    dim3 GridDim((N + OUT_TILE - 1) / OUT_TILE, 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);

    cudaEvent_t Start, Stop;
    float Milliseconds = 0.0f;;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    // Naive
    cudaEventRecord(Start);
    Conv1D<<<GridDim, BlockDim>>>(Vec_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Naive): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * N, cudaMemcpyDeviceToHost));

    Compare(Res, FeatureVec, N, "Error while applying 1d convolutional filter to vector.");

    // Tiled
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    cudaEventRecord(Start);
    Conv1DTiled<<<GridDim, BlockDim>>>(Vec_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Tiled): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * N, cudaMemcpyDeviceToHost));

    Compare(Res, FeatureVec, N, "Error while applying 1d convolutional filter to vector.");

    // Tiled + Cache
    GridDim.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    cudaEventRecord(Start);
    Conv1DTiledCache<<<GridDim, BlockDim>>>(Vec_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Tiled + Cache): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * N, cudaMemcpyDeviceToHost));

    Compare(Res, FeatureVec, N, "Error while applying 1d convolutional filter to vector.");

    cudaFree(Vec_d);
    cudaFree(Res_d);
    free(Vec);
    free(Res);
    free(FeatureVec);
    return 0;
}
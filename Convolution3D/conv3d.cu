#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "..\cuda_check.cuh"

#define FILTER_SIZE 5
#define BLOCK_SIZE 32
#define OUT_TILE (BLOCK_SIZE - (2*FILTER_SIZE))
#define C 3     // Channel
#define M 1024  // Height
#define N 1024  // Width
#define RANGE 10


__constant__ float Kernel[2*FILTER_SIZE + 1][2*FILTER_SIZE + 1][2*FILTER_SIZE + 1]; // in constant memory
__constant__ float KernelDW[C][2*FILTER_SIZE + 1][2*FILTER_SIZE + 1];

// Volumetric 3D convolution executing on cpu
float* Conv3DCpu(float* Mat, float Kernel[][2*FILTER_SIZE + 1][2*FILTER_SIZE + 1]) {
    float* Res = (float*)malloc(sizeof(float) * C * M * N);

    for(int k = 0; k<C; ++k) {
        for(int i = 0; i<M; ++i) {
            for(int j = 0; j<N; ++j) {
                float Value = 0.0f;
                for(int FilterChann = -FILTER_SIZE; FilterChann < FILTER_SIZE + 1; ++FilterChann) {
                    for(int FilterRow = -FILTER_SIZE; FilterRow < FILTER_SIZE + 1; ++FilterRow) {
                        for(int FilterCol = -FILTER_SIZE; FilterCol < FILTER_SIZE + 1; ++FilterCol) {
                            int Channel = k + FilterChann;
                            int Row = i + FilterRow;
                            int Col = j + FilterCol;
                            if(Channel >= 0 && Channel < C && Row >= 0 && Row < M && Col >= 0 && Col < N) {
                                Value += Mat[Channel*(M*N) + Row*N + Col] * Kernel[FilterChann+FILTER_SIZE][FilterRow+FILTER_SIZE][FilterCol+FILTER_SIZE];
                            }
                        }
                    }
                }
                Res[k*(M*N) + i*N + j] = Value;
            }
        }
    }
    
    return Res;
}

// Depthwise 3d convolution, executing on cpu
float* Conv3DCpuDW(float* Mat, float Kernel[][2*FILTER_SIZE + 1][2*FILTER_SIZE + 1]) {
    float* Res = (float*)malloc(sizeof(float) * C * M * N);

    for(int k = 0; k<C; ++k) {
        for(int i = 0; i<M; ++i) {
            for(int j = 0; j<N; ++j) {
                float Value = 0.0f;
                for(int FilterRow = -FILTER_SIZE; FilterRow < FILTER_SIZE + 1; ++FilterRow) {
                    for(int FilterCol = -FILTER_SIZE; FilterCol < FILTER_SIZE + 1; ++FilterCol) {
                        int Row = i + FilterRow;
                        int Col = j + FilterCol;
                        if(Row >= 0 && Row < M && Col >= 0 && Col < N) {
                            Value += Mat[k*(M*N) + Row*N + Col] * Kernel[k][FilterRow+FILTER_SIZE][FilterCol+FILTER_SIZE];
                        }
                    }
                }
                Res[k*(M*N) + i*N + j] = Value;
            }
        }
    }
    return Res;
}    

// Volumetric 3D convolution
__global__
void Conv3D(float* Mat, float* Res) {
    int Channel = blockIdx.z*blockDim.z + threadIdx.z;
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    if(Channel < C && Row < M && Col < N) {
        float Value = 0.0f;
        for(int FilterChannel = -FILTER_SIZE; FilterChannel < FILTER_SIZE + 1; ++FilterChannel) {
            for(int FilterRow = -FILTER_SIZE; FilterRow < FILTER_SIZE + 1; ++FilterRow) {
                for(int FilterCol = -FILTER_SIZE; FilterCol < FILTER_SIZE + 1; ++FilterCol) {
                    int InChannel = Channel + FilterChannel;
                    int InRow = Row + FilterRow;
                    int InCol = Col + FilterCol;
                    if(InChannel >= 0 && InChannel < C && InRow >= 0 && InRow < M && InCol >= 0 && InCol < N) {
                        Value += Mat[InChannel*(M*N) + InRow*N + InCol]*Kernel[FilterChannel+FILTER_SIZE][FilterRow+FILTER_SIZE][FilterCol+FILTER_SIZE];
                    }
                }
            }
        }
        Res[Channel*(M*N) + Row*N + Col] = Value;
    }

}

// Depthwise Tiled Convolution
__global__
void Conv3DDW(float* Mat, float* Res) {
    int OutRow = blockIdx.y * OUT_TILE + threadIdx.y;
    int OutCol = blockIdx.x * OUT_TILE + threadIdx.x;
    int InRow = OutRow - FILTER_SIZE;
    int InCol = OutCol - FILTER_SIZE;

    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];

    for(int Channel = 0; Channel < C; ++Channel) {
        float Value = 0.0f;
        if(InRow >= 0 && InRow < M && InCol >= 0 && InCol < N) {
            Mds[threadIdx.y][threadIdx.x] = Mat[Channel*(M*N) + InRow*N + InCol];
        }
        else {
            Mds[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        if(threadIdx.x < OUT_TILE && threadIdx.y < OUT_TILE) {
            for(int FilterRow = 0; FilterRow < 2*FILTER_SIZE+1; ++FilterRow) {
                for(int FilterCol = 0; FilterCol < 2*FILTER_SIZE+1; ++FilterCol) {
                        Value += Mds[threadIdx.y+FilterRow][threadIdx.x+FilterCol]*KernelDW[Channel][FilterRow][FilterCol];
                }
            }
        }
        __syncthreads();
        if(OutRow < M && OutCol < N && threadIdx.x < OUT_TILE && threadIdx.y < OUT_TILE) {
            Res[Channel*(M*N) + OutRow*N + OutCol] = Value;
        }
    }
}

int main() {
    srand(time(NULL));

    float* Mat = (float*)malloc(sizeof(float) * C * M * N);
    float* Res = (float*)malloc(sizeof(float) * C * M * N);
    for(int i = 0; i<C*M*N; ++i) {
        Mat[i] = rand() % RANGE * 0.1f;
    }
    float Kernel_h[2*FILTER_SIZE + 1][2*FILTER_SIZE + 1][2*FILTER_SIZE + 1];
    for(int i = 0; i<2*FILTER_SIZE + 1; ++i) {
        for(int j = 0; j<2*FILTER_SIZE + 1; ++j) {
            for(int k = 0; k<2*FILTER_SIZE + 1; ++k) {
                Kernel_h[i][j][k] = 1.0f / (2*FILTER_SIZE+1)*(2*FILTER_SIZE+1)*(2*FILTER_SIZE+1);
            }
        }
    }
    float* FeatureMat = Conv3DCpu(Mat, Kernel_h);
    float* FeatureMatDW = Conv3DCpuDW(Mat, Kernel_h);
    float* Mat_d;
    float* Res_d;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&Mat_d, sizeof(float) * C * M * N));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Kernel, Kernel_h, sizeof(float) * (2*FILTER_SIZE + 1)*(2*FILTER_SIZE + 1)*(2*FILTER_SIZE + 1)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(KernelDW, Kernel_h, sizeof(float) * C *(2*FILTER_SIZE + 1)*(2*FILTER_SIZE + 1)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&Res_d, sizeof(float) * C * M * N));
    CHECK_CUDA_ERROR(cudaMemcpy(Mat_d, Mat, sizeof(float) * C * M * N, cudaMemcpyHostToDevice));

    dim3 GridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, C);
    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    cudaEvent_t Start, Stop;
    float Milliseconds = 0.0f;;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    // Naive
    cudaEventRecord(Start);
    Conv3D<<<GridDim, BlockDim>>>(Mat_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Naive): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * C * M * N, cudaMemcpyDeviceToHost));
    Compare(Res, FeatureMat, C*M*N, "Error while applying 3d convolutional filter to matrix.");

    // Depthwise Tiled
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    GridDim.x = (N + OUT_TILE - 1) / OUT_TILE;
    GridDim.y = (M + OUT_TILE - 1) / OUT_TILE;
    GridDim.z = C;
    cudaEventRecord(Start);
    Conv3DDW<<<GridDim, BlockDim>>>(Mat_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Tiled DepthWise): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * C * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, FeatureMatDW, C*M*N, "Error while applying 3d convolutional filter to matrix.");

    cudaFree(Mat_d);
    cudaFree(Res_d);
    free(Mat);
    free(Res);
    free(FeatureMat);
    free(FeatureMatDW);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "..\cuda_check.cuh"

#define FILTER_SIZE 5
#define BLOCK_SIZE 32
#define OUT_TILE (BLOCK_SIZE - (2*FILTER_SIZE))
#define M 1024
#define N 4096
#define RANGE 100


__constant__ float Kernel[2*FILTER_SIZE + 1][2*FILTER_SIZE + 1]; // in constant memory

float* Conv2DCpu(float* Mat, float Kernel[][2*FILTER_SIZE + 1]) {
    float* Res = (float*)malloc(sizeof(float) * M * N);

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float Value = 0.0f;
            for(int KernelRow = -FILTER_SIZE; KernelRow < FILTER_SIZE + 1; ++KernelRow) {
                for(int KernelCol = -FILTER_SIZE; KernelCol < FILTER_SIZE + 1; ++KernelCol) {
                    int Row = i + KernelRow;
                    int Col = j + KernelCol;
                    if(Row >= 0 && Row < M && Col >= 0 && Col < N) {
                        Value += Mat[Row*N + Col] * Kernel[KernelRow+FILTER_SIZE][KernelCol+FILTER_SIZE];
                    }
                }
            }
            Res[i*N + j] = Value;
        }
    }
    return Res;
}

__global__
void Conv2D(float* Mat, float* Res) {
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    if(Row < M && Col < N) {
        float Value = 0.0f;
        for(int i = -FILTER_SIZE; i < FILTER_SIZE + 1; ++i) {
            for(int j = -FILTER_SIZE; j < FILTER_SIZE + 1; ++j) {
                int r = Row + i;
                int c = Col + j;
                if(r >= 0 && r < M && c >= 0 && c < N) {
                    Value += Mat[r*N + c] * Kernel[i+FILTER_SIZE][j+FILTER_SIZE];
                }
            }
        }
        Res[Row*N + Col] = Value;
    }

}

__global__
void Conv2DTiled(float* Mat, float* Res) {
    int OutRow = blockIdx.y * OUT_TILE + threadIdx.y;
    int OutCol = blockIdx.x * OUT_TILE + threadIdx.x;

    int InRow = OutRow - FILTER_SIZE;
    int InCol = OutCol - FILTER_SIZE;

    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];

    if(InRow >= 0 && InRow < M && InCol >= 0 && InCol < N) {
        Mds[threadIdx.y][threadIdx.x] = Mat[InRow*N + InCol];
    }
    else {
        Mds[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads(); // wait for all the threads to finish loading data!

    if(OutRow < M && OutCol < N && threadIdx.x < OUT_TILE && threadIdx.y < OUT_TILE) {
        float Value=0.0f;
        for(int FilterRow = 0; FilterRow < 2*FILTER_SIZE + 1; ++FilterRow) {
            for(int FilterCol = 0; FilterCol < 2*FILTER_SIZE + 1; ++FilterCol) {
                Value += Mds[threadIdx.y+FilterRow][threadIdx.x+FilterCol]*Kernel[FilterRow][FilterCol];
            }
        }
        Res[OutRow*N + OutCol] = Value;
    }
}

int main() {
    srand(time(NULL));

    float* Mat = (float*)malloc(sizeof(float) * M * N);
    float* Res = (float*)malloc(sizeof(float) * M * N);
    for(int i = 0; i<M*N; ++i) {
        Mat[i] = rand() % RANGE;
    }
    float Kernel_h[2*FILTER_SIZE + 1][2*FILTER_SIZE + 1];
    for(int i = 0; i<2*FILTER_SIZE + 1; ++i) {
        for(int j = 0; j<2*FILTER_SIZE + 1; ++j) {
            Kernel_h[i][j] = 0.1f * (i+j);
        }
    }
    float* FeatureMat = Conv2DCpu(Mat, Kernel_h);
    float* Mat_d;
    float* Res_d;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&Mat_d, sizeof(float) * M * N));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Kernel, Kernel_h, sizeof(float) * (2*FILTER_SIZE + 1)*(2*FILTER_SIZE + 1)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&Res_d, sizeof(float) * M * N));
    CHECK_CUDA_ERROR(cudaMemcpy(Mat_d, Mat, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    dim3 GridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    cudaEvent_t Start, Stop;
    float Milliseconds = 0.0f;;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    // Naive
    cudaEventRecord(Start);
    Conv2D<<<GridDim, BlockDim>>>(Mat_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Naive): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, FeatureMat, M*N, "Error while applying 2d convolutional filter to matrix.");

    // Tiled
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    GridDim.x = (N + OUT_TILE - 1) / OUT_TILE;
    GridDim.y = (M + OUT_TILE - 1) / OUT_TILE;
    cudaEventRecord(Start);
    Conv2DTiled<<<GridDim, BlockDim>>>(Mat_d, Res_d);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaEventRecord(Stop);
    cudaEventSynchronize(Stop);

    cudaEventElapsedTime(&Milliseconds, Start, Stop);
    printf("Execution time (Tiled): %f ms\n", Milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(Res, Res_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    Compare(Res, FeatureMat, M*N, "Error while applying 2d convolutional filter to matrix.");
    cudaFree(Mat_d);
    cudaFree(Res_d);
    free(Mat);
    free(Res);
    free(FeatureMat);
    return 0;
}
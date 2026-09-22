#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N (1 << 24)
#define BLOCK_SIZE 128
#define RANGE 100

#define CHECK_CUDA_ERROR(Val) Check((Val), #Val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() CheckLast(__FILE__, __LINE__) 

inline 
void Compare(float* Mat1, float* Mat2) {
  for(int i = 0; i<N; ++i){
    if(fabs(Mat1[i] - Mat2[i]) >= 1e-6) {
      printf("Error while adding vectors.\n");
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

float* VecAddCpu(float* A, float* B) {
  float* C = (float*)malloc(sizeof(float) * N);

  for(int i = 0; i<N; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

__global__ void VecAdd(float *A, float *B, float *C)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
  {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, const char *argv[])
{
  srand(time(NULL));

  float *A = (float*)malloc(sizeof(float) * N);
  float *B = (float*)malloc(sizeof(float) * N);
  float *C = (float*)malloc(sizeof(float) * N);

  for (int i = 0; i<N; ++i)
  {
    A[i] = rand() % RANGE;
    B[i] = rand() % RANGE;
    C[i] = 0;
  }
  float* VecAddResult = VecAddCpu(A, B);

  float *A_d;
  float *B_d;
  float *C_d;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeof(float) * N)); 
  CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeof(float) * N));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeof(float) * N));
  CHECK_CUDA_ERROR(cudaMemcpy(A_d, A, sizeof(float) * N, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(B_d, B, sizeof(float) * N, cudaMemcpyHostToDevice));

  cudaEvent_t Start, Stop;
  float Milliseconds = 0.0f;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);

  dim3 GridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);

  cudaEventRecord(Start);
  VecAdd<<<GridDim, BlockDim>>>(A_d, B_d, C_d);    

  cudaDeviceSynchronize();
  CHECK_LAST_CUDA_ERROR();

  cudaEventRecord(Stop);
  cudaEventSynchronize(Stop); 
  cudaEventElapsedTime(&Milliseconds, Start, Stop); 
  printf("Execution time: %f ms\n", Milliseconds);
  
  CHECK_CUDA_ERROR(cudaMemcpy(C, C_d, sizeof(float) * N, cudaMemcpyDeviceToHost));
  Compare(C, VecAddResult);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  free(A);
  free(B);
  free(C);
  free(VecAddResult);
  return 0;
}

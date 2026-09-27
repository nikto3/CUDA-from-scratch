#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H
#include <stdio.h>
#include <math.h>
#define CHECK_CUDA_ERROR(Val) Check((Val), #Val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() CheckLast(__FILE__, __LINE__) 


inline
void Check(cudaError_t Err, const char* Func, const char* File, int Line) {
	if(Err != cudaSuccess) {
		printf("Cuda Runtime Error At %s: %d\n", File, Line);
		printf("%s %s\n", cudaGetErrorString(Err), Func);
	}
}  

inline
void CheckLast(const char* File, int Line) {
	cudaError_t Err = cudaGetLastError();
	if(Err != cudaSuccess) {
		printf("Cuda Runtime Error At %s: %d\n", File, Line);
		printf("%s\n", cudaGetErrorString(Err));
	}
}

inline
void Compare(float* M, float* N, size_t Length, const char* Message) {
    for(size_t i = 0; i < Length; ++i) {
        if(fabsf(M[i] - N[i]) >= 1e-3) {
            printf("%s\n", Message);
            exit(-1);
        }
    }
}

#endif
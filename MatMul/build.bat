@echo off

if not exist build mkdir build
pushd build

nvcc ..\naive_matmul.cu --ptxas-options=-v -arch=sm_61 -o matmul.exe

popd

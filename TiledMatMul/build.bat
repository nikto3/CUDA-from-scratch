@echo off

if not exist build mkdir build
pushd build

nvcc ..\tiled_matmul.cu --ptxas-options=-v -arch=sm_61 -o tiled_matmul.exe

popd

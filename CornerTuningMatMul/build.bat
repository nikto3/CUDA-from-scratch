@echo off

if not exist build mkdir build
pushd build

nvcc ..\corner_turning_matmul.cu --ptxas-options=-v -arch=sm_61 -o corner_turning_matmul.exe

popd

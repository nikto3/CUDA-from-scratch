@echo off

if not exist build mkdir build
pushd build

nvcc ..\mat_transpose.cu --ptxas-options=-v -arch=sm_61 -lineinfo -o mat_transpose.exe

popd

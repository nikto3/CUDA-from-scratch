@echo off

if not exist build mkdir build
pushd build

nvcc ..\coars_tiled_matmul.cu --ptxas-options=-v -arch=sm_61 -o coars_tiled_matmul.exe

popd

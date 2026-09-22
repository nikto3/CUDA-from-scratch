@echo off

if not exist build mkdir build
pushd build

nvcc ..\vec_add.cu --ptxas-options=-v -arch=sm_61 -o vec_add.exe

popd

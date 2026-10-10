@echo off

if not exist build mkdir build 
pushd build 

nvcc ..\stencil.cu --ptxas-options=-v -arch=sm_61 -o stencil.exe

popd

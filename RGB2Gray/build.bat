@echo off

if not exist build mkdir build
pushd build

nvcc ..\rgb2gray.cu -I ..\..\stb --ptxas-options=-v -arch=sm_61 -o rgb2gray.exe

popd

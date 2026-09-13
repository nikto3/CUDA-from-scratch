@echo off

if not exist build mkdir build
pushd build

nvcc ..\blur.cu -I ..\..\stb --ptxas-options=-v -arch=sm_61 -o blur.exe

popd

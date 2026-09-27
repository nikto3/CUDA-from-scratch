@echo off

if not exist build mkdir build 
pushd build 

nvcc ..\conv2d.cu --ptxas-options=-v -arch=sm_61 -o conv2d.exe

popd

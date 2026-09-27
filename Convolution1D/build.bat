@echo off

if not exist build mkdir build 
pushd build 

nvcc ..\conv1d.cu --ptxas-options=-v -arch=sm_61 -o conv1d.exe

popd

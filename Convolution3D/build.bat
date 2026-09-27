@echo off

if not exist build mkdir build 
pushd build 

nvcc ..\conv3d.cu --ptxas-options=-v -arch=sm_61 -o conv3d.exe

popd

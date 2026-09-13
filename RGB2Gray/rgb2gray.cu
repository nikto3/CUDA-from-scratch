#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void rgb2grayscale(unsigned char *img, unsigned char *gray, int x, int y, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < y && col < x)
  {
    int grayOffset = row * x + col;

    int rgbOffset = grayOffset * n;
    unsigned char r = img[rgbOffset];
    unsigned char g = img[rgbOffset + 1];
    unsigned char b = img[rgbOffset + 2];

    gray[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b; 
  }
}

int main(int argc, const char *argv[])
{
  int x, y, n;
  unsigned char *img = stbi_load("..\\img.jpg", &x, &y, &n, 0); // 3 zbog rgb

  if(img == NULL)
  {
    printf("Image could not be loaded!\n");
    return -1;
  }
  else
  {
    printf("Image loaded successfully!\n");
  }
      
  unsigned char *img_d;
  unsigned char *grayscale = (unsigned char *)malloc(sizeof(unsigned char) * y * x);
  unsigned char *grayscale_d;
  cudaMalloc((void **)&img_d, sizeof(unsigned char) * y * x * n);
  cudaMalloc((void **)&grayscale_d, sizeof(unsigned char) * y * x);
  cudaMemcpy(img_d, img, sizeof(unsigned char) * y * x * n, cudaMemcpyHostToDevice);


  dim3 dimGrid((x + 32 - 1)/32, (y + 32 - 1) / 32, 1);
  dim3 dimBlock(32, 32, 1);
  rgb2grayscale<<<dimGrid, dimBlock>>>(img_d, grayscale_d, x, y, n);
  cudaMemcpy(grayscale, grayscale_d, sizeof(unsigned char) * y * x, cudaMemcpyDeviceToHost);

  
  if(stbi_write_jpg("img_gray.jpg", x, y, 1, (void *)grayscale, 100) == 0)
  {
    printf("Image was not saved!\n");
    return -1;
  }
  else
  {
    printf("Image was saved!\n");
  }
    
  stbi_image_free(img);
  cudaFree(img_d);
  cudaFree(grayscale_d);
  free(grayscale);
  
  return 0;
}

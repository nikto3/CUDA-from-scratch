#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include <stdlib.h>
#include <stdio.h>
#define BLUR_SIZE 3

__global__ void blur(unsigned char *image, unsigned char *image_blur, int x, int y, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(col < x && row < y)
  {
    int rCount = 0, gCount = 0, bCount = 0;
    int pixelCount = 0;
    int offset;
    for(int i = -BLUR_SIZE; i < BLUR_SIZE + 1; ++i)
    {
      for(int j = -BLUR_SIZE; j < BLUR_SIZE + 1; ++j)
      {
        int blurRow = row + i;
        int blurCol = col + j;
        if(blurRow >= 0 && blurRow < y && blurCol >= 0 && blurCol < x)
        {
          offset = (blurRow*x + blurCol) * n;
          rCount += image[offset];
          gCount += image[offset + 1];
          bCount += image[offset + 2];
          ++pixelCount;
        }
      }
    }
    offset = (row*x + col) * n;
    image_blur[offset]     = (unsigned char)(rCount / pixelCount);
    image_blur[offset + 1] = (unsigned char)(gCount / pixelCount);
    image_blur[offset + 2] = (unsigned char)(bCount / pixelCount);
  }
}

int main()
{
  int x, y, n;
  unsigned char *image = stbi_load("..\\img.jpg", &x, &y, &n, 0);
  unsigned char *image_result = (unsigned char *)malloc(sizeof(unsigned char) * y * x * n);
  if(image == NULL)
  {
    printf("Image could not be loaded!\n");
    return -1;
  }
  else {
    printf("Image was loaded!\n");
  }
  unsigned char *image_d;
  unsigned char *image_blur;
  cudaMalloc((void **)&image_blur, sizeof(unsigned char) * y * x * n);
  cudaMalloc((void **)&image_d, sizeof(unsigned char) * y * x * n);
  cudaMemcpy(image_d, image, sizeof(unsigned char) * y * x * n, cudaMemcpyHostToDevice);

  dim3 dimGrid((x + 32 - 1)/32, (y + 32 - 1)/32, 1);
  dim3 dimBlock(32, 32, 1);

  blur<<<dimGrid, dimBlock>>>(image_d, image_blur, x, y, n);

  cudaMemcpy(image_result, image_blur, sizeof(unsigned char) * y * x * n, cudaMemcpyDeviceToHost);

  if(stbi_write_png("blur.png", x, y, n, (void *)image_result, x*n) == 0)
  {
    printf("Image was not saved!\n");
    return -1;
  }
  else
  {
    printf("Image was saved!\n");
  }
  stbi_image_free(image);
  cudaFree(image_d);
  cudaFree(image_blur);
  free(image_result);
  return 0;
}

/*
 * thread_block_test.cu
 * Copyright (C) 2016  <@A0835-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <cstdlib>

int N = 21504;

int blockPerGrid(const int dim, const int threadPerBlock)
{
   int temp = dim / threadPerBlock;
   if (dim % threadPerBlock != 0) {
      temp += 1; 
   }
   return temp;
}

__device__ void initial_a(int *a, int tid)
{
    a[tid] = tid * tid;
}

__device__ void initial_b(int *b)
{
    int tid = threadIdx.x;
    b[tid] = -tid;
}
__global__ void add(long long *a, long long *b, long long *c, int *bdim, int *gdim)
{
//    int tid = blockIdx.x;
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
//    while (tid < N) {
        a[tid] = tid * tid;
        b[tid] = -tid;
        c[tid] = a[tid] + b[tid];
//        tid += blockDim.x * gridDim.x;
//    }
    if (tid == 0) {
        *bdim = blockDim.x;
        *gdim = gridDim.x;
    }
}

int main(int argc, char **argv)
{
    if (argc > 1)
        N = atoi(argv[1]);
    long long *c = new long long[N];
    int bdim, gdim;
    long long *dev_a, *dev_b, *dev_c;
    int *dev_bdim, *dev_gdim;
    cudaMalloc(&dev_a, N * sizeof(long long));
    cudaMalloc(&dev_b, N * sizeof(long long));
    cudaMalloc(&dev_c, N * sizeof(long long));
    cudaMalloc(&dev_bdim, sizeof(int));
    cudaMalloc(&dev_gdim, sizeof(int));
    //    add<<<N, 1>>>(dev_a, dev_b, dev_c);
    int threadPerBlock = 256;
    int blockSize = blockPerGrid(N, threadPerBlock);
    add<<<blockSize, threadPerBlock>>>(dev_a, dev_b, dev_c, dev_bdim, dev_gdim);
    cudaMemcpy(c, dev_c, N * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bdim, dev_bdim, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gdim, dev_gdim, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_bdim);
    cudaFree(dev_gdim);
    for (int i = 0; i < N; ++i) {
        std::cout << i << ": " << c[i] << std::endl; 
    }
    std::cout << "blockDim.x: " << bdim << std::endl;
    std::cout << "gridDim.x: " << gdim << std::endl;
    free(c);
    return 0;
}



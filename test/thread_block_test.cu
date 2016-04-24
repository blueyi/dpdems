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

inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
   if (error != cudaSuccess) {
      std::cerr << "CUDA CALL FAILED: " << file << "( " << line << " )- " <<
         cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
   else
      std::cout << "cuda call success" << std::endl;
}

inline void checkCudaState(const char *msg, const char *file, const int line)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
      std::cerr << "---" << msg << " Error--" << std::endl;
      std::cerr << file << "( " << line << " )- " << 
         cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
   else
      std::cout << "cuda state Success: " << msg << std::endl;
}

#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__);
#define CHECK_STATE(msg) checkCudaState(msg, __FILE__, __LINE__);

void print_device(const int id)
{
   cudaDeviceProp props;
   CHECK_ERROR(cudaGetDeviceProperties(&props, id));
   std::cout << "---Property of currently device used---" << std::endl;
   std::cout << "Device " << id << ": " << props.name << std::endl;
   std::cout << "CUDA Capability: " << props.major << "." << props.minor
      << std::endl;
   std::cout << "MultiProcessor count: " << props.multiProcessorCount << std::endl;
}

void setCudaDevice(int id)
{
   int numDevice = 0;
   CHECK_ERROR(cudaGetDeviceCount(&numDevice));
   std::cout << "Total CUDA device number: " << numDevice << std::endl;
   if (numDevice > 1) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, id);
      int maxMultiProcessors = props.multiProcessorCount;
      for (int device = 1; device < numDevice; ++device) {
         CHECK_ERROR(cudaGetDeviceProperties(&props, device));
         if (maxMultiProcessors < props.multiProcessorCount) {
            maxMultiProcessors = props.multiProcessorCount;
            id = device;
         }
      }
   }
   CHECK_ERROR(cudaSetDevice(id));
   print_device(id);
}


int main(int argc, char **argv)
{
   int id = 0;
   setCudaDevice(id);
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



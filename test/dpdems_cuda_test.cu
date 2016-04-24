/*
 * dpdems_cuda.cu
 * Copyright (C) 2016  <@A0835-PC>
 *
 * Distributed under terms of the MIT license.
 */
#include <iostream>
#include <cctype>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int maxdim = 100;
const int maxThreads = 21504;
const int threadPerBlock = 512;
int blockPerGrid(const int dim, const int threadPerBlock)
{
   return (dim + threadPerBlock - 1) / threadPerBlock;
}

inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
   if (error != cudaSuccess) {
      std::cerr << "CUDA CALL FAILED: " << file << "( " << line << " )- " <<
         cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
//   else
//      std::cout << "cuda call success" << std::endl;
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
//   else
//      std::cout << "cuda state Success: " << msg << std::endl;
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

__global__ void cudaScale(double *dev_xt, double *dev_yt, double *dev_zt, int *dev_x, int *dev_y, int *dev_z, int readnum, int maxdim)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   while (tid < readnum) {
      dev_x[tid] = dev_xt[tid] * dev_xt[readnum] + maxdim;
      dev_y[tid] = dev_yt[tid] * dev_yt[readnum] + maxdim;
      dev_z[tid] = dev_zt[tid] * dev_zt[readnum] + maxdim;
      tid += blockDim.x * gridDim.x;
   }
}

double scalev(double &, const double &);
void swapv(double *, double *, double *, int, int, double);
bool isInGrid(const int &, const int &, const int &, const int &);
unsigned updatePosition(int *, int *, int *, double *, double *, double *, const int &, const int &, int ***, const unsigned long &);
unsigned long long collision(int *, int *, int *, double *, double *, double *, const int &, const int &, int ***, const unsigned long &, std::ostream &);

int main(int argc, char **argv)
{
   std::cout.setf(std::ios::scientific);
   std::cout.precision(19);

   int device_id = 0;
   setCudaDevice(device_id);
   std::size_t readnum = 100;

   clock_t t;
   t = clock();

   cudaEvent_t start, stop;
   CHECK_STATE("cudaEvent1");
   CHECK_ERROR(cudaEventCreate(&start));
   CHECK_STATE("cudaEvent2");
   CHECK_ERROR(cudaEventCreate(&stop));
   CHECK_ERROR(cudaEventRecord(start, 0));
   CHECK_ERROR(cudaEventSynchronize(start));

   int *x = new int[readnum];
   int *y = new int[readnum];
   int *z = new int[readnum];
   int *dev_x;
   int *dev_y;
   int *dev_z;
   double *dev_xt;
   double *dev_yt;
   double *dev_zt;
   CHECK_ERROR(cudaMalloc((void**)&dev_x, readnum * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**)&dev_y, readnum * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**)&dev_z, readnum * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**)&dev_xt, (readnum + 1) * sizeof(double)));
   CHECK_ERROR(cudaMalloc((void**)&dev_yt, (readnum + 1) * sizeof(double)));
   CHECK_ERROR(cudaMalloc((void**)&dev_zt, (readnum + 1) * sizeof(double)));

   int threads = threadPerBlock;
   int blocks = blockPerGrid(readnum, threads);
//   cudaScale<<<blocks, threads>>>(dev_xt, dev_yt, dev_zt, dev_x, dev_y, dev_z, readnum, maxdim);
   CHECK_STATE("cudaScale call");
   CHECK_ERROR(cudaMemcpy(x, dev_x, readnum * sizeof(int), cudaMemcpyDeviceToHost));
   CHECK_ERROR(cudaMemcpy(y, dev_y, readnum * sizeof(int), cudaMemcpyDeviceToHost));
   CHECK_ERROR(cudaMemcpy(z, dev_z, readnum * sizeof(int), cudaMemcpyDeviceToHost));
   CHECK_ERROR(cudaFree(dev_x));
   CHECK_ERROR(cudaFree(dev_y));
   CHECK_ERROR(cudaFree(dev_z));
   CHECK_ERROR(cudaFree(dev_xt));
   CHECK_ERROR(cudaFree(dev_yt));
   CHECK_ERROR(cudaFree(dev_zt));

   CHECK_ERROR(cudaEventRecord(stop, 0));
   CHECK_ERROR(cudaEventSynchronize(stop));
   float elapsedTime;
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
   CHECK_ERROR(cudaEventDestroy(start));
   CHECK_ERROR(cudaEventDestroy(stop));
   std::cout << "CUDA elapsed: " << elapsedTime / 1000.0 << std::endl;

   int ***grid;
   grid = new int **[maxdim];
   for (int i = 0; i < maxdim; ++i) {
      grid[i] = new int *[maxdim];
      for (int j = 0; j < maxdim; ++j) {
         grid[i][j] = new int[maxdim];
         for (int k = 0; k < maxdim; ++k)
            grid[i][j][k] = 0;
      }
   }
   std::cout << *(x + 0) << " " << *(y + 0) << " " << *(z + 0) << std::endl;
   for (int i = 0; i < readnum; ++i) {
      grid[*(x + i)][*(y + i)][*(z + i)] = i;
   }

   std::cout << x[0] << " : " << y[0] << " : " << z[0] << std::endl;

   t = clock() - t;
   double seconds = (double)t / CLOCKS_PER_SEC;

   std::cout << std::endl << "Total time consumed: " << seconds << " seconds" << std::endl;

   delete [] x;
   std::cout << "delete x" << std::endl;
//   delete [] y;
   std::cout << "delete y" << std::endl;
//   delete [] z;
   std::cout << "delete z" << std::endl;
   for (int i = 0; i < maxdim; ++i) {
      for (int j = 0; j < maxdim; ++j) {
         delete[] grid[i][j];
      }
      delete[] grid[i];
   }
   delete[] grid;
   std::cout << "delete grid" << std::endl;
   std::cout << std::endl;
   return 0;
}

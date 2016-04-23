/*
 * dpdems_cuda.cu
 * Copyright (C) 2016  <@A0835-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "..\include\inputDatas.h"
#include "..\include\config.h"
#include <iostream>
#include <fstream>
#include <cctype>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


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

void init(std::vector<double *>&, const std::vector<Particle>&);

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



int main(int argc, char **argv)
{
   std::cout.setf(std::ios::scientific);
   std::cout.precision(19);
   std::string configFile = "config.txt";

   if (ini_conf(configFile.c_str()))
      std::cout << "Success" << std::endl;
   else 
      std::cout << "ini_conf error!" << std::endl;

   std::string ifileName = dataFile;
   std::ifstream inf;
   if (2 > argc){
      std::cout << "Use the default input file name from config.txt: inputdatas.txt" << std::endl;
   }
   else {
      ifileName = argv[1];
      std::cout << "Use input file: " << ifileName << std::endl;
   }
   inf.open(ifileName);
   if (!inf) {
      std::cout << "File Error: " << ifileName << std::endl;
      return 0;
   }
   unsigned particle_num = 0;
   double ttime0, dt, elasticmod, poissonp, rho, xlen, ylen, zlen;
   inf >> particle_num >> ttime0 >> dt >> elasticmod >>
      poissonp >> rho >> xlen >> ylen >> zlen;

   if ( 3 > maxdim) {
      std::cout << "maxdim too small" << std::endl;
      std::cout << "Execute terminate!" << std::endl;
      return 0;
   }

   if (10000000 < timestep * stepnum) {
      std::string str;
      std::cout << "timestep * stepnum too big, it may excute too long, contine? Y/N: " << std::endl;
      std::cin>> str;
      for (auto &c : str) {
         c = tolower(c);
      }
      if (str[0] != 'y') {
         std::cout << "Execute terminate!" << std::endl;
         return 0;
      }
   }

   if (maxdim >= 300) {
      std::string str;
      std::cout << "You need at least 1.7G memory, contine? Y/N: " << std::endl;
      std::cin>> str;
      for (auto &c : str) {
         c = tolower(c);
      }
      if (str[0] != 'y') {
         std::cout << "Execute terminate!" << std::endl;
         return 0;
      }

      if (maxdim >= 500) {
         std::string str;
         std::cout << "You need at least 6.5G memory, contine? Y/N: " << std::endl;
         std::cin>> str;
         for (auto &c : str) {
            c = tolower(c);
         }
         if (str[0] != 'y') {
            std::cout << "Execute terminate!" << std::endl;
            return 0;
         }
      }

   }

   std::string ofs_result = ifileName + ".log";
   std::ofstream ofresult(ofs_result);

   std::cout << " Particle Num: " << particle_num << std::endl;
   std::cout << "    Time step: " << timestep << std::endl;
   std::cout << "      Max dim: -" << maxdim << " ~ +" << maxdim << std::endl;
   std::cout << "Time step num: " << stepnum << std::endl;
   std::cout << std::endl << "************Start*************" << std::endl;

   ofresult << " Particle Num: " << particle_num << std::endl;
   ofresult << "    Time step: " << timestep << std::endl;
   ofresult << "      Max dim: -" << maxdim << " ~ +" << maxdim << std::endl;
   ofresult << "Time step num: " << stepnum << std::endl;
   ofresult << std::endl << "************Start*************" << std::endl;



   std::vector<Particle> pv(particle_num);
   std::size_t readnum = 0;
   for (auto &p : pv) {
      if (!inf)
         break;
      ++readnum;
      p.asign(inf);
   }
   std::cout << "Read particle data num: " << readnum << std::endl << std::endl;
   ofresult << "Read particle data num: " << readnum << std::endl << std::endl;
   inf.close();
   particle_num = readnum;
   pv.resize(particle_num);

   int device_id = 0;
   setCudaDevice(device_id);

   double maxx, maxy, maxz;
   maxx = fabs((pv[0]).xyz.x);
   maxy = fabs((pv[0]).xyz.y);
   maxz = fabs((pv[0]).xyz.z);
   for (auto p : pv) {
      if (maxx < fabs(p.xyz.x))
         maxx = fabs(p.xyz.x);
      if (maxy < fabs(p.xyz.y))
         maxy = fabs(p.xyz.y);
      if (maxz < fabs(p.xyz.z))
         maxz = fabs(p.xyz.z);
   }

   int grid_maxx, grid_maxy, grid_maxz;
   grid_maxx = grid_maxy = grid_maxz = maxdim;

   XYZ<int> grid_limit(grid_maxx - 1, grid_maxy - 1, grid_maxz - 1);
   double scal_x, scal_y, scal_z;
   scal_x = maxx == 0.0 ? 0.0 : (double)grid_limit.x / maxx;
   scal_y = maxy == 0.0 ? 0.0 : (double)grid_limit.y / maxy;
   scal_z = maxz == 0.0 ? 0.0 : (double)grid_limit.z / maxz;

   double *xt = new double(readnum + 1);
   double *yt = new double(readnum + 1);
   double *zt = new double(readnum + 1);
   double *vx = new double(readnum);
   double *vy = new double(readnum);
   double *vz = new double(readnum);
   std::vector<double *> ppv{xt, yt, zt, vx, vy, vz};
   init(ppv, pv);
   xt[readnum] = scal_x;
   yt[readnum] = scal_y;
   zt[readnum] = scal_z;


   int *x = new int(readnum);
   int *y = new int(readnum);
   int *z = new int(readnum);
   int *dev_x;
   int *dev_y;
   int *dev_z;
   double *dev_xt;
   double *dev_yt;
   double *dev_zt;

   std::cout << xt[0] << std::endl;

   CHECK_ERROR(cudaMalloc((void**)&dev_x, readnum * sizeof(int)));

   std::cout << vz[0] << std::endl;

   CHECK_ERROR(cudaMalloc((void**)&dev_y, readnum * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**)&dev_z, readnum * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**)&dev_xt, (readnum + 1) * sizeof(double)));
   CHECK_ERROR(cudaMalloc((void**)&dev_yt, (readnum + 1) * sizeof(double)));
   CHECK_ERROR(cudaMalloc((void**)&dev_zt, (readnum + 1) * sizeof(double)));
   CHECK_ERROR(cudaMemcpy(dev_xt, xt, (readnum + 1) * sizeof(double), cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(dev_yt, yt, (readnum + 1) * sizeof(double), cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(dev_zt, zt, (readnum + 1) * sizeof(double), cudaMemcpyHostToDevice));
   int threads = threadPerBlock;
   int blocks = blockPerGrid(readnum, threads);
   cudaScale<<<blocks, threads>>>(dev_xt, dev_yt, dev_zt, dev_x, dev_y, dev_z, readnum, maxdim);
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

   for (int i = 0; i < maxdim; ++i) {
      for (int j = 0; j < maxdim; ++j) {
         delete[] grid[i][j];
      }
      delete[] grid[i];
   }
   delete[] grid;

   std::cout << std::endl;
   return 0;
}

void init(std::vector<double *> &ppv, const std::vector<Particle> &pv)
{
   int readnum = pv.size();
   for (int i = 0; i < readnum; ++i) {
      *(ppv[0] + i) = (pv[i]).xyz.x;
      *(ppv[1] + i) = (pv[i]).xyz.y;
      *(ppv[2] + i) = (pv[i]).xyz.z;
      *(ppv[3] + i) = (pv[i]).v.x;
      *(ppv[4] + i) = (pv[i]).v.y;
      *(ppv[5] + i) = (pv[i]).v.z;
   }
}


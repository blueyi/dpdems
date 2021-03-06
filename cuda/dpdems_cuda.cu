/*
 * dpdems_cuda.cu
 * Copyright (C) 2016  <@A0835-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/inputDatas.h"
#include "../include/config.h"
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

void print_device(const int id, std::ostream &os)
{
   cudaDeviceProp props;
   CHECK_ERROR(cudaGetDeviceProperties(&props, id));
   os << "---Property of currently device used---" << std::endl;
   os << "Device " << id << ": " << props.name << std::endl;
   os << "CUDA Capability: " << props.major << "." << props.minor
      << std::endl;
   os << "MultiProcessor count: " << props.multiProcessorCount << std::endl;
}

void setCudaDevice(int id, std::ofstream &of)
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
   print_device(id, std::cout);
   print_device(id, of);
}

void init(std::vector<double *>&, const std::vector<Particle>&);

__global__ void cudaScale_kernel(double *dev_xt, double *dev_yt, double *dev_zt, int *dev_x, int *dev_y, int *dev_z, int readnum, int maxdim)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   while (tid < readnum) {
      dev_x[tid] = (int)(dev_xt[tid] * dev_xt[readnum]) + maxdim;
      dev_y[tid] = (int)(dev_yt[tid] * dev_yt[readnum]) + maxdim;
      dev_z[tid] = (int)(dev_zt[tid] * dev_zt[readnum]) + maxdim;
      tid += blockDim.x * gridDim.x;
   }
}

void cudaScale(int*, int*, int*, double *, double *, double *, const int&, const int&, const int&, const int&);

double scalev(double &, const double &);
void swapv(double *, double *, double *, int, int, double);
bool isInGrid(const int &, const int &, const int &, const int &);
unsigned updatePosition(int *, int *, int *, double *, double *, double *, const int &, const int &, int ***, const unsigned long &);
unsigned long long collision(int *, int *, int *, double *, double *, double *, const int &, const int &, int ***, const unsigned long &, std::ostream &);

void outputCurrentCor(int *x, int *y, int *z, int parNum);
int main(int argc, char **argv)
{
   std::cout.setf(std::ios::scientific);
   std::cout.precision(19);
   std::string configFile = "config.txt";
   unsigned specifyParNum = 0;

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
      std::string t = argv[1];
      if (t.find(".") != std::string::npos) {
         ifileName = argv[1];
         std::cout << "Use input file: " << ifileName << std::endl;
      }
      else 
         specifyParNum = std::stoul(t);
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

   if (specifyParNum != 0)
      particle_num = specifyParNum;

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

   std::string ofs_result = "gpu_" + ifileName + "_" + std::to_string(particle_num) + ".log";
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
   int readnum = 0;
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
   setCudaDevice(device_id, ofresult);

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
   scal_x = (maxx == 0.0 ? 0.0 : (double)grid_limit.x / maxx);
   scal_y = (maxy == 0.0 ? 0.0 : (double)grid_limit.y / maxy);
   scal_z = (maxz == 0.0 ? 0.0 : (double)grid_limit.z / maxz);

   double *xt = new double[readnum + 1];
   double *yt = new double[readnum + 1];
   double *zt = new double[readnum + 1];
   double *vx = new double[readnum];
   double *vy = new double[readnum];
   double *vz = new double[readnum];
   std::vector<double *> ppvt{xt, yt, zt, vx, vy, vz};
   init(ppvt, pv);
   xt[readnum] = scal_x;
   yt[readnum] = scal_y;
   zt[readnum] = scal_z;

   //   std::cout << "========" << std::endl;
   //   std::cout << scal_x << " = " << scal_y << " = " << scal_z << std::endl;
   //   std::cout << maxx << " = " << maxy << " = " << maxz << std::endl;
   //   std::cout << "========" << std::endl;
   clock_t t;
   t = clock();

   int threads = threadPerBlock;
   int blocks = blockPerGrid(readnum, threads);
   int *x = new int[readnum];
   int *y = new int[readnum];
   int *z = new int[readnum];

   /*
      cudaEvent_t start, stop;
      CHECK_ERROR(cudaEventCreate(&start));
      CHECK_ERROR(cudaEventCreate(&stop));
      CHECK_ERROR(cudaEventRecord(start, 0));
      CHECK_ERROR(cudaEventSynchronize(start));
    */

   cudaScale(x, y, z, xt, yt, zt, readnum, maxdim, threads, blocks);

   /*
      CHECK_ERROR(cudaEventRecord(stop, 0));
      CHECK_ERROR(cudaEventSynchronize(stop));
      float elapsedTime;
      CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
      CHECK_ERROR(cudaEventDestroy(start));
      CHECK_ERROR(cudaEventDestroy(stop));
      std::cout << "CUDA elapsed: " << elapsedTime / 1000.0 << std::endl;
    */

   //   outputCurrentCor(x, y, z, readnum);

   t = clock() - t;
   double seconds = (double)t / CLOCKS_PER_SEC;
   std::string share_file = "share_" + ifileName + "_" + std::to_string(particle_num) + ".dat";
   std::ifstream iShare(share_file);
   if (!iShare) {
      std::cout << "File Error: " << share_file << std::endl;
      std::cout << "Check is your CPU programme has run finished ?" << std::endl;
      return 0;
   }
   double common_time = 0.0;
   double common_scale = 0.0;
   std::string str;
   if (getline(iShare, str)) {
      while (str.empty() || str == "\r") getline(iShare, str);
   }
   iShare.close();
   std::istringstream iss(str);
   iss >> common_time >> common_scale;
   double scale_seconds = seconds * common_scale;
   double total_Time = scale_seconds + common_time;
//   std::cout << total_Time << " : " << common_time << " : " << seconds << std::endl;

   int gdim = maxdim * 2;

   int ***grid;
   grid = new int **[gdim];
   for (int i = 0; i < gdim; ++i) {
      grid[i] = new int *[gdim];
      for (int j = 0; j < gdim; ++j) {
         grid[i][j] = new int[gdim];
         for (int k = 0; k < gdim; ++k)
            grid[i][j][k] = 0;
      }
   }

   for (int i = 0; i < readnum; ++i) {
      grid[*(x + i)][*(y + i)][*(z + i)] = i;
   }

   collision(x, y, z, vx, vy, vz, readnum, gdim, grid, timestep * stepnum, ofresult);

//   t = clock() - t;
//   double seconds = (double)t / CLOCKS_PER_SEC;
   //   std::cout << x[0] << " : " << y[0] << " : " << z[0] << std::endl;



   //   std::cout << "CUDA elapsed: " << elapsedTime / 1000.0 << std::endl;
   std::cout << "Total time consumed: " << total_Time << " seconds" << std::endl;
   std::cout << "Result output to file: " << ofs_result << std::endl;

   std::cout << std::endl << "************Config Info*************" << std::endl;
   std::cout << " Particle Num: " << particle_num << std::endl;
   std::cout << "    Time step: " << timestep << std::endl;
   std::cout << "      Max dim: -" << maxdim << " ~ +" << maxdim << std::endl;
   std::cout << "Time step num: " << stepnum << std::endl;
   std::cout << std::endl << "************End*************" << std::endl;

   //   ofresult << "CUDA elapsed: " << elapsedTime / 1000.0 << std::endl;
   ofresult << std::endl << "Total time consumed: " << total_Time << " seconds" << std::endl;
   ofresult << std::endl << "************Config Info*************" << std::endl;
   ofresult << " Particle Num: " << particle_num << std::endl;
   ofresult << "    Time step: " << timestep << std::endl;
   ofresult << "      Max dim: -" << maxdim << " ~ +" << maxdim << std::endl;
   ofresult << "Time step num: " << stepnum << std::endl;
   ofresult << std::endl << "************End*************" << std::endl;

   ofresult.close();

   delete [] xt;
   delete [] yt;
   delete [] zt;

   delete [] x;
   delete [] y;
   delete [] z;
   delete [] vx;
   delete [] vy;
   delete [] vz;
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

double scalev(double &num, const double &factor)
{
   return (num + factor);
}

void swapv(double *vx, double *vy, double *vz, int num1, int num2, double factor)
{
   double tvx, tvy, tvz;
   if (vx[num1] + vy[num1] + vz[num1] < 1.0) {
      tvx = scalev(vx[num1], factor);
      tvy = scalev(vy[num1], factor);
      tvz = scalev(vz[num1], factor);
   }
   else {
      tvx = vx[num1];
      tvy = vy[num1];
      tvz = vz[num1];
   }
   if (vx[num2] + vy[num2] + vz[num2] < 1.0) {
      vx[num1] = scalev(vx[num2], factor);
      vy[num1] = scalev(vy[num2], factor);
      vz[num1] = scalev(vz[num2], factor);
   }
   else {
      vx[num1] = vx[num2];
      vy[num1] = vy[num2];
      vz[num1] = vz[num2];
   }
   vx[num2] = tvx;
   vy[num2] = tvy;
   vz[num2] = tvz;
}

bool isInGrid(const int &x, const int &y, const int &z, const int &gdim)
{
   return  (x < gdim && y < gdim && z < gdim);
}

unsigned updatePosition(int *x, int *y, int *z, double *vx, double *vy, double *vz, const int &num, const int &gdim, int ***grid, const unsigned long &time)
{
   double fix_step_length = 2.0;
   double fix_speed = 0.2;
   double fix_hit_v = 0.2;
   //   if (0.0 == fabs(vx[num]) + fabs(vy[num]) + fabs(vz[num])) {
   //      vx[num] = vy[num] = vz[num] = 0.8;
   //   }
   while ((fabs(vx[num]) + fabs(vy[num]) + fabs(vz[num])) * fix_step_length < 1.0) {
      fix_step_length += 2.0;
   }
   unsigned hit_num = 0;
   unsigned long ttime = time;
   if (!isInGrid(x[num], y[num], z[num], gdim))
      runError("Particle out of bound", "update_position");
   while (ttime--) {
      int tx = x[num];
      int ty = y[num];
      int tz = z[num];
      int fx = rint(vx[num] * fix_step_length);
      int fy = rint(vy[num] * fix_step_length);
      int fz = rint(vz[num] * fix_step_length);

      tx += ((fx < 0 && abs(fx) > tx) ? 0 : fx);
      ty += ((fy < 0 && abs(fy) > ty) ? 0 : fy);
      tz += ((fz < 0 && abs(fz) > tz) ? 0 : fz);

      if (tx >= gdim || tx < 0) {
         ++hit_num;
         tx -= ((fx < 0 && abs(fx) > tx) ? 0 : fx);
         if (tx >= gdim || tx < 0) 
            tx %= (gdim - 1);
         if (vy[num] < 1.0 || vz[num] < 1.0) {
            vy[num] += vx[num] * fix_speed;
            vz[num] += vx[num] * fix_speed;
         }
         vx[num] = - vx[num];
      }
      if (ty >= gdim || ty < 0) {
         ++hit_num;
         ty -= ((fy < 0 && abs(fy) > ty) ? 0 : fy);
         if (ty >= gdim || ty < 0) 
            ty %= (gdim - 1);
         if (vx[num] < 1.0 || vz[num] < 1.0){
            vx[num] += vy[num] * fix_speed;
            vz[num] += vy[num] * fix_speed;
         } 
         vy[num] = - vy[num];
      }
      if (tz >= gdim || tz < 0) {
         ++hit_num;
         tz -= ((fz < 0 && abs(fz) > tz) ? 0 : fz);
         if (tz >= gdim || tz < 0) 
            tz %= (gdim - 1);
         if (vx[num] < 1.0 || vy[num] < 1.0) {
            vy[num] += vz[num] * fix_speed;
            vx[num] += vz[num] * fix_speed;
         }
         vz[num] = - vz[num];
      }
      if (grid[tx][ty][tz] == 0) {
         grid[tx][ty][tz] = num;
         grid[x[num]][y[num]][z[num]] = 0;
         x[num] = tx;
         y[num] = ty;
         z[num] = tz;
      }
      else {
         ++hit_num;
         int tn = grid[tx][ty][tz];
         swapv(vx, vy, vz, num, tn, fix_hit_v);
         grid[x[num]][y[num]][z[num]] = 0;
         x[num] = tx;
         y[num] = ty;
         z[num] = tz;
         grid[tx][ty][tz] = num;
         if (grid[tx][ty][tz] != 0) {
            int tx_old = tx;
            int ty_old = ty;
            int tz_old = tz;
            int fx = rint(vx[tn] * fix_step_length);
            int fy = rint(vy[tn] * fix_step_length);
            int fz = rint(vz[tn] * fix_step_length);

            tx += ((fx < 0 && abs(fx) > tx) ? 0 : fx);
            ty += ((fy < 0 && abs(fy) > ty) ? 0 : fy);
            tz += ((fz < 0 && abs(fz) > tz) ? 0 : fz);

            if (tx >= gdim || tx < 0) {
               ++hit_num;
               tx -= ((fx < 0 && abs(fx) > tx) ? 0 : fx);
               if (tx >= gdim || tx < 0) 
                  tx %= (gdim - 1);
               if (vy[tn] < 1.0 || vz[tn] < 1.0) {
                  vy[tn] += vx[tn] * fix_speed;
                  vz[tn] += vx[tn] * fix_speed;
               }
               vx[tn] = - vx[tn];
            }
            if (ty >= gdim || ty < 0) {
               ++hit_num;
               ty -= ((fy < 0 && abs(fy) > ty) ? 0 : fy);
               if (ty >= gdim || ty < 0) 
                  ty %= (gdim - 1);
               if (vx[tn] < 1.0 || vz[tn] < 1.0){
                  vx[tn] += vy[tn] * fix_speed;
                  vz[tn] += vy[tn] * fix_speed;
               } 
               vy[tn] = - vy[tn];
            }
            if (tz >= gdim || tz < 0) {
               ++hit_num;
               tz -= ((fz < 0 && abs(fz) > tz) ? 0 : fz);
               if (tz >= gdim || tz < 0) 
                  tz %= (gdim - 1);
               if (vx[tn] < 1.0 || vy[tn] < 1.0) {
                  vy[tn] += vz[tn] * fix_speed;
                  vx[tn] += vz[tn] * fix_speed;
               }
               vz[tn] = - vz[tn];
            }
            if (tx_old == tx && ty_old == ty && tz_old == tz)
               break;
            if (grid[tx][ty][tz] != 0) {
               int ttn = grid[tx][ty][tz];
               swapv(vx, vy, vz, tn, ttn, fix_hit_v);
               ++hit_num;
            }
         }
         grid[tx][ty][tz] = tn;
      }
   }
   return hit_num;
}

unsigned long long collision(int *x, int *y, int *z, double *vx, double *vy, double *vz, const int &readnum, const int &gdim, int ***grid, const unsigned long &time, std::ostream &os)
{
   unsigned long long total_hit = 0;
   for (int i = 0; i < readnum; ++i) {
      unsigned hit_times = updatePosition(x, y, z, vx, vy, vz, i, gdim, grid, time);
      total_hit += hit_times;
      std::cout << std::endl << "Particle " << i + 1 << " hit times: " << hit_times << std::endl;
      std::cout << "      Total hit times: " << total_hit << std::endl;
      std::cout << "Particle current info: " << std::endl;
      std::cout << x[i] << "\t" << y[i] << "\t" << z[i] << std::endl;
      std::cout << vx[i] << "\t" << vy[i] << "\t" << vz[i] << std::endl;

      os.setf(std::ios::scientific);
      os.precision(16);
      os << std::endl << "********************" << std::endl;
      os << "Particle " << i + 1 << " hit times: " << hit_times << std::endl;
      os << "Total hit times: " << total_hit << std::endl << std::endl;
      //os << "Particle origin info: " << std::endl;
      os << "Particle current info: " << std::endl;
      os << x[i] << "\t" << y[i] << "\t" << z[i] << std::endl;
      os << vx[i] << "\t" << vy[i] << "\t" << vz[i] << std::endl;
   }
   return total_hit;
}

void cudaScale(int *x, int *y, int *z, double *xt, double *yt, double *zt, const int &readnum, const int &maxdim, const int &threads, const int &blocks)
{
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
   CHECK_ERROR(cudaMemcpy(dev_xt, xt, (readnum + 1) * sizeof(double), cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(dev_yt, yt, (readnum + 1) * sizeof(double), cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(dev_zt, zt, (readnum + 1) * sizeof(double), cudaMemcpyHostToDevice));
   cudaScale_kernel<<<blocks, threads>>>(dev_xt, dev_yt, dev_zt, dev_x, dev_y, dev_z, readnum, maxdim);
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
}

void outputCurrentCor(int *x, int *y, int *z, int parNum)
{
   std::ofstream of("Partic_cor.txt");
   for (int i = 0; i < parNum; ++i) {
      of << "***************" << std::endl;
      of << i << ": " << std::endl;
      of << x[i] - 100 << "\t" << y[i] - 100 << "\t" << z[i] - 100 << std::endl << std::endl;
   }
   of.close();
}


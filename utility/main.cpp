/*
 * search.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/grid.h"
#include <iostream>
#include <fstream>

const int grid_minx = -100;
const int grid_miny = -100;
const int grid_minz = -100;
const int grid_maxx = 100;
const int grid_maxy = 100;
const int grid_maxz = 100;

unsigned hit(std::vector<ParticlePtr> &, Grid &, unsigned long);

int main(int argc, char **argv)
{
   // std::cout.setf(std::ios::scientific);
   // std::cout.precision(19);
   std::string ifileName = "inputdatas.txt";
   std::ifstream inf;
   if (2 > argc){
      std::cout << "Use the default input file: inputdatas.txt" << std::endl;
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
   std::vector<Particle> pv(particle_num);
   for (auto &p : pv) {
      p.asign(inf);
   }
   inf.close();
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
   XYZ<int> grid_limit(grid_maxx - 1, grid_maxy - 1, grid_maxz - 1);
   double scal_x, scal_y, scal_z;
   scal_x = maxx == 0.0 ? 0.0 : (double)grid_limit.x / maxx;
   scal_y = maxy == 0.0 ? 0.0 : (double)grid_limit.y / maxy;
   scal_z = maxz == 0.0 ? 0.0 : (double)grid_limit.z / maxz;
   XYZ<double> scal_factor(scal_x, scal_y, scal_z);

   std::vector<ParticlePtr> ppv(particle_num);
   //    for (auto pp : ppv) {
   //       std::cout << "=" << pp.no() << "=" << std::endl;
   //    }
   auto ppb = ppv.begin();

   for (auto pb = pv.begin(); pb != pv.end(); ++pb, ++ppb) {
      ppb->asign(*pb, scal_factor);
   }
//   for (auto pp : ppv) {
//      std::cout << "*" << pp.no() << "*" << std::endl;
//      pp.print(std::cout);
//   }

   std::size_t gdimx = axis_conv(grid_maxx, abs(grid_minx));
   std::size_t gdimy = axis_conv(grid_maxy, abs(grid_miny));
   std::size_t gdimz = axis_conv(grid_maxz, abs(grid_minz));

   XYZ<int> offset(grid_maxx, grid_maxy, grid_maxz);
   Grid grid(gdimx, gdimy, gdimz, offset);
   grid.fill(ppv);

   unsigned hit_times = hit(ppv, grid, 1234);
   std::cout << "hit_times: " << hit_times << std::endl;

//   for (auto pp : ppv) {
//      std::cout << "*" << pp.no() << "*" << std::endl;
//      pp.print(std::cout);
//   }

   std::cout << "Unnull: " << grid.unNullPtrNum() << std::endl;


   //std::cout << maxx << " " << maxy << " " << maxz << std::endl;
   //std::cout << p0.x << " " << p0.y << " " << p0.z << std::endl;
   //std::cout << (pv[0]).xyz.x << " " << (pv[0]).xyz.y << " " << (pv[0]).xyz.z;
   std::cout << std::endl;
   return 0;
}

unsigned hit(std::vector<ParticlePtr> &ppv, Grid &grid, unsigned long time)
{
   unsigned hit_times = 0;
   for (auto &pp : ppv) {
      hit_times += pp.move(grid, time);
   }
   return hit_times;
}

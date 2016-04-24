/*
 * search.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/grid.h"
#include "../include/config.h"
#include <iostream>
#include <fstream>
#include <cctype>
#include <ctime>

//const int grid_minx = -50;
//const int grid_miny = -50;
//const int grid_minz = -50;
//const int grid_maxx = 50;
//const int grid_maxy = 50;
//const int grid_maxz = 50;

unsigned long long hit(std::vector<ParticlePtr> &ppv, Grid &grid, unsigned long time, std::ostream &os, const std::vector<Particle> &pv);

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



   clock_t t;
   t = clock();

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

   int grid_minx, grid_miny, grid_minz, grid_maxx, grid_maxy, grid_maxz;
   grid_minx = grid_miny = grid_minz = grid_maxx = grid_maxy = grid_maxz = maxdim;

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
   (ppv.begin() + 1)->modify_cor(1, 1, 4);

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


   hit(ppv, grid, timestep * stepnum, ofresult, pv);
   t = clock() - t;
   double seconds = (double)t / CLOCKS_PER_SEC;

   std::cout << std::endl << "Total time consumed: " << seconds << " seconds" << std::endl;
   std::cout << "Result output to file: " << ofs_result << std::endl;

   std::cout << std::endl << "************Config Info*************" << std::endl;
   std::cout << " Particle Num: " << particle_num << std::endl;
   std::cout << "    Time step: " << timestep << std::endl;
   std::cout << "      Max dim: -" << maxdim << " ~ +" << maxdim << std::endl;
   std::cout << "Time step num: " << stepnum << std::endl;
   std::cout << std::endl << "************End*************" << std::endl;

   ofresult << std::endl << "Total time consumed: " << seconds << " seconds" << std::endl;
   ofresult << std::endl << "************Config Info*************" << std::endl;
   ofresult << " Particle Num: " << particle_num << std::endl;
   ofresult << "    Time step: " << timestep << std::endl;
   ofresult << "      Max dim: -" << maxdim << " ~ +" << maxdim << std::endl;
   ofresult << "Time step num: " << stepnum << std::endl;
   ofresult << std::endl << "************End*************" << std::endl;

   ofresult.close();

   //   for (auto pp : ppv) {
   //      std::cout << "*" << pp.no() << "*" << std::endl;
   //      pp.print(std::cout);
   //   }

   //   std::cout << "Unnull: " << grid.unNullPtrNum() << std::endl;


   //std::cout << maxx << " " << maxy << " " << maxz << std::endl;
   //std::cout << p0.x << " " << p0.y << " " << p0.z << std::endl;
   //std::cout << (pv[0]).xyz.x << " " << (pv[0]).xyz.y << " " << (pv[0]).xyz.z;
   std::cout << std::endl;
   return 0;
}

unsigned long long hit(std::vector<ParticlePtr> &ppv, Grid &grid, unsigned long time, std::ostream &os, const std::vector<Particle> &pv)
{
   unsigned long long total_hit = 0;
   for (auto &pp : ppv) {
      unsigned hit_times = pp.move(grid, time);
      total_hit += hit_times;
      std::cout << std::endl << "Particle " << pp.no() << " hit times: " << hit_times << std::endl;
      std::cout << "      Total hit times: " << total_hit << std::endl;
      std::cout << "Particle current info: " << std::endl;
      pp.print(std::cout);

      os.setf(std::ios::scientific);
      os.precision(19);
      os << std::endl << "********************" << std::endl;
      os << "Particle " << pp.no() << " hit times: " << hit_times << std::endl;
      os << "Particle origin info: " << std::endl;
      (pv[pp.no() - 1]).print(os);
      os << "Particle current info: " << std::endl;
      pp.print(os);
      os << "Total hit times: " << total_hit << std::endl << std::endl;
   }
   return total_hit;
}

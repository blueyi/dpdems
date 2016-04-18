/*
 * search.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/inputDatas.h"
#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
    std::cout.setf(std::ios::scientific);
    std::cout.precision(19);
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
    std::vector<ParticlePtr> ppv(particle_num);
    for (auto &p : pv) {
        p.asign(inf);
    }
    inf.close();
    auto ppb = ppv.begin();
    for (auto pb = pv.begin(); pb != pv.end(); ++pb, ++ppb) {
        *ppb = *pb;
    }
    /*
    for (auto pp : ppv) {
        pp.print(std::cout);
    }
    */
    return 0;
}



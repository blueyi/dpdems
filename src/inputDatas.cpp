/*
 * inputDatas.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/inputDatas.h"
#include <sstream>


Particle::Particle(std::ifstream &infile)
{
    xyz.asign(infile);
    v.asign(infile);
    q.asign(infile);
    w.asign(infile);
    infile >> mass;
    ip.asign(infile);
    r.asign(infile);
    n.asign(infile);
    rxb = std::make_shared<std::vector<XYZ<double>>>();
    rxbp = std::make_shared<std::vector<XYZ<double>>>();
    for (std::size_t i = 0; i < n.x; ++i) {
        XYZ<double> temp;
        temp.asign(infile);
        rxb->push_back(temp);
        temp.asign(infile);
        rxbp->push_back(temp);
    }
    edge = std::make_shared<std::vector<XY<int>>>();
    for (std::size_t i = 0; i < n.y; ++i) {
        XY<int> temp;
        temp.asign(infile);
        edge->push_back(temp);
    }
    surf = std::make_shared<std::vector<std::vector<int>>>();
    for (std::size_t i = 0; i < n.z; ++i) {
        std::string line;
        std::vector<int> tempvec;
        if (getline(infile, line)) {
            int tmpf, tmpi;
            std::istringstream ist(line);
            ist >> tmpf;
            while (tmpf-- > 0) {
                ist >> tmpi;
                tempvec.push_back(tmpi);
            }
        }
        surf->push_back(tempvec);
    }
}

void Particle::print(std::ostream &os) const
{
    xyz.print(os) << std::endl;
    v.print(os) << std::endl;
    q.print(os) << std::endl;
    w.print(os) << std::endl;
    os << mass << "\t";
    ip.print(os) << std::endl;
    r.print(os) << std::endl;
    n.print(os) << std::endl;
    for (auto r : *rxb) 
        r.print(os)  << std::endl;
    for (auto rp : *rxbp) 
        rp.print(os)  << std::endl;
    for (auto e : *edge) 
        e.print(os)  << std::endl;
    for (auto s : *surf) {
        os << s.size() << "\t";
        for (auto i : s)
            os << i << "\t";
        os << std::endl;
    }
}


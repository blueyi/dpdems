/*
 * inputDatas.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/inputDatas.h"
#include <sstream>


Particle::Particle(std::istream &infile)
{
    xyz.asign(infile);
    v.asign(infile);
    q.asign(infile);
    w.asign(infile);
    infile >> mass;
    ip.asign(infile);
    r.asign(infile);
    n.asign(infile);
    rb = std::make_shared<std::vector<XYZ<double>>>();
    rbp = std::make_shared<std::vector<XYZ<double>>>();
    for (std::size_t i = 0; i < n.x; ++i) {
        XYZ<double> temp;
        temp.asign(infile);
        rb->push_back(temp);
        temp.asign(infile);
        rbp->push_back(temp);
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
            while (line.empty() || line == "\r")
                getline(infile, line);
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

std::istream& Particle::asign(std::istream &infile)
{
    xyz.asign(infile);
    v.asign(infile);
    q.asign(infile);
    w.asign(infile);
    infile >> mass;
    ip.asign(infile);
    r.asign(infile);
    n.asign(infile);
    rb = std::make_shared<std::vector<XYZ<double>>>();
    rbp = std::make_shared<std::vector<XYZ<double>>>();
    for (std::size_t i = 0; i < n.x; ++i) {
        XYZ<double> temp;
        temp.asign(infile);
        rb->push_back(temp);
        temp.asign(infile);
        rbp->push_back(temp);
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
            while (line.empty() || line == "\r")
                getline(infile, line);
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
    return infile;
}

std::ostream& Particle::print(std::ostream &os) const
{
    xyz.print(os) << std::endl;
    v.print(os) << std::endl;
    q.print(os) << std::endl;
    w.print(os) << std::endl;
    os << mass << "\t";
    ip.print(os) << std::endl;
    r.print(os) << std::endl;
    n.print(os) << std::endl;
    for (auto r = (*rb).begin(), rp = (*rbp).begin(); r != (*rb).end(); ++r, ++rp) {
        (*r).print(os)  << std::endl;
        (*rp).print(os)  << std::endl;
    }
    for (auto e : *edge) 
        e.print(os)  << std::endl;
    for (auto s : *surf) {
        os << s.size() << "\t";
        for (auto i : s)
            os << i << "\t";
        os << std::endl;
    }
    return os;
}

ParticlePtr::ParticlePtr(const Particle &part, const XYZ<double> &scal)
{
    xyz = scale(part.xyz, scal);
    v = part.v;
}

ParticlePtr& ParticlePtr::asign(const Particle &par, const XYZ<double> &scal)
{
    this->xyz = scale(par.xyz, scal);
    this->v = par.v;
    return *this;
}

ParticlePtr& ParticlePtr::operator=(const ParticlePtr pptr)
{
    this->xyz = pptr.xyz;
    this->v = pptr.v;
    return *this;
}

std::ostream& ParticlePtr::print(std::ostream &os) const 
{
    xyz.print(os) << std::endl;
    v.print(os) << std::endl;
    return os;
}

void ParticlePtr::hit_v(ParticlePtr *pb)
{
    auto t = this->v;
    this->v = pb->v;
    if (this->v.sum() < 1.0)
        this->v.scale(0.2);
    pb->v = t;
    if (pb->v.sum() < 1.0)
        pb->v.scale(0.2);
}

unsigned ParticlePtr::move(Grid &grid, unsigned long time)
{
    if (v.iszero())
        return 0;
    return grid.update_position(*this, time);
}

void ParticlePtr::modify_cor(int x, int y, int z)
{
    xyz.asign(x, y, z);
}
void ParticlePtr::modify_v(double x, double y, double z)
{
    v.asign(x, y, z);
}

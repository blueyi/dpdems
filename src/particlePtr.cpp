/*
 * particlePtr.cpp
 * Copyright (C) 2016  <@A0835-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/particlePtr.h"


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

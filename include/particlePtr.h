/*
 * particlePtr.h
 * Copyright (C) 2016  <@A0835-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/Grid.h"
#ifndef PARTICLEPTR_H
#define PARTICLEPTR_H
#include "../include/inputDatas.h"

class Grid;

static unsigned static_no = 0;
class ParticlePtr : public BParticle {
public:
    friend class Grid;
    ParticlePtr() {num = ++static_no;} 
    ParticlePtr(const Particle &, const XYZ<double> &);
    unsigned no() {return num;}
    XYZ<int>  cor() {return xyz;}
    void modify_cor(int x, int y, int z);
    void modify_v(double, double, double);
    ParticlePtr& asign(const Particle &, const XYZ<double> &);
    ParticlePtr& operator=(const ParticlePtr);
    void hit_v(ParticlePtr *);
    unsigned move(Grid &, unsigned long);
    std::ostream& print(std::ostream &) const override;

protected:
    XYZ<int> xyz{0, 0, 0};
    XYZ<double> v{0, 0, 0};
    unsigned num{0};
};



#endif /* !PARTICLEPTR_H */

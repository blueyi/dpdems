/*
 * inputDatas.h
 * Copyright (C) 2016  <@BLUEYI-PC>
 * 
 * Distributed under terms of the MIT license.
 */

#include "../include/grid.h"
#ifndef INPUTDATAS_H
#define INPUTDATAS_H
#include "common.h"
#include <memory>
#include <vector>

class BParticle {
public:
    BParticle() = default;
    virtual std::ostream& print(std::ostream &) const = 0;
    virtual ~BParticle() = default;
};

//暂时只做数据保存
class Particle : public BParticle {
public:
    Particle() = default;
    Particle(std::istream &);
    std::istream& asign(std::istream &);
    std::ostream& print(std::ostream &) const override;
    ~Particle() = default;

    XYZ<double> xyz{0.0, 0.0, 0.0};
    XYZ<double> v{0.0, 0.0, 0.0};
    Felement<double> q{0.0, 0.0, 0.0, 0.0};
    XYZ<double> w{0.0, 0.0, 0.0};
    double mass{0.0};
    XYZ<double> ip{0.0, 0.0, 0.0};
    XY<double> r{0.0, 0.0};
    XYZ<std::size_t> n{0, 0, 0};
    std::shared_ptr<std::vector<XYZ<double>>> rb;
    std::shared_ptr<std::vector<XYZ<double>>> rbp;
    std::shared_ptr<std::vector<XY<int>>> edge;
    std::shared_ptr<std::vector<std::vector<int>>> surf;
};

class Grid;

static unsigned static_no = 0;
class ParticlePtr : public BParticle {
public:
    friend class Grid;
    ParticlePtr() {num = ++static_no;} 
    ParticlePtr(const Particle &, const XYZ<double> &);
    unsigned no() {return num;}
    XYZ<int>  cor() {return xyz;}
    ParticlePtr& asign(const Particle &, const XYZ<double> &);
    ParticlePtr& operator=(const ParticlePtr);
    void hit_v(ParticlePtr &);
    bool move(Grid &, unsigned long);
    std::ostream& print(std::ostream &) const override;

protected:
    XYZ<int> xyz{0, 0, 0};
    XYZ<double> v{0, 0, 0};
    unsigned num{0};
};

#endif /* !INPUTDATAS_H */

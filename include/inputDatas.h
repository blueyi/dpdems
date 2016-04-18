/*
 * inputDatas.h
 * Copyright (C) 2016  <@BLUEYI-PC>
 * 
 * Distributed under terms of the MIT license.
 */

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

class ParticlePtr : public BParticle {
public:
    ParticlePtr() = default;
    ParticlePtr(const Particle &);
    ParticlePtr& operator=(const Particle);
    ParticlePtr& operator=(const ParticlePtr);
    void hit(ParticlePtr &);
    std::ostream& print(std::ostream &) const override;

protected:
    XYZ<double> xyz{0.0, 0.0, 0.0};
    XYZ<double> v{0.0, 0.0, 0.0};
};

#endif /* !INPUTDATAS_H */

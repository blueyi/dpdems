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

//暂时只做数据保存
class Particle {
public:
    Particle(std::ifstream &);

    XYZ<double> xyz{0.0, 0.0, 0.0};
    XYZ<double> v{0.0, 0.0, 0.0};
    Felement<double> q{0.0, 0.0, 0.0, 0.0};
    XYZ<double> w{0.0, 0.0, 0.0};
    double mass{0.0};
    XYZ<double> ip{0.0, 0.0, 0.0};
    XY<double> r{0.0, 0.0};
    XYZ<std::size_t> n{0, 0, 0};
    std::shared_ptr<std::vector<XYZ<double>>> rxb;
    std::shared_ptr<std::vector<XYZ<double>>> rxbp;
    std::shared_ptr<std::vector<XY<int>>> edge;
    std::shared_ptr<std::vector<std::vector<int>>> surf;
};

#endif /* !INPUTDATAS_H */

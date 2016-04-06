/*
 * inputDatas.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "inputDatas.h"
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

}




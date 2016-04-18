/*
 * testinputDatas.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/inputDatas.h"
#include <iostream>
#include <fstream>

int main(void)
{
    std::cout.setf(std::ios::scientific);
    std::cout.precision(19);
    std::ifstream inf("inputdatas.txt");
    if (!inf) {
        std::cout << "File Error" << std::endl;
        return 0;
    }
    Particle part(inf);
    part.print(std::cout);
    inf.close();
    return 0;
}




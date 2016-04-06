/*
 * testCommon.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/common.h"
#include <iostream>

int main()
{
    XYZ<double> vec(23, 22, 0.1);
    std::cout << vec.x << " " << vec.y << " " << vec.z << std::endl;
    return 0;
}




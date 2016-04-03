/*
 * testConf.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include "config.h"

int main()
{
    if (ini_conf("../misc/inp_new.txt"))
        std::cout << "Success" << std::endl;
    else 
        std::cout << "ini_conf error!" << std::endl;

    return 0;
}



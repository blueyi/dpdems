/*
 * common.h
 * Copyright (C) 2016  <@BLUEYI-PC>
 * blueyi
 * Contain most of constant value and some inline function
 * Distributed under terms of the MIT license.
 */

#ifndef COMMON_H
#define COMMON_H
#include <exception>
#include <string>


/******Const value*****/
const double pi(3.14159265359);
const double waterrhow(1025.0);
const int npair(50);
const int npair_struct_bound(500);
const int nGFpair(20);
const double maxsn(-1e9);
const double maxst(-1e9);



/******Function*****/

void runError(const std::string &err, const std::string &msg)
{
    throw std::runtime_error(err + " error: " + msg);
}


#endif /* !COMMON_H */
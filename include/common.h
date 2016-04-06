/*
 * common.h
 * Copyright (C) 2016  <@BLUEYI-PC>
 * blueyi
 * Contain most of constant value and some inline function
 * Distributed under terms of the MIT license.
 */

#ifndef COMMON_H
#define COMMON_H
#include <stdexcept>
#include <string>
#include <fstream>


/******Const value*****/
const double pi(3.14159265359);
const double waterrhow(1025.0);
const int npair(50);
const int npair_struct_bound(500);
const int nGFpair(20);
const double maxsn(-1e9);
const double maxst(-1e9);


/******Datas common type*****/
template <typename T> class XY {
public:
    XY() = default;
    XY(T i, T j) : x(i), y(j) {}
    std::ifstream& asign(std::ifstream &inf) { inf >> x >> y; return inf; }

    T x;
    T y;
};

template <typename T> class XYZ : public XY<T> {
public:
    XYZ() = default;
    XYZ(T i, T j, T k) : XY<T>(i, j), z(k) {}
    std::ifstream& asign(std::ifstream &inf) { inf >> this->x >> this->y >> z; return inf; }

    T z;
};

template<typename T> class Felement : public XYZ<T> {
public:
    Felement() = default;
    Felement(T i, T j, T k, T l) : XYZ<T>(i, j, k), w(l) {}
    std::ifstream& asign(std::ifstream &inf) { inf >> XYZ<T>::x >> XYZ<T>::y >> XYZ<T>::z >> w; return inf; }

    T w;
};

/******Function*****/

void runError(const std::string &err, const std::string &msg)
{
    throw std::runtime_error(err + " error: " + msg);
}


#endif /* !COMMON_H */

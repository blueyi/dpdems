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
#include <istream>
#include <ostream>


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
    XY<T>& operator=(const XY<T>);
    virtual std::istream& asign(std::istream &inf) { inf >> x >> y; return inf; }
    virtual std::ostream& print(std::ostream &os) const {os << x << "\t" << y; return os;}
    virtual ~XY() = default;

    T x;
    T y;
};

template <typename T> class XYZ : public XY<T> {
public:
    XYZ() = default;
    XYZ(T i, T j, T k) : XY<T>(i, j), z(k) {}
    XYZ<T>& operator=(const XYZ<T>);
    std::istream& asign(std::istream &inf) override { inf >> this->x >> this->y >> z; return inf; }
    std::ostream& print(std::ostream &os) const override {os << this->x << "\t" << this->y << "\t" << z; return os;}
    virtual ~XYZ() = default;

    T z;
};

template<typename T> class Felement : public XYZ<T> {
public:
    Felement() = default;
    Felement(T i, T j, T k, T l) : XYZ<T>(i, j, k), w(l) {}
    std::istream& asign(std::istream &inf) override { inf >> XYZ<T>::x >> XYZ<T>::y >> XYZ<T>::z >> w; return inf; }
    std::ostream& print(std::ostream &os) const override {os << this->x << "\t" << this->y << "\t" << this->z << "\t" << w; return os;}
    virtual ~Felement() = default;

    T w;
};

/******Function*****/


template <typename T> XY<T>& XY<T>::operator=(const XY<T> r)
{
    this->x = r.x;
    this->y = r.y;
    return *this;
}

template <typename T> XYZ<T>& XYZ<T>::operator=(const XYZ<T> r)
{
    this->x = r.x;
    this->y = r.y;
    this->z = r.z;
    return *this;
}

void runError(const std::string &err, const std::string &msg);

#endif /* !COMMON_H */

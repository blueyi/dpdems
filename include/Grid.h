/*
 * Grid.h
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRID_H
#define GRID_H
#include "inputDatas.h"

class Grid {
public:
    friend class ParticlePtr;
    Grid() = default;
    Grid(std::size_t x, std::size_t y, std::size_t z, XYZ<int> o) : gdimx(x), gdimy(y), gdimz(z), offset(o) {
        gridptr = std::make_shared<std::vector<std::vector<std::vector<ParticlePtr*>>>>(gdimx, std::vector<std::vector<ParticlePtr*>>(gdimy, std::vector<ParticlePtr*>(gdimz, nullptr)));
    };
    void fill(std::vector<ParticlePtr> &);
    unsigned checkPPNo(const XYZ<int> &);
    bool isInGrid(const XYZ<int> &);
    bool isInGrid(const XYZ<unsigned> &);
    unsigned unNullPtrNum();
    unsigned update_position(ParticlePtr &pp, unsigned long time);

    std::shared_ptr<std::vector<std::vector<std::vector<ParticlePtr*>>>> gridptr;

private:
    std::size_t gdimx{0};
    std::size_t gdimy{0};
    std::size_t gdimz{0};
    XYZ<int> offset;
};



#endif /* !GRID_H */

/*
 * grid.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/grid.h"

void Grid::fill( std::vector<ParticlePtr> &ppv)
{
    for (auto &pp : ppv) {
        XYZ<unsigned> t = axis_conv(pp.xyz, offset, false);
        (*gridptr)[t.x][t.y][t.z] = &pp;
    }
}

unsigned Grid::checkPPNo(const XYZ<int> &cor)
{
    XYZ<unsigned> t = axis_conv(cor, offset, false);
    if ((*gridptr)[t.x][t.y][t.z] != nullptr)
        return ((*gridptr)[t.x][t.y][t.z])->no();
    else
        return 0;
}

bool Grid::isInGrid(const XYZ<int> &cor)
{
    XYZ<unsigned> tg = axis_conv(cor, offset, false);
    if (tg.x > gdimx || tg.y > gdimy || tg.z > gdimz)
        return false;
    return true;
}

bool Grid::isInGrid(const XYZ<unsigned> &cor)
{
    if (cor.x > gdimx || cor.y > gdimy || cor.z > gdimz)
        return false;
    return true;
}

unsigned Grid::update_position(ParticlePtr &pp, unsigned long time)
{
    unsigned hit_num = 0;
    unsigned long ttime = time;
    XYZ<int> start = pp.cor();
    if (!isInGrid(start))
        runError("Particle out of bound", "update_position");
    XYZ<unsigned> gs = axis_conv(start, offset, false);
    for (unsigned x = gs.x, y = gs.y, z = gs.z;ttime-- > 0;) {
        unsigned tx = x;
        unsigned ty = y;
        unsigned tz = z;
        x += pp.v.x;
        y += pp.v.y;
        z += pp.v.z;
        //简化考虑越界问题，速度为矢量
        while (x >= gdimx || x < 0) {
            ++hit_num;
            x -= pp.v.x;
            pp.v.x = - pp.v.x;
        }
        while (y >= gdimy || y < 0) {
            ++hit_num;
            y -= pp.v.y;
            pp.v.y = - pp.v.y;
        }
        while (z >= gdimz || z < 0) {
            ++hit_num;
            z -= pp.v.z;
            pp.v.z = - pp.v.z;
        }

        if ((*gridptr)[x][y][z] == nullptr) {
            pp.xyz.asign(axis_conv(x, -offset.x), axis_conv(y, -offset.y), axis_conv(z, -offset.z));
            (*gridptr)[x][y][z] = &pp;
            (*gridptr)[tx][ty][tz] = nullptr;
        }
        else {
            hit_num++;
            //两个颗粒应该交换速度
            auto tpptr = (*gridptr)[x][y][z];
            pp.hit_v(tpptr);
            pp.xyz.asign(axis_conv(x, -offset.x), axis_conv(y, -offset.y), axis_conv(z, -offset.z));
            auto tp = axis_conv(tpptr->cor(), offset, false);
            (*gridptr)[x][y][z] = &pp;
            (*gridptr)[tx][ty][tz] = nullptr;
            while ((*gridptr)[tp.x][tp.y][tp.z] != nullptr) {
                auto ttptr = (*gridptr)[tp.x][tp.y][tp.z];
                tp.x += tpptr->v.x;
                tp.y += tpptr->v.y;
                tp.z += tpptr->v.z;
                while (tp.x >= gdimx || tp.x < 0) {
                    tp.x -= pp.v.x;
                    pp.v.x = - pp.v.x;
                    hit_num++;
                }
                while (tp.y >= gdimy || tp.y < 0) {
                    tp.y -= pp.v.y;
                    pp.v.y = - pp.v.y;
                    hit_num++;
                }
                while (tp.z >= gdimz || tp.z < 0) {
                    tp.z -= pp.v.z;
                    pp.v.z = - pp.v.z;
                    hit_num++;
                }
                if ((*gridptr)[tp.x][tp.y][tp.z] != nullptr) {
                    auto ttptrr = (*gridptr)[tp.x][tp.y][tp.z];
                    ttptrr->hit_v(ttptr);
                    hit_num++;
                }
            }

            (*gridptr)[tp.x][tp.y][tp.z] = tpptr;
        }
    }
    return hit_num;
}

unsigned Grid::unNullPtrNum()
{
    unsigned sum = 0;
    for (std::size_t x = 0; x < gdimx; ++x) {
        for (std::size_t y = 0; y < gdimy; ++y) {
            for (std::size_t z = 0; z < gdimz; ++z) {
                if ((*gridptr)[x][y][z] != nullptr)
                    ++sum;
            }
        }
    }
    return sum;
}

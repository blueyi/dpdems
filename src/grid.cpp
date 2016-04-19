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
        auto tpp = (*gridptr)[t.x][t.y][t.z];
        if (tpp == nullptr)
            tpp = nullptr;
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

void Grid::update_position(ParticlePtr &pp, const XYZ<int> &end)
{
    XYZ<int> start = pp.cor();
    if (!isInGrid(start))
        runError("Particle out of bound", "update_position");
    XYZ<unsigned> gs = axis_conv(start, offset, false);
    XYZ<unsigned> ge = axis_conv(end, offset, false);
    for (unsigned x = gs.x, y = gs.y, z = gs.z; x < ge.x;) {
        unsigned tx = x;
        unsigned ty = y;
        unsigned tz = z;
        x += pp.v.x;
        y += pp.v.y;
        z += pp.v.z;
        //简化考虑越界问题，速度为矢量
        if (x > gdimx || x < 0) {
            x -= pp.v.x;
            pp.v.x = - pp.v.x;
        }
        if (y > gdimy || y < 0) {
            y -= pp.v.y;
            pp.v.y = - pp.v.y;
        }
        if (z > gdimz || z < 0) {
            z -= pp.v.z;
            pp.v.z = - pp.v.z;
        }

        if ((*gridptr)[x][y][z] == nullptr) {
            pp.xyz.asign(axis_conv(x, -offset.x), axis_conv(y, -offset.y), axis_conv(z, -offset.z));
            (*gridptr)[x][y][z] = &pp;
            (*gridptr)[tx][ty][tz] = nullptr;
        }
        else {
            //两个颗粒应该交换速度
            pp.xyz.asign(axis_conv(x, -offset.x), axis_conv(y, -offset.y), axis_conv(z, -offset.z));
            auto tp = axis_conv(((*gridptr)[x][y][z])->cor(), offset, false);
            auto tpptr = (*gridptr)[x][y][z];
            (*gridptr)[x][y][z] = &pp;
            (*gridptr)[tx][ty][tz] = nullptr;
            while ((*gridptr)[tp.x][tp.y][tp.z] != nullptr) {
                tp.x += pp.v.x;
                tp.y += pp.v.y;
                tp.z += pp.v.z;
            }
            (*gridptr)[tp.x][tp.y][tp.z] = tpptr;
        }
    }

}


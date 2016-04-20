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
    double fix_step_length = 2.0;
    double fix_speed = 0.2;
    while ((fabs(pp.v.x) + fabs(pp.v.y) + fabs(pp.v.z)) * fix_step_length < 1.0)
        fix_step_length += 2.0;

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
        int fx = rint(pp.v.x * fix_step_length);
        int fy = rint(pp.v.y * fix_step_length);
        int fz = rint(pp.v.z * fix_step_length);

        x += ((fx < 0 && abs(fx) > x) ? 0 : fx);
        y += ((fy < 0 && abs(fy) > y) ? 0 : fy);
        z += ((fz < 0 && abs(fz) > z) ? 0 : fz);

        //简化考虑越界问题，速度为矢量
        if (x >= gdimx || x < 0) {
            ++hit_num;
            x -= ((fx < 0 && abs(fx) > x) ? 0 : fx);
            if (x >= gdimx || x < 0) 
                x %= (gdimx-1);
            if (pp.v.y < 1.0 || pp.v.z < 1.0) {
                pp.v.y += pp.v.x * fix_speed;
                pp.v.z += pp.v.x * fix_speed;
            }
            pp.v.x = - pp.v.x;
        }
        if (y >= gdimy || y < 0) {
            ++hit_num;
            y -= ((fy < 0 && abs(fy) > y) ? 0 : fy);
            if (y >= gdimy || y < 0) 
                y %= (gdimy - 1);
            if (pp.v.x < 1.0 || pp.v.z < 1.0){
                pp.v.x += pp.v.y * fix_speed;
                pp.v.z += pp.v.y * fix_speed;
            } 
            pp.v.y = - pp.v.y;
        }
        if (z >= gdimz || z < 0) {
            ++hit_num;
            z -= ((fz < 0 && abs(fz) > z) ? 0 : fz);
            if (z >= gdimz || z < 0) 
                z %= (gdimz - 1);
            if (pp.v.x < 1.0 || pp.v.y < 1.0) {
                pp.v.y += pp.v.z * fix_speed;
                pp.v.x += pp.v.z * fix_speed;
            }
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
            //不考虑连续碰撞
            while ((*gridptr)[tp.x][tp.y][tp.z] != nullptr) {
                auto tp_old = tp;
                auto ttptr = (*gridptr)[tp.x][tp.y][tp.z];
                int fx = rint(tpptr->v.x * fix_step_length);
                int fy = rint(tpptr->v.y * fix_step_length);
                int fz = rint(tpptr->v.z * fix_step_length);
                tp.x += ((fx < 0 && abs(fx) > tp.x) ? 0 : fx);
                tp.y += ((fy < 0 && abs(fy) > tp.y) ? 0 : fy);
                tp.z += ((fz < 0 && abs(fz) > tp.z) ? 0 : fz);
                if (tp.x >= gdimx || tp.x < 0) {
                    tp.x -= ((fx < 0 && abs(fx) > x) ? 0 : fx);
                    if (tp.x >= gdimx || tp.x < 0) 
                        tp.x %= (gdimx-1);
                    tpptr->v.y += tpptr->v.y * fix_speed;
                    tpptr->v.z += tpptr->v.z * fix_speed;
                    tpptr->v.x = -tpptr->v.x;
                    hit_num++;
                }
                if (tp.y >= gdimy || tp.y < 0) {
                    tp.y -= ((fy < 0 && abs(fy) > y) ? 0 : fy);
                    if (tp.y >= gdimy || tp.y < 0) 
                        tp.y %= (gdimy - 1);
                    tpptr->v.x += tpptr->v.x * fix_speed;
                    tpptr->v.z += tpptr->v.z * fix_speed;
                    tpptr->v.y = -tpptr->v.y;
                    hit_num++;
                }
                if (tp.z >= gdimz || tp.z < 0) {
                    tp.z -= ((fz < 0 && abs(fz) > z) ? 0 : fz);
                    if (tp.z >= gdimz || tp.z < 0) 
                        tp.z %= (gdimz - 1);
                    tpptr->v.y += tpptr->v.y * fix_speed;
                    tpptr->v.x += tpptr->v.x * fix_speed;
                    tpptr->v.z = -tpptr->v.z;
                    hit_num++;
                }
                if (tp == tp_old)
                    break;
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

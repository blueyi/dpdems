#include "../include/common.h"

extern void runError(const std::string &err, const std::string &msg)
{
    throw std::runtime_error(err + " error: " + msg);
}

extern XYZ<int> scale(const XYZ<double> &cor, const XYZ<double> &scal)
{
    double tx = cor.x * scal.x; 
    double ty = cor.y * scal.y; 
    double tz = cor.z * scal.z; 
    XYZ<int> t((int)tx, (int)ty, (int)tz);
    return t;
}

extern unsigned axis_conv(int x, int offset)
{
    return x + offset;
}

extern XYZ<unsigned> axis_conv(const XYZ<int> &cor, const XYZ<int> &offset, bool re) 
{
    unsigned tx, ty, tz;

    if (re) {
        tx = axis_conv(cor.x, -offset.x);
        ty = axis_conv(cor.y, -offset.y);
        tz = axis_conv(cor.z, -offset.z);
    }
    else {
        tx = axis_conv(cor.x, offset.x);
        ty = axis_conv(cor.y, offset.y);
        tz = axis_conv(cor.z, offset.z);
    }
    XYZ<unsigned> t(tx, ty, tz);
    return t;
}

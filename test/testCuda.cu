/*
 * testCuda.cu
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/common.h"
#include <iostream>

__global__ void add(XYZ<double> *a, XYZ<double> *b, XYZ<int> *c)
{
    c->x = a->x + b->x;
    c->y = a->y + b->y;
    c->z = a->z + b->z;
}

int main(void)
{
    XYZ<double> a(1.0, 2.0, 3.0);
    XYZ<double> b(1.0, 2.0, 3.0);
    XYZ<int> c;
    XYZ<double> *dev_a;
    XYZ<double> *dev_b;
    XYZ<int> *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(a));
    cudaMalloc((void**)&dev_b, sizeof(b));
    cudaMalloc((void**)&dev_c, sizeof(c));
    cudaMemcpy(dev_a, &a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, sizeof(b), cudaMemcpyHostToDevice);
    add<<<1, 1>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(&c, &dev_c, sizeof(c), cudaMemcpyDeviceToHost);
    c.print(std::cout);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}



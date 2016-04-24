/*
 * testCuda.cu
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
struct XYZ {
    int x;
    int y;
    int z;
};

__global__ void add(XYZ *a, XYZ *b, XYZ *c)
{
    c->x = a->x + b->x;
    c->y = a->y + b->y;
    c->z = a->z + b->z;
}

int main(void)
{
    XYZ a{1, 2, 3};
    XYZ b{1, 2, 3};
    XYZ c;
    XYZ *dev_a;
    XYZ *dev_b;
    XYZ *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(a));
    cudaMalloc((void**)&dev_b, sizeof(b));
    cudaMalloc((void**)&dev_c, sizeof(c));
    cudaMemcpy(dev_a, &a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, sizeof(b), cudaMemcpyHostToDevice);
    add<<<1, 1>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(&c, &dev_c, sizeof(c), cudaMemcpyDeviceToHost);
    std::cout << c.x << " " << c.y << " " << c.z << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}



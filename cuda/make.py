
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016  <@BLUEYI-PC>
#
# Distributed under terms of the MIT license.

import os
import platform

compile_config = os.system('nvcc -c ../src/config.cpp  --ptxas-options=-v -std=c++11')
compile_inputdata = os.system('nvcc -c ../src/inputDatas.cpp  --ptxas-options=-v -std=c++11')
compile_common = os.system('nvcc -c ../src/common.cpp  --ptxas-options=-v -std=c++11')
compile_search = os.system('nvcc -c ./dpdems_cuda.cu  --ptxas-options=-v -std=c++11')

if compile_inputdata == 0 and compile_common == 0 and compile_search == 0 and compile_config == 0 :
  if platform.system() == 'Windows' : 
    link = os.system('nvcc common.obj config.obj inputDatas.obj dpdems_cuda.obj -o Dpdems_cuda -std=c++11')
  else:
    link = os.system('nvcc common.o config.o inputDatas.o dpdems_cuda.o -o Dpdems_cuda -std=c++11')
if link == 0 :
  print('link success!')
  isrun = input('Run? y or n : ')
  if isrun == 'y':
    if platform.system() == 'Windows' : 
      os.system('Dpdems_cuda.exe')
    else :
      os.system('./Dpdems_cuda')
else :
  print('compile error')




#! \usr\bin\env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016  <@BLUEYI-PC>
#
# Distributed under terms of the MIT license.

'''

'''
import os
import platform

compile_config = os.system('nvcc -c ..\src\config.cpp --std=c++11 --ptxas-options=-v')
compile_inputdata = os.system('nvcc -c ..\src\inputDatas.cpp --std=c++11 --ptxas-options=-v')
compile_common = os.system('nvcc -c ..\src\common.cpp --std=c++11 --ptxas-options=-v')
compile_search = os.system('nvcc -c .\dpdems_cuda.cu --std=c++11 --ptxas-options=-v')

if compile_inputdata == 0 and compile_common == 0 and compile_search == 0 and compile_config == 0 :
  link = os.system('nvcc inputDatas.obj common.obj config.obj dpdems_cuda.obj -o Dpdems_cuda --std=c++11')
  if link == 0 :
    print('link success!')
    isrun = input('Run? y or n : ')
    if isrun == 'y':
      if platform.system() == 'Windows' : 
        os.system('Dpdems_cuda.exe')
      else :
        os.system('./Dpdems_cuda.exe')
else :
  print('compile error')



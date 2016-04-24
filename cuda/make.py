
#! \usr\bin\env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016  <@BLUEYI-PC>
#
# Distributed under terms of the MIT license.

import os
import platform

compile_config = os.system('nvcc -c ..\src\config.cpp  --ptxas-options=-v')
compile_inputdata = os.system('nvcc -c ..\src\inputDatas.cpp  --ptxas-options=-v')
compile_common = os.system('nvcc -c ..\src\common.cpp  --ptxas-options=-v')
compile_search = os.system('nvcc -c .\dpdems_cuda.cu  --ptxas-options=-v')

if compile_inputdata == 0 and compile_common == 0 and compile_search == 0 and compile_config == 0 :
  link = os.system('nvcc common.obj config.obj inputDatas.obj dpdems_cuda.obj -o Dpdems_cuda ')
  if link == 0 :
    print('link success!')
    os.system('Dpdems_cuda.exe')
#    isrun = input('Run? y or n : ')
#    if isrun == 'y':
#      if platform.system() == 'Windows' : 
#        os.system('Dpdems_cuda.exe')
#      else :
else :
  print('compile error')



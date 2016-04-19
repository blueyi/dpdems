
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016  <@BLUEYI-PC>
#
# Distributed under terms of the MIT license.

'''

'''
import os

compile_inputdata = os.system('g++ -c -g ../src/inputDatas.cpp -std=c++11')
compile_common = os.system('g++ -c -g ../src/common.cpp -std=c++11')
compile_grid = os.system('g++ -c -g ../src/grid.cpp -std=c++11')
compile_search = os.system('g++ -c -g ./main.cpp -std=c++11')

if compile_inputdata == 0 and compile_common == 0 and compile_search == 0 and compile_grid == 0 :
  link = os.system('g++ inputDatas.o common.o main.o grid.o -o main -std=c++11 -g')
  if link == 0 :
    print('link success!')
    isrun = input('Run? y or n : ')
    if isrun == 'y':
      os.system('main.exe')
else :
  print('compile error')



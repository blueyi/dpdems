#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016  <@BLUEYI-PC>
#
# Distributed under terms of the MIT license.

import os
import platform

list_num = ['512', '1024', '5120', '10240', '51200', '102400']
sys_str = ''
if platform.system() == 'Windows' :
  sys_str = 'Dpdems.exe'
else: 
  sys_str = './Dpdems'
isRun = 0
for i in range(len(list_num)) :
  command = sys_str + ' ' + list_num[i]
  isRun = os.system(command)
  if isRun != 0 :
    break

if isRun == 0:  
  if platform.system() == 'Windows' :
    sys_str = 'Dpdems_cuda.exe'
  else: 
    sys_str = './Dpdems_cuda'
  for i in range(len(list_num)) :
    command = sys_str + ' ' + list_num[i]
    isRun = os.system(command)
    if isRun != 0 :
      break



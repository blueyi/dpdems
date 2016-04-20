/*
 * config.h
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <sstream>
#include <vector>
#include <map>


extern std::string dPath, dataFile;
extern int timestep, maxdim, stepnum, oFNum, eDuration, cSInterval, dOFile, sFQ, nProc, boundType, isBounding;
extern double oFInterval, dOADraw, dOInfo, fDBeg, fDInt, gravity, mubed, mus, res, waterLine, waterVel, windVel, cDF, cDM, aMCoef, nStrength, tStrength; 

//将所有参数读入到map中，返回读取的参数数量
std::size_t read_conf(const char *fstr, std::map<std::string, std::string> &con);

//从map中获取所需要的变量，返回生成的变量数量
template <typename T> std::size_t inconf(const std::map<std::string, std::string> &aval, const std::vector<std::string> &valname, const std::vector<T *> &val);

//输入参数配置文件，进行参数配置
bool ini_conf(const std::string &);

#endif /* !CONFIG_H */

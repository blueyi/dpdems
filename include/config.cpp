/*
 * config.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "config.h"
#include "common.h"
#include <fstream>

std::map<std::string, std::string> confin;


std::string dPath;
int oFNum, eDuration, cSInterval, dOFile, sFQ, nProc, boundType, isBounding;
double oFInterval, dOADraw, dOInfo, fDBeg, fDInt, gravity, mubed, mus, res, waterLine, waterVel, windVel, cDF, cDM, aMCoef, nStrength, tStrength; 

std::vector<int *> confi{&oFNum, &eDuration, &cSInterval, &dOFile, &sFQ, &nProc, &boundType, &isBounding};
std::vector<std::string> confis{"oFNum", "eDuration", "cSInterval", "dOFile", "sFQ", "nProc", "boundType", "isBounding"};


std::vector<double *> confd{&oFInterval, &dOADraw, &dOInfo, &fDBeg, &fDInt, &gravity, &mubed, &mus, &res, &waterLine, &waterVel, &windVel, &cDF, &cDM, &aMCoef, &nStrength, &tStrength};
std::vector<std::string> confds{"oFInterval", "dOADraw", "dOInfo", "fDBeg", "fDInt", "gravity", "mubed", "mus", "res", "waterLine", "waterVel", "windVel", "cDF", "cDM", "aMCoef", "nStrength", "tStrength"};




template <typename T> std::size_t inconf(const std::map<std::string, std::string> &con, const std::vector<std::string> &valname, const std::vector<T *> &val)
{
    std::size_t cont = 0; 
    for (auto v : val) {
        auto valpair = con.find(valname[cont]);
        if (valpair == con.end())
            runError("Config parameter", valname[cont]);
        std::istringstream is(valpair->second);
        is >> *v;
        ++cont;
    }
    return cont;
}

std::size_t read_conf(const char *fstr, std::map<std::string, std::string> &con)
{
    std::size_t count = 0;
    std::ifstream inf(fstr);
    if (!inf) {
        runError("Config file read", fstr);
        return 0;
    }
    std::string line, val, arg;
    while (getline(inf, line)) {
        std::istringstream is(line);
        is >> val >> arg;
        con[arg] = val;
        ++count;
    }
    return count;
}


bool ini_conf(const char *filepath)
{
    std::size_t acont = read_conf(filepath, confin);
    std::string dPath = confin["dPath"];
    if (dPath.empty())
        runError("Config parameter", "dPath");
    std::size_t icont = inconf(confin, confis, confi);
    std::size_t dcont = inconf(confin, confds, confd);
    if (acont == icont + dcont)
        return true;
    else
        return false;
}

/*
 * config.cpp
 * Copyright (C) 2016  <@BLUEYI-PC>
 *
 * Distributed under terms of the MIT license.
 */

#include "../include/config.h"
#include "../include/common.h"
#include <fstream>

std::map<std::string, std::string> confin;


std::string dPath, dataFile;
int timestep, maxdim, stepnum, oFNum, eDuration, cSInterval, dOFile, sFQ, nProc, boundType, isBounding;
double oFInterval, dOADraw, dOInfo, fDBeg, fDInt, gravity, mubed, mus, res, waterLine, waterVel, windVel, cDF, cDM, aMCoef, nStrength, tStrength; 

std::vector<int *> confi{&timestep, &maxdim, &stepnum, &oFNum, &eDuration, &cSInterval, &dOFile, &sFQ, &nProc, &boundType, &isBounding};
std::vector<std::string> confis{"timestep", "maxdim", "stepnum", "oFNum", "eDuration", "cSInterval", "dOFile", "sFQ", "nProc", "boundType", "isBounding"};


std::vector<double *> confd{&oFInterval, &dOADraw, &dOInfo, &fDBeg, &fDInt, &gravity, &mubed, &mus, &res, &waterLine, &waterVel, &windVel, &cDF, &cDM, &aMCoef, &nStrength, &tStrength};
std::vector<std::string> confds{"oFInterval", "dOADraw", "dOInfo", "fDBeg", "fDInt", "gravity", "mubed", "mus", "res", "waterLine", "waterVel", "windVel", "cDF", "cDM", "aMCoef", "nStrength", "tStrength"};




template <typename T> std::size_t inconf(const std::map<std::string, std::string> &con, const std::vector<std::string> &valname, const std::vector<T *> &val)
{
    std::size_t cont = 0; 
    for (auto v : val) {
        auto valpair = con.find(valname[cont]);
        if (valpair == con.end()) {
            runError("Config parameter", valname[cont]);
            return 0;
        }
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
        while (line.empty() || line == "\r")
            getline(inf, line);
        std::istringstream is(line);
        is >> val >> arg;
        con[arg] = val;
        ++count;
    }
    inf.close();
    return count;
}


bool ini_conf(const std::string &filepath)
{
    std::size_t acont = read_conf(filepath.c_str(), confin);
    if (acont == 0)
        return false;
    dPath = confin["dPath"];
    dataFile = confin["dataFile"];
    if (dPath.empty())
        runError("Config parameter", "dPath");
    std::size_t icont = inconf(confin, confis, confi);
    std::size_t dcont = inconf(confin, confds, confd);
    if (acont == icont + dcont + 2)
        return true;
    else
        return false;
}

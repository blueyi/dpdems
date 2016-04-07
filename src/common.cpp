#include "../include/common.h"

extern void runError(const std::string &err, const std::string &msg)
{
    throw std::runtime_error(err + " error: " + msg);
}

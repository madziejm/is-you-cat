#pragma once

#include "hsv.hpp"

struct rgb {
    double r;       // from [0, 1]
    double g;       // from [0, 1]
    double b;       // from [0, 1]
};

rgb interpolate_green_red(float level);

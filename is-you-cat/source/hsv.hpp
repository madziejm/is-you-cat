#pragma once

#include "rgb.hpp"
struct rgb;
struct hsv {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1

    hsv& operator*(float m);
    hsv& operator+(const hsv& c);
    hsv& operator-(const hsv& c);
};

hsv rgb2hsv(rgb in);
rgb hsv2rgb(hsv in);

#include "hsv.hpp"
#include "rgb.hpp"

rgb interpolate_green_red(float level)
// convert level from [0.0, 1.0] numerical range to color between green and red
{
  hsv hsv_red = rgb2hsv({1.0, 0.0, 0.0});
  hsv hsv_green = rgb2hsv({0.0, 1.0, 0.0});
  return hsv2rgb(hsv_red + (hsv_green - hsv_red) * level);
}

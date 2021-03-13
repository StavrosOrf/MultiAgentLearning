// Some functions migrated from rebhuhnc/libraries/Math/easymath.h
#ifndef UTILITIES_H_
#define UTILITIES_H_

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

#include <vector>
#include <math.h>
#include <stdlib.h>

namespace easymath{
// Returns a random number between two values
double rand_interval(double low, double high) ;

// Normalise angles between +/-PI
double pi_2_pi(double) ;

// Sum elements in a vector
double sum(std::vector<double>) ;
} // namespace easymath
#endif // UTILITIES_H_

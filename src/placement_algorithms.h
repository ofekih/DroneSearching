#pragma once

#include "utils.hpp"
#include <vector>

/**
 * Implementation of place_algorithm_4 which places circles around the perimeter
 * of a unit circle.
 * 
 * @param p The scaling parameter for the radius calculation
 * @param pk Function to calculate radius based on p and k (defaults to p^k)
 * @return Vector of circles placed around the perimeter
 */
std::vector<Circle> place_algorithm_4(
    double p, 
    std::function<double(double, double)> pk = default_pk
);

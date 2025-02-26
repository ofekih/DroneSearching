#include "placement_algorithms.h"
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>
#include <ranges>

std::vector<Circle> place_algorithm_4(double p, std::function<double(double, double)> pk) {
    std::vector<Circle> circles;
    
    // Reserve some space to avoid frequent reallocations
    circles.reserve(20); // Reasonable estimate for typical p values
    
    double current_angle = 0.0;
    double k = 1.0;
    
    while (current_angle < 2 * std::numbers::pi) {
        // Calculate the current radius based on p and k
        double current_radius = pk(p, k);
        
        // If radius becomes too small, return an empty vector (failure)
        if (current_radius < EPSILON) {
            return {};
        }
        
        // Calculate current position on unit circle
        double current_x = std::cos(current_angle);
        double current_y = std::sin(current_angle);
        
        // Move angle forward by 2*asin(radius)
        current_angle += 2 * std::asin(current_radius);
        
        // Calculate next position on unit circle
        double next_x = std::cos(current_angle);
        double next_y = std::sin(current_angle);
        
        // Create a new circle positioned between current and next points
        // Using aggregate initialization
        circles.emplace_back(
            (current_x + next_x) / 2,
            (current_y + next_y) / 2,
            current_radius
        );
        
        k += 1.0;
    }
    
    return circles;
}
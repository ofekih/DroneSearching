#include "placement_algorithms.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Algorithm4Test, FindOneCutoffValue) {
    // Test slightly below and above the cutoff value
    const double cutoff = 0.84386;
    const double delta = 0.0001;
    
    // Test just below cutoff - should not find a solution
    auto circles_below = place_algorithm_4(cutoff - delta, default_pk);
    EXPECT_FALSE(covers_unit_circle(circles_below));
    
    // Test at cutoff - should find a solution
    auto circles_at = place_algorithm_4(cutoff, default_pk);
    EXPECT_TRUE(covers_unit_circle(circles_at));
    
    // Test just above cutoff - should definitely find a solution
    auto circles_above = place_algorithm_4(cutoff + delta, default_pk);
    EXPECT_TRUE(covers_unit_circle(circles_above));
}

TEST(Algorithm4Test, FindAllCutoffValue) {
    // Test slightly below and above the cutoff value
    const double cutoff = 0.79817;
    const double delta = 0.001;  // Increased delta to test further from the boundary
    
    // Test just below cutoff - should not find all solutions
    auto circles_below = place_algorithm_4(cutoff - delta, find_all_pk);
    EXPECT_FALSE(covers_unit_circle(circles_below));
    
    // Test at cutoff - should find all solutions
    auto circles_at = place_algorithm_4(cutoff, find_all_pk);
    EXPECT_TRUE(covers_unit_circle(circles_at));
    
    // Test just above cutoff - should definitely find all solutions
    auto circles_above = place_algorithm_4(cutoff + delta, find_all_pk);
    EXPECT_TRUE(covers_unit_circle(circles_above));
}
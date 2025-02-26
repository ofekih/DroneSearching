#pragma once

#include <vector>
#include <cmath>
#include <numbers>
#include <optional>
#include <algorithm>
#include <ranges>
#include <concepts>
#include <array>
#include <functional>

// Small value to handle floating-point precision issues
inline constexpr double EPSILON = 1e-3;

/**
 * Represents a circle with a center point (x,y) and radius r
 */
struct Circle {
    double x; // x-coordinate of circle center
    double y; // y-coordinate of circle center
    double r; // radius of circle
    
    // Constructor
    constexpr Circle(double x, double y, double r) noexcept : x(x), y(y), r(r) {}
};

// Define a concept for radius calculation functions
template<typename F>
concept RadiusFunctionConcept = requires(F f, double p, double k) {
    { f(p, k) } -> std::convertible_to<double>;
};

// Default pk function that returns p^k
inline constexpr auto default_pk = [](double p, double k) -> double {
    return std::pow(p, k);
};

// pk function that returns p^((k+1)/2) for find-all mode
inline constexpr auto find_all_pk = [](double p, double k) -> double {
    return std::pow(p, (k + 1.0) / 2.0);
};

/**
 * Represents a horizontal line segment
 */
struct HorizontalLine {
    double start; // x-coordinate of start point
    double end;   // x-coordinate of end point
    
    // Constructor
    constexpr HorizontalLine(double start, double end) noexcept : start(start), end(end) {}
    
    // Check if this line contains another line
    [[nodiscard]] constexpr bool contains(const HorizontalLine& other) const noexcept {
        return start <= other.start && end >= other.end;
    }
    
    // Check if this line overlaps with another line
    [[nodiscard]] constexpr bool overlaps(const HorizontalLine& other) const noexcept {
        return start <= other.end && end >= other.start;
    }
    
    // Create a union of two overlapping lines
    [[nodiscard]] constexpr HorizontalLine merge(const HorizontalLine& other) const noexcept {
        return HorizontalLine(
            std::min(start, other.start),
            std::max(end, other.end)
        );
    }
};

/**
 * Represents a square with a corner at (x,y) and a given side length
 */
struct Square {
    double x; // x-coordinate of bottom-left corner
    double y; // y-coordinate of bottom-left corner
    double side_length; // length of the square's side
    
    // Constructor
    constexpr Square(double x, double y, double side_length) noexcept 
        : x(x), y(y), side_length(side_length) {}
};

// The unit circle centered at (0,0) with radius 1
inline constexpr Circle UNIT_CIRCLE{0.0, 0.0, 1.0};

/**
 * Checks if a point (x,y) is covered by a circle
 */
[[nodiscard]] constexpr bool is_point_covered(const Circle& circle, double x, double y) noexcept {
    return (x - circle.x) * (x - circle.x) + (y - circle.y) * (y - circle.y) <= circle.r * circle.r;
}

/**
 * Checks if a point (x,y) is covered by any circle in the given collection
 */
template<typename CircleContainer>
requires std::ranges::range<CircleContainer>
[[nodiscard]] constexpr bool is_point_covered_by_any(
    const CircleContainer& circles, double x, double y) noexcept {
    return std::ranges::any_of(circles, [x, y](const auto& circle) {
        return is_point_covered(circle, x, y);
    });
}

/**
 * Checks if a square is fully covered by a circle
 */
[[nodiscard]] constexpr bool is_fully_covered(const Square& square, const Circle& circle) noexcept {
    // Check all four corners of the square
    return is_point_covered(circle, square.x, square.y) &&
           is_point_covered(circle, square.x + square.side_length, square.y) &&
           is_point_covered(circle, square.x, square.y + square.side_length) &&
           is_point_covered(circle, square.x + square.side_length, square.y + square.side_length);
}

/**
 * Gets a horizontal line segment where a circle intersects the given y-coordinate
 */
[[nodiscard]] std::optional<HorizontalLine> get_horizontal_line(const Circle& circle, double y) noexcept;

/**
 * Merges a collection of horizontal lines into a minimal set of non-overlapping lines
 */
template<typename LineContainer>
requires std::ranges::range<LineContainer>
[[nodiscard]] std::vector<HorizontalLine> get_line_union(const LineContainer& lines) {
    if (lines.empty()) {
        return {};
    }
    
    // Create a copy of the lines and sort them by start position
    std::vector<HorizontalLine> sorted_lines(std::begin(lines), std::end(lines));
    std::ranges::sort(sorted_lines, [](const auto& a, const auto& b) {
        return a.start < b.start;
    });
    
    std::vector<HorizontalLine> result;
    auto current_line = sorted_lines.front();
    
    for (const auto& line : sorted_lines | std::views::drop(1)) {
        if (line.start <= current_line.end) {
            // Lines overlap, merge them
            current_line = current_line.merge(line);
        } else {
            // No overlap, add the current line to the result and start a new one
            result.push_back(current_line);
            current_line = line;
        }
    }
    
    // Don't forget the last line
    result.push_back(current_line);
    return result;
}

/**
 * Checks if the given set of circles covers the horizontal slice of the unit circle at height y
 */
[[nodiscard]] bool do_circles_cover_unit_circle(const std::vector<Circle>& circles, double y);

/**
 * Implementation of covers_unit_circle - version 2
 * 
 * Checks if the given circles cover the entire unit circle by sampling
 * horizontal slices at regular intervals.
 */
[[nodiscard]] bool covers_unit_circle_2(const std::vector<Circle>& circles);

/**
 * Implementation of covers_unit_circle
 * 
 * Checks if the given circles cover the entire unit circle by using
 * a recursive subdivision approach with squares.
 */
[[nodiscard]] bool covers_unit_circle(const std::vector<Circle>& circles);

/**
 * Recursively checks if a square is covered by a set of circles
 * 
 * A square is considered covered if:
 * 1. All corners inside the unit circle are covered by at least one circle, or
 * 2. The square is entirely outside the unit circle, or
 * 3. The square is entirely covered by one of the circles, or
 * 4. When divided into four sub-squares, all sub-squares are covered
 */
[[nodiscard]] bool is_square_covered(
    const std::vector<Circle>& circles, const Square& square);

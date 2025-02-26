#include "utils.hpp"

std::optional<HorizontalLine> get_horizontal_line(const Circle& circle, double y) noexcept {
    // Check if y is within the circle's vertical range
    if (std::abs(y - circle.y) > circle.r) {
        return std::nullopt;
    }
    
    // Calculate the half-width of the horizontal line at height y
    double half_width = std::sqrt(circle.r * circle.r - (y - circle.y) * (y - circle.y));
    
    return HorizontalLine(circle.x - half_width, circle.x + half_width);
}

bool do_circles_cover_unit_circle(const std::vector<Circle>& circles, double y) {
    // Get the horizontal line for the unit circle at height y
    auto unit_line = get_horizontal_line(UNIT_CIRCLE, y);
    if (!unit_line) {
        return true; // y is outside the unit circle's range
    }
    
    // Get all horizontal lines for the given circles at height y
    std::vector<HorizontalLine> lines;
    for (const auto& circle : circles) {
        auto line = get_horizontal_line(circle, y);
        if (line) {
            lines.push_back(*line);
        }
    }
    
    // Calculate the union of all the lines
    auto union_lines = get_line_union(lines);
    
    // Check if any union line covers the unit circle line
    return std::ranges::any_of(union_lines, [&unit_line](const auto& line) {
        return line.start <= unit_line->start && line.end >= unit_line->end;
    });
}

bool covers_unit_circle_2(const std::vector<Circle>& circles) {
    double y = -1.0;
    while (y < 1.0) {
        if (!do_circles_cover_unit_circle(circles, y)) {
            return false;
        }
        y += EPSILON;
    }
    return true;
}

bool is_square_covered(const std::vector<Circle>& circles, const Square& square) {
    // Get the four corners of the square
    std::vector<std::pair<double, double>> corners = {
        {square.x, square.y}, 
        {square.x + square.side_length, square.y},
        {square.x, square.y + square.side_length}, 
        {square.x + square.side_length, square.y + square.side_length}
    };
    
    // Check if all corners are outside unit circle - if so, ignore this square
    int corners_outside_unit_circle = 0;
    for (const auto& [x, y] : corners) {
        if (!is_point_covered(UNIT_CIRCLE, x, y)) {
            corners_outside_unit_circle++;
        }
    }
    if (corners_outside_unit_circle == 4) {
        return true;
    }
    
    // Count corners inside the unit circle but not covered by any circle
    for (const auto& [x, y] : corners) {
        if (is_point_covered(UNIT_CIRCLE, x, y) && !is_point_covered_by_any(circles, x, y)) {
            return false;
        }
    }
    
    // If square is entirely covered by any circle, it's covered
    if (std::ranges::any_of(circles, [&square](const auto& circle) {
        return is_fully_covered(square, circle);
    })) {
        return true;
    }
    
    // If the square is small enough, consider it covered
    if (square.side_length < EPSILON) {
        return true;
    }
    
    // Divide into four sub-squares and check recursively
    double new_side = square.side_length / 2.0;
    std::array<Square, 4> subsquares = {
        Square(square.x, square.y, new_side),
        Square(square.x + new_side, square.y, new_side),
        Square(square.x, square.y + new_side, new_side),
        Square(square.x + new_side, square.y + new_side, new_side)
    };
    
    // Check all sub-squares
    return std::ranges::all_of(subsquares, [&circles](const auto& subsquare) {
        return is_square_covered(circles, subsquare);
    });
}

bool covers_unit_circle(const std::vector<Circle>& circles) {
    // Check all four quadrants with 1x1 squares instead of one 2x2 square
    std::array<Square, 4> initial_squares = {
        Square(-1.0, -1.0, 1.0),
        Square(-1.0, 0.0, 1.0),
        Square(0.0, -1.0, 1.0),
        Square(0.0, 0.0, 1.0)
    };
    
    return std::ranges::all_of(initial_squares, [&circles](const auto& square) {
        return is_square_covered(circles, square);
    });
}
#include <cmath>
#include <iostream>
#include <vector>
#include <geos_c.h>
#include "placement_algorithms.h"
#include "utils.hpp"

// Global GEOS context handle
GEOSContextHandle_t handle = nullptr;

GEOSGeometry* circleToPolygon(double centerX, double centerY, double radius, int segments = 32) {
    // Create coordinate sequence
    GEOSCoordSequence* seq = GEOSCoordSeq_create_r(handle, segments + 1, 2);
    
    // Fill coordinate sequence
    for (int i = 0; i < segments; i++) {
        double angle = 2.0 * M_PI * i / segments;
        double x = centerX + radius * cos(angle);
        double y = centerY + radius * sin(angle);
        GEOSCoordSeq_setX_r(handle, seq, i, x);
        GEOSCoordSeq_setY_r(handle, seq, i, y);
    }
    
    // Close the ring
    GEOSCoordSeq_setX_r(handle, seq, segments, centerX + radius);
    GEOSCoordSeq_setY_r(handle, seq, segments, centerY);
    
    // Create linear ring
    GEOSGeometry* ring = GEOSGeom_createLinearRing_r(handle, seq);
    if (!ring) {
        GEOSCoordSeq_destroy_r(handle, seq);
        return nullptr;
    }
    
    // Create polygon
    return GEOSGeom_createPolygon_r(handle, ring, nullptr, 0);
}

GEOSGeometry* createUnitCircle(int segments = 32) {
    return circleToPolygon(0, 0, 1.0, segments);
}

int main() {
    // Initialize GEOS with a single context
    handle = GEOS_init_r();
    if (!handle) {
        std::cerr << "Failed to initialize GEOS" << std::endl;
        return 1;
    }
    
    const auto& circles = place_algorithm_4(0.8, default_pk);
    
    std::vector<GEOSGeometry*> polygons;
    for (const auto& circle : circles) {
        if (auto poly = circleToPolygon(circle.x, circle.y, circle.r)) {
            polygons.push_back(poly);
        }
    }

    GEOSGeometry* union_result = nullptr;
    GEOSGeometry* unit_circle = nullptr;
    char* union_wkt = nullptr;
    char* circle_wkt = nullptr;
    
    try {
        // Create union of all polygons
        if (!polygons.empty()) {
            GEOSGeometry* collection = GEOSGeom_createCollection_r(
                handle, GEOS_GEOMETRYCOLLECTION, 
                polygons.data(), polygons.size()
            );
            
            if (collection) {
                union_result = GEOSUnaryUnion_r(handle, collection);
                GEOSGeom_destroy_r(handle, collection);
            }
        }

        if (union_result) {
            // Create unit circle for comparison
            unit_circle = createUnitCircle(64);  // Using more segments for accuracy
            
            if (unit_circle) {
                // Check if union contains unit circle
                union_wkt = GEOSGeomToWKT_r(handle, union_result);
                circle_wkt = GEOSGeomToWKT_r(handle, unit_circle);
                
                if (union_wkt && circle_wkt) {
                    std::cout << "Union WKT: " << union_wkt << std::endl;
                    std::cout << "Circle WKT: " << circle_wkt << std::endl;
                    
                    // Calculate areas
                    double union_area = 0.0;
                    double circle_area = 0.0;
                    GEOSArea_r(handle, union_result, &union_area);
                    GEOSArea_r(handle, unit_circle, &circle_area);
                    
                    std::cout << "Union area: " << union_area << std::endl;
                    std::cout << "Unit circle area: " << circle_area << std::endl;
                    
                    // Check coverage
                    double coverage_ratio = union_area / circle_area;
                    std::cout << "Coverage ratio: " << coverage_ratio * 100 << "%" << std::endl;
                    
                    // Check if union covers circle using GEOS operations
                    if (GEOSContains_r(handle, union_result, unit_circle)) {
                        std::cout << "The polygons FULLY cover the unit circle!" << std::endl;
                    } else {
                        std::cout << "The polygons do NOT fully cover the unit circle." << std::endl;
                        
                        // Calculate difference to show uncovered area
                        GEOSGeometry* difference = GEOSDifference_r(handle, unit_circle, union_result);
                        if (difference) {
                            double uncovered_area = 0.0;
                            GEOSArea_r(handle, difference, &uncovered_area);
                            std::cout << "Uncovered area: " << uncovered_area << " square units" << std::endl;
                            GEOSGeom_destroy_r(handle, difference);
                        }
                    }
                }
            }
        }
    } catch (...) {
        std::cerr << "An error occurred during geometry operations" << std::endl;
    }

    // Cleanup in reverse order of creation
    if (circle_wkt) GEOSFree_r(handle, circle_wkt);
    if (union_wkt) GEOSFree_r(handle, union_wkt);
    if (unit_circle) GEOSGeom_destroy_r(handle, unit_circle);
    if (union_result) GEOSGeom_destroy_r(handle, union_result);
    
    // Cleanup original polygons
    for (auto poly : polygons) {
        if (poly) GEOSGeom_destroy_r(handle, poly);
    }
    
    // Finish GEOS last
    if (handle) GEOS_finish_r(handle);
    
    return 0;
}

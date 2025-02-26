#include "placement_algorithms.h"
#include "utils.hpp"
#include <benchmark/benchmark.h>
#include <vector>
#include <array>
#include <ranges>

// Benchmark place_algorithm_4 with default_pk function
static void BM_PlaceAlgorithm4_DefaultPk(benchmark::State& state) {
    const double p = static_cast<double>(state.range(0)) / 100.0; // Convert range to a decimal value
    
    for (auto _ : state) {
        auto circles = place_algorithm_4(p, default_pk);
        benchmark::DoNotOptimize(circles);
    }
    
    state.SetComplexityN(state.range(0));
}

// Benchmark place_algorithm_4 with find_all_pk function
static void BM_PlaceAlgorithm4_FindAllPk(benchmark::State& state) {
    const double p = static_cast<double>(state.range(0)) / 100.0; // Convert range to a decimal value
    
    for (auto _ : state) {
        auto circles = place_algorithm_4(p, find_all_pk);
        benchmark::DoNotOptimize(circles);
    }
    
    state.SetComplexityN(state.range(0));
}

// Benchmark covers_unit_circle function
static void BM_CoversUnitCircle(benchmark::State& state) {
    const double p = static_cast<double>(state.range(0)) / 100.0;
    
    // Generate circles only once, outside the benchmark loop
    auto circles = place_algorithm_4(p, default_pk);
    
    for (auto _ : state) {
        bool covered = covers_unit_circle(circles);
        benchmark::DoNotOptimize(covered);
    }
    
    // Set a custom label to show the number of circles in the result
    state.SetLabel(std::to_string(circles.size()) + " circles");
}

// Benchmark covers_unit_circle_2 function
static void BM_CoversUnitCircle2(benchmark::State& state) {
    const double p = static_cast<double>(state.range(0)) / 100.0;
    
    // Generate circles only once, outside the benchmark loop
    auto circles = place_algorithm_4(p, default_pk);
    
    for (auto _ : state) {
        bool covered = covers_unit_circle_2(circles);
        benchmark::DoNotOptimize(covered);
    }
    
    // Set a custom label to show the number of circles in the result
    state.SetLabel(std::to_string(circles.size()) + " circles");
}

// Define the argument values for our benchmarks
constexpr std::array p_values = {50, 60, 70, 80, 90, 95};

// Register benchmarks with various p values ranging from 0.50 to 0.95
BENCHMARK(BM_PlaceAlgorithm4_DefaultPk)
    ->Args({50})
    ->Args({60})
    ->Args({70})
    ->Args({80})
    ->Args({90})
    ->Args({95})
    ->Complexity();

BENCHMARK(BM_PlaceAlgorithm4_FindAllPk)
    ->Args({50})
    ->Args({60})
    ->Args({70})
    ->Args({80})
    ->Args({90})
    ->Args({95})
    ->Complexity();

BENCHMARK(BM_CoversUnitCircle)
    ->Args({76})
    ->Args({80})
    ->Args({90})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CoversUnitCircle2)
    ->Args({76})
    ->Args({80})
    ->Args({90})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
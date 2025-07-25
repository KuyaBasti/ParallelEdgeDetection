#include "parallel_utils.hpp"

#include <omp.h>
#include <random>
#include <chrono>

std::vector<float> nukedata(1);

// In a separate file so the optimizer really REALLY can't do anything about it.
void CacheNukePrepare(size_t size)
{
    // code from CCPreference for random numbers...
    nukedata = std::vector<float>(size);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);
    for (auto &a : nukedata)
    {
        a = (float)dis(gen);
    }
}

float CacheNuke()
{

    auto id = (float)omp_get_thread_num();
    for (const auto &a : nukedata)
    {
        id += a;
    }
    return id;
}

long GetTiming(std::function<void()> f)
{
    auto start = std::chrono::system_clock::now();
    f();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(diff);
    return microsec.count();
}

void parallelNuke(){
#pragma omp parallel 
{
    CacheNuke();
}
}
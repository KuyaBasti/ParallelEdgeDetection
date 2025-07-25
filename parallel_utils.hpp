#ifndef _PARALLEL_UTILS
#define _PARALLEL_UTILS
#include <vector>
#include <functional>

float CacheNuke();
void CacheNukePrepare(size_t size);

long GetTiming(std::function<void()>);
void parallelNuke();

#endif

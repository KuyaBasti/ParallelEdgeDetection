#include "parallel_image.hpp"

// This will ONLY be called if the formatting of the
// image is different in your parallel implementation.
std::shared_ptr<Image> ParallelImage::to_host()
{
    throw ImageException("Need to implement");
    return nullptr;
}
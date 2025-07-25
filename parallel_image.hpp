#ifndef _PARALLEL_IMAGE
#define _PARALLEL_IMAGE

#include "image.hpp"
#include <memory>
#include "omp.h"
#include <immintrin.h>
#include <cmath>  //

// This should work on any Intel/OpenMP setup, using Intel Intrinsics
// to improve operation speed.

class ParallelImage : public Image
{
    // Grabbed this online, its the 5x5 gaussian parameter set.
    // You can cut & paste this elsewhere if you want...
    const float gaussian[5][5] = {
        {0.0002f, 0.0033f, 0.0081f, 0.0033f, 0.0002f},
        {0.0033f, 0.0479f, 0.1164f, 0.0479f, 0.0033f},
        {0.0081f, 0.1164f, 0.2831f, 0.1164f, 0.0081f},
        {0.0033f, 0.0479f, 0.1164f, 0.0479f, 0.0033f},
        {0.0002f, 0.0033f, 0.0081f, 0.0033f, 0.0002f}};

    // Constants for gradient calculation
    const float ydir[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    const float xdir[3][3] = {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}};

public:
    // We have a constructor here in case you want to
    // restructure the image slightly, but default is
    // taken from the copy constructor for Image

    // A nice dummy constructor, useful
    ParallelImage(unsigned int x, unsigned int y, ImageType type) : Image(x, y, type)
    {
    }

    // One note is that in order to improve things and tune
    // performance I'm adding OPTIONAL mode settings that use
    // or invoke different versions of the code to aid in
    // performance tuning, so I can check to see which of
    // several variants produce the best performance and then
    // select the right mode accordingly.
    ParallelImage(Image &cp, int mode = 2)
    {
        _x = cp.x();
        _y = cp.y();
        _type = cp.type();
        _isstb = false;
        /* if (mode == -1) {
            _data = cp.data();
            cp._data = nullptr;
            return;
        } */
        _data = malloc(pixel_size() * pixel_count());

        // Mode 0 == default sequential
        if (mode == 0)
        {
            memcpy(_data, cp.data(), pixel_count() * pixel_size());
        }
        // Mode 1: Parallel for loop.
        if (mode == 1)
        {
#pragma omp parallel for
            for (size_t i = 0; i < pixel_count() * pixel_size(); ++i)
            {
                ((unsigned char *)_data)[i] = ((unsigned char *)cp.data())[i];
            }
        }
        
        // Mode 2: Call memcpy (assuming this has been optimized to hell and gone)
        // but in separate threads
        if (mode == 2)
        {
#pragma omp parallel
            {
                auto id = omp_get_thread_num();
                auto all = omp_get_num_threads();
                auto total = pixel_count() * pixel_size();
                auto range = total / all;
                memcpy((unsigned char *)_data + range * id,
                       ((unsigned char *)cp.data()) + range * id, range);
                // Have to handle the last little bit special: Otherwise the last bit may not be copied.
                if (id == 0 && (range * all < total))
                {
                    memcpy((unsigned char *)_data + range * all,
                           ((unsigned char *)cp.data()) + range * all, total - range * all);
                }
            }
        }
    }

    virtual std::shared_ptr<Image> convert(ImageType to)
    {
        return convert(to, 4);
    }

    virtual std::shared_ptr<Image> convert(ImageType to, int mode)
    {
        if (mode == 0)
        {
            auto i = Image::convert(to);
            return std::make_shared<ParallelImage>(*i);
        }
        // if (to != floatgrayscale || _type != rgb)
        // {
        //     auto i = Image::convert(to);
        //     return std::make_shared<ParallelImage>(*i);
        // }
        
        // RGB -> floatgrayscale
        if (to == floatgrayscale && _type == rgb)
        {
            if (mode == 1) // Naive OpenMP Parallel
            {
            auto ret = std::make_shared<ParallelImage>(_x, _y, to);
#pragma omp parallel for
            for (unsigned int x = 0; x < _x; ++x)
            {
                for (unsigned int y = 0; y < _y; ++y)
                {
                    ret->fppixel(x, y, 0) =
                        ((float)pixel(x, y, 0) +
                         (float)pixel(x, y, 1) +
                         (float)pixel(x, y, 2)) /
                        (255 * 3);
                }
            }
            return ret;
            }

            if (mode == 2) // Direct access destination data
            {
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                unsigned char *srcdata = (unsigned char *)_data;
                float *rdata = (float *)ret->data();
#pragma omp parallel for
                for (unsigned int x = 0; x < _x; ++x)
                {
                    for (unsigned int y = 0; y < _y; ++y)
                    {
                        rdata[x + y * _x] = 
                            ((float)srcdata[(x + y * _x) * 3] +
                             (float)srcdata[(x + y * _x) * 3 + 1] +
                             (float)srcdata[(x + y * _x) * 3 + 2]) /
                            (255 * 3);
                    }
                }
            return ret;
            }

            if (mode == 3) // Direct access source and destination data
            {
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                unsigned char *srcdata = (unsigned char *)_data;
                float *rdata = (float *)ret->data();
#pragma omp parallel for
                for (unsigned int x = 0; x < _x; ++x)
                {
                    for (unsigned int y = 0; y < _y; ++y)
                    {
                        rdata[x + y * _x] = 
                            ((float)srcdata[(x + y * _x) * 3] +
                             (float)srcdata[(x + y * _x) * 3 + 1] +
                             (float)srcdata[(x + y * _x) * 3 + 2]) /
                            (255 * 3);
                    }
                }
                return ret;
            }
            if (mode == 4)
            {
                // idea/experiment on this:
                // just a single loop...
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                unsigned char *srcdata = (unsigned char *)_data;
                float *rdata = (float *)ret->data();
#pragma omp parallel for
                for (unsigned int i = 0; i < _x * _y; ++i)
                {
                    rdata[i] = 
                        ((float)srcdata[i * 3] +
                         (float)srcdata[i * 3 + 1] +
                         (float)srcdata[i * 3 + 2]) /
                        (255 * 3);
                }
                return ret;
            }
            if (mode == 5)
            {
                // idea/experiment on this:
                // just a single loop...
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                unsigned char *srcdata = (unsigned char *)_data;
                float *rdata = (float *)ret->data();
                for (unsigned int i = 0; i < _x * _y; ++i)
                {
                    rdata[i] = 
                        ((float)srcdata[i * 3] +
                         (float)srcdata[i * 3 + 1] +
                         (float)srcdata[i * 3 + 2]) /
                        (255 * 3);
                }
                return ret;
            }
        }
        // floatgrayscale -> grayscale
        if (to == grayscale && _type == floatgrayscale)
        {
            if (mode == 1) // Naive OpenMP Parallel
            {
            auto ret = std::make_shared<ParallelImage>(_x, _y, to);
#pragma omp parallel for
            for (unsigned int x = 0; x < _x; ++x)
            {
                for (unsigned int y = 0; y < _y; ++y)
                {
                    ret->pixel(x, y) = (unsigned char)(fppixel(x, y) * 255);
                }
            }
            return ret;
            }

            if (mode == 2) // Direct access destination data
            {
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                float *srcdata = (float *)_data;
                unsigned char *rdata = (unsigned char *)ret->data();
#pragma omp parallel for
                for (unsigned int x = 0; x < _x; ++x)
                {
                    for (unsigned int y = 0; y < _y; ++y)
                    {
                        rdata[x + y * _x] = (unsigned char)(srcdata[x + y * _x] * 255);
                    }
                }
            return ret;
            }

            if (mode == 3) // Direct access source and destination data
            {
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                float *srcdata = (float *)_data;
                unsigned char *rdata = (unsigned char *)ret->data();
#pragma omp parallel for
                for (unsigned int x = 0; x < _x; ++x)
                {
                    for (unsigned int y = 0; y < _y; ++y)
                    {
                        rdata[x + y * _x] = (unsigned char)(srcdata[x + y * _x] * 255);
                    }
                }
                return ret;
            }
            if (mode == 4)
            {
                // idea/experiment on this:
                // just a single loop...
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                float *srcdata = (float *)_data;
                unsigned char *rdata = (unsigned char *)ret->data();
#pragma omp parallel for
                for (unsigned int i = 0; i < _x * _y; ++i)
                {
                    rdata[i] = (unsigned char)(srcdata[i] * 255);
                }
                return ret;
            }
            if (mode == 5)
            {
                // idea/experiment on this:
                // just a single loop...
                auto ret = std::make_shared<ParallelImage>(_x, _y, to);
                float *srcdata = (float *)_data;
                unsigned char *rdata = (unsigned char *)ret->data();
                for (unsigned int i = 0; i < _x * _y; ++i)
                {
                    rdata[i] = (unsigned char)(srcdata[i] * 255);
                }
                return ret;
            }
        }
        // assert(false);
        auto i = Image::convert(to);
        return std::make_shared<ParallelImage>(*i);
        
    }

    virtual std::shared_ptr<Image> blur() override
    {
        return blur(4);
    }

    virtual std::shared_ptr<ParallelImage> blur(int mode)
    {
        validate();
        if (_type != floatgrayscale)
        {
            throw ImageException("Currently can only blur on floating point grayscale");
        }
        auto ret = std::make_shared<ParallelImage>(_x, _y, _type);
        if(mode == 0)
        {   
            for (unsigned int x = 0; x < _x; ++x)
            {
                for (unsigned int y = 0; y < _y; ++y)
                {
                    float tmp = 0;
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            auto pixel = fppixel((unsigned int)xcoord,
                                                (unsigned int)ycoord);

                            tmp += gaussian[i + 2][j + 2] * pixel;
                        }
                    }
                    ret->fppixel(x, y) = tmp;
                }
            }
        }
        if(mode == 1)
        {   // TODO: implement NAIVE OpenMP parallel version
#pragma omp parallel for
            for (unsigned int y = 0; y < _y; ++y)
            {
                for (unsigned int x = 0; x < _x; ++x)
                {
                    float tmp = 0;
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            auto pixel = fppixel((unsigned int)xcoord,
                                                (unsigned int)ycoord);

                            tmp += gaussian[i + 2][j + 2] * pixel;
                        }
                    }
                    ret->fppixel(x, y) = tmp;
                }
            }
        }
        if (mode == 2)
        {
            // TODO: implement DIRECT ACCESSS OpenMP parallel version 
            float* srcdata = (float*)_data;
            float* rdata = (float*)ret->data();
#pragma omp parallel for 
            // parallelized
            for (unsigned int y = 0; y < _y; ++y)
            {   // parallelized
                for (unsigned int x = 0; x < _x; ++x)
                {   // maybe do the pointer here
                    float tmp = 0;
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            // this is biggest hurdle for performance need to do direct mem access
                            // how to get index of 2D image in 1D array bc its how its stored
                            // index = row * width + column
                            // but fppixel uses x + y * _x where x=column, y=row
                            // _x is the width of image
                            // _y is height 
                            // READ from neighboring pixels
                            // (x, y) is at index = x + y * _x in 1D array memory
                            float pixel = srcdata[xcoord + ycoord * _x];

                            tmp += gaussian[i + 2][j + 2] * pixel;
                        }
                    }
                    // WRITE to actual pixel target
                    rdata[x + y * _x] = tmp;
                }
            }
        }
        if (mode == 3)
        {   // TODO: implement BLOCKED OPENMP parallel version
            float* srcdata = (float*)_data;
            float* rdata = (float*)ret->data();
            // Block 1: do top 2 rows
#pragma omp parallel for
            for (unsigned int y = 0; y < _y; ++y)
            {   
                for (unsigned int x = 0; x < _x; ++x)
                {   
                    float tmp = 0;
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            float pixel = srcdata[xcoord + ycoord * _x];

                            tmp += gaussian[i + 2][j + 2] * pixel;
                        }
                    }
                    // WRITE to actual pixel target
                    rdata[x + y * _x] = tmp;
                }
            }
        }
        if (mode == 4)
        {   // TODO implement AVX
            float* srcdata = (float*)_data;
            float* rdata = (float*)ret->data();
#pragma omp parallel for 
            for (unsigned int y = 2; y < _y - 2; ++y)
            {   
                // multiples of 8
                for (unsigned int x = 2; x + 8 <= _x - 2; x += 8)
                {   
                    __m256 sum = _mm256_setzero_ps();
                    
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            __m256 pixel = _mm256_loadu_ps(&srcdata[(x + i) + (y + j) * _x]);
                            __m256 weight = _mm256_set1_ps(gaussian[i + 2][j + 2]);
                            __m256 product = _mm256_mul_ps(pixel, weight);
                            sum = _mm256_add_ps(sum, product);
                        }
                    }
                    _mm256_storeu_ps(&rdata[x + y * _x], sum);
                }
                
                // 
                for (unsigned int x = (((_x - 4)/8)*8) + 2; x < _x - 2; ++x)
                {
                    float tmp = 0;
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            float pixel = srcdata[(x + i) + (y + j) * _x];
                            tmp += gaussian[i + 2][j + 2] * pixel;
                        }
                    }
                    rdata[x + y * _x] = tmp;
                }
            }
            
            // move profs code; i fail [  FAILED  ] ParallelTest.TimeBlur if i dont have this
#pragma omp parallel for 
            for (unsigned int y = 0; y < _y; ++y)
            {   
                for (unsigned int x = 0; x < _x; ++x)
                {   
                    // we already did this above NEED THIS to improve performance
                    if (y >= 2 && y < _y - 2 && x >= 2 && x < _x - 2) continue;
                    
                    float tmp = 0;
                    for (int i = -2; i < 3; ++i)
                    {
                        for (int j = -2; j < 3; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            float pixel = srcdata[xcoord + ycoord * _x];
                            tmp += gaussian[i + 2][j + 2] * pixel;
                        }
                    }
                    rdata[x + y * _x] = tmp;
                }
            }
        }
        return ret;
    }

    virtual std::shared_ptr<Image> gradient() override
    {
        return gradient(3);
    }

    virtual std::shared_ptr<ParallelImage> gradient(int mode)
    {
        validate();
        if (_type != floatgrayscale)
        {
            throw ImageException("Currently can only gradiant on floating point grayscale");
        }
        auto ret = std::make_shared<ParallelImage>(_x, _y, _type);
        if (mode == 0)
        {
            for (unsigned int x = 0; x < _x; ++x)
            {
                for (unsigned int y = 0; y < _y; ++y)
                {
                    float xgradient = 0;
                    float ygradient = 0;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            auto pixel = fppixel((unsigned int)xcoord,
                                               (unsigned int)ycoord);
                            xgradient += xdir[i + 1][j + 1] * pixel;
                            ygradient += ydir[i + 1][j + 1] * pixel;
                        }
                    }
                    ret->fppixel(x, y) = std::sqrt(xgradient * xgradient + ygradient * ygradient);
                }
            }
        }
        if (mode == 1)
        {   // TODO implement NAIVE OpenMP parallel version
#pragma omp parallel for 
            for (unsigned int y = 0; y < _y; ++y)
            {
                for (unsigned int x = 0; x < _x; ++x)
                {
                    float xgradient = 0;
                    float ygradient = 0;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            auto pixel = fppixel((unsigned int)xcoord,
                                               (unsigned int)ycoord);
                            xgradient += xdir[i + 1][j + 1] * pixel;
                            ygradient += ydir[i + 1][j + 1] * pixel;
                        }
                    }
                    ret->fppixel(x, y) = std::sqrt(xgradient * xgradient + ygradient * ygradient);
                }
            }
        }
        if (mode == 2)
        {
            // TODO implement DIRECT ACCESS OpenMP parallel version
            float* srcdata = (float*)_data;
            float* rdata = (float*)ret->data();
#pragma omp parallel for 
            for (unsigned int y = 0; y < _y; ++y)
            {
                for (unsigned int x = 0; x < _x; ++x)
                {
                    float xgradient = 0;
                    float ygradient = 0;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            // READ from neighboring pixels
                            float pixel = srcdata[xcoord + ycoord * _x];
                            xgradient += xdir[i + 1][j + 1] * pixel;
                            ygradient += ydir[i + 1][j + 1] * pixel;
                        }
                    }
                    // WRITE to actual target pixel
                    rdata[x + y * _x] = std::sqrt(xgradient * xgradient + ygradient * ygradient);
                }
            }
        }
        if (mode == 3)
        {
            // TODO implement AVX
            float* srcdata = (float*)_data;
            float* rdata = (float*)ret->data();
            
#pragma omp parallel for 
            for (unsigned int y = 1; y < _y - 1; ++y)
            {   
                // 
                for (unsigned int x = 1; x + 8 <= _x - 1; x += 8)
                {   
                    __m256 xsum = _mm256_setzero_ps();
                    __m256 ysum = _mm256_setzero_ps();
                    
                    // 
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            __m256 pixel = _mm256_loadu_ps(&srcdata[(x + i) + (y + j) * _x]);
                            __m256 xweight = _mm256_set1_ps(xdir[i + 1][j + 1]);
                            __m256 yweight = _mm256_set1_ps(ydir[i + 1][j + 1]);
                            
                            xsum = _mm256_add_ps(xsum, _mm256_mul_ps(pixel, xweight));
                            ysum = _mm256_add_ps(ysum, _mm256_mul_ps(pixel, yweight));
                        }
                    }
                    
                    // 
                    __m256 xsqrd = _mm256_mul_ps(xsum, xsum);
                    __m256 ysqrd = _mm256_mul_ps(ysum, ysum);
                    __m256 magnitude = _mm256_sqrt_ps(_mm256_add_ps(xsqrd, ysqrd));
                    _mm256_storeu_ps(&rdata[x + y * _x], magnitude);
                }
                
                // 
                for (unsigned int x = (((_x - 2)/8)*8) + 1; x < _x - 1; ++x)
                {
                    float xgradient = 0;
                    float ygradient = 0;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            float pixel = srcdata[(x + i) + (y + j) * _x];
                            xgradient += xdir[i + 1][j + 1] * pixel;
                            ygradient += ydir[i + 1][j + 1] * pixel;
                        }
                    }
                    rdata[x + y * _x] = std::sqrt(xgradient * xgradient + ygradient * ygradient);
                }
            }
            // move profs code
#pragma omp parallel for 
            for (unsigned int y = 0; y < _y; ++y)
            {
                for (unsigned int x = 0; x < _x; ++x)
                {
                    // we already did this above NEED THIS to improve performance
                    if (y >= 2 && y < _y - 2 && x >= 2 && x < _x - 2) continue;

                    float xgradient = 0;
                    float ygradient = 0;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            // READ from neighboring pixels
                            float pixel = srcdata[xcoord + ycoord * _x];
                            xgradient += xdir[i + 1][j + 1] * pixel;
                            ygradient += ydir[i + 1][j + 1] * pixel;
                        }
                    }
                    // WRITE to actual target pixel
                    rdata[x + y * _x] = std::sqrt(xgradient * xgradient + ygradient * ygradient);
                }
            }
        }
        return ret;
    }

    virtual std::shared_ptr<Image> edges(float low, float high) override
    {
        return edges(low, high, 2);
    }

    virtual std::shared_ptr<ParallelImage> edges(float low, float high, int mode)
    {
        validate();
        if (_type != floatgrayscale)
        {
            throw ImageException("Currently can only gradiant on floating point grayscale");
        }
        auto ret = std::make_shared<ParallelImage>(_x, _y, _type);
        if (mode == 0)
        {
            for (unsigned int x = 0; x < _x; ++x)
            {
                for (unsigned int y = 0; y < _y; ++y)
                {
                    bool nearstrong = false;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            auto pixel = fppixel((unsigned int)xcoord,
                                               (unsigned int)ycoord);
                            if (pixel > high)
                                nearstrong = true;
                        }
                    }
                    ret->fppixel(x, y) = fppixel(x, y) > high ? 1 : ((fppixel(x, y) > low && nearstrong) ? 1 : 0);
                }
            }
        }
        if (mode == 1)
        {   // TODO implement NAIVE OpenMP parallel version
#pragma omp parallel for 
            for (unsigned int y = 0; y < _y; ++y)
            {
                for (unsigned int x = 0; x < _x; ++x)
                {
                    bool nearstrong = false;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            auto pixel = fppixel((unsigned int)xcoord,
                                               (unsigned int)ycoord);
                            if (pixel > high)
                                nearstrong = true;
                        }
                    }
                    ret->fppixel(x, y) = fppixel(x, y) > high ? 1 : ((fppixel(x, y) > low && nearstrong) ? 1 : 0);
                }
            }
        }
        if (mode == 2)
        {
            // TODO implement DIRECT ACCESS OpenMP parallel version
            float* srcdata = (float*)_data;
            float* rdata = (float*)ret->data();
#pragma omp parallel for 
            for (unsigned int y = 0; y < _y; ++y)
            {
                for (unsigned int x = 0; x < _x; ++x)
                {
                    bool nearstrong = false;
                    for (int i = -1; i < 2; ++i)
                    {
                        for (int j = -1; j < 2; ++j)
                        {
                            int xcoord = (int)x + i;
                            int ycoord = (int)y + j;
                            if (xcoord < 0)
                                xcoord = 0;
                            if (ycoord < 0)
                                ycoord = 0;
                            if (xcoord >= (int)_x)
                                xcoord = _x - 1;
                            if (ycoord >= (int)_y)
                                ycoord = _y - 1;
                            // READ from neighboring pixels
                            float pixel = srcdata[xcoord + ycoord * _x];
                            if (pixel > high)
                                nearstrong = true;
                        }
                    }
                    // WRITE to actual target pixel
                    float current_pixel = srcdata[x + y * _x];
                    rdata[x + y * _x] = current_pixel > high ? 1 : ((current_pixel > low && nearstrong) ? 1 : 0);
                }
            }
        }
        return ret;
    }

    bool different_format()
    {
        return false;
    }

    // A function that allows you to convert back to the host
    // image format.  ONLY needed if you restructure the image
    // in the constructor, and only called if different_format is true
    std::shared_ptr<Image> to_host();
};

#endif
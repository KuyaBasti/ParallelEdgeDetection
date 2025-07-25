// Do not modify this file unless you LIKE merge conflicts...

#include "image.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include <ranges>
#include <cstring>
#include <iostream>
#include <cmath>

Image::Image(const std::string file)
{
    int x, y, n;
    // Damn C library, x and y only make sense as unsigned...
    // So this eliminates a type warning this way.
    _data = (void *)stbi_load(file.c_str(), &x, &y, &n, 0);
    if (!_data)
    {
        std::string msg = "Failed to load, Error: ";
        msg += stbi_failure_reason();
        throw ImageException(msg);
    }
    _isstb = true;
    _type = (ImageType)n;
    _x = (unsigned int)x;
    _y = (unsigned int)y;
}

// This assumes that we have no pitch beyond the
// default because its the built in library.
void Image::write_png(const std::string &filename)
{
    if (_type == floatgrayscale)
    {
        convert(grayscale)->write_png(filename);
        return;
    }
    if (_type > 4)
        throw ImageException("Unable to write floating point data");
    if (!_type || !_data)
        throw ImageException("Unable to write invalid image");
    auto ret = stbi_write_png(filename.c_str(),
                              _x, _y, (int)_type, (unsigned char *)_data, _x * ((unsigned int)_type));
    if (!ret)
    {
        std::string msg = "Failed to write, Error: ";
        msg += stbi_failure_reason();
        throw ImageException(msg);
    }
}

std::shared_ptr<Image> Image::shrink(uint factor)
{

    validate();
    if (_x <= factor || _y <= factor)
        throw ImageException("Can't be shrunk that much!");

    auto ret = std::make_shared<Image>(_x / factor, _y / factor, _type);
    if (_type < 5)
    {
        for (unsigned int x = 0; x < ret->_x; ++x)
        {
            for (unsigned int y = 0; y < ret->_y; ++y)
            {
                for (unsigned int i = 0; i < (unsigned int)_type; ++i)
                {
                    ret->pixel(x, y, i) = pixel(x * factor, y * factor, i);
                }
            }
        }
    }
    else
    {
        throw ImageException("Not implemented (yet)");
    }
    return ret;
}

// Grabbed this online, its the 5x5 gaussian parameter set.
// You can cut & paste this elsewhere if you want...
const float gaussian[5][5] = {
    {0.0002f, 0.0033f, 0.0081f, 0.0033f, 0.0002f},
    {0.0033f, 0.0479f, 0.1164f, 0.0479f, 0.0033f},
    {0.0081f, 0.1164f, 0.2831f, 0.1164f, 0.0081f},
    {0.0033f, 0.0479f, 0.1164f, 0.0479f, 0.0033f},
    {0.0002f, 0.0033f, 0.0081f, 0.0033f, 0.0002f}};


std::shared_ptr<Image> Image::blur()
{
    validate();
    if (_type != floatgrayscale)
    {
        throw ImageException("Currently can only blur on floating point grayscale");
    }
    auto ret = std::make_shared<Image>(_x, _y, _type);
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
    return ret;
}

const float ydir[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
const float xdir[3][3] = {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}};

std::shared_ptr<Image> Image::gradient()
{
    validate();
    if (_type != floatgrayscale)
    {
        throw ImageException("Currently can only gradiant on floating point grayscale");
    }
    auto ret = std::make_shared<Image>(_x, _y, _type);
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
    return ret;
}

std::shared_ptr<Image> Image::edges(float low, float high)
{
    validate();
    if (_type != floatgrayscale)
    {
        throw ImageException("Currently can only gradiant on floating point grayscale");
    }
    auto ret = std::make_shared<Image>(_x, _y, _type);
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
    return ret;
}

std::shared_ptr<Image> Image::convert(ImageType to)
{
    validate();
    auto ret = std::make_shared<Image>(_x, _y, to);
    if (_type == to)
    {
        memcpy(ret->_data, _data, pixel_size() * pixel_count());
        return ret;
    }
    if (_type == rgb && to == floatgrayscale)
    {
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
    if (_type == floatgrayscale && to == grayscale)
    {
        for (unsigned int x = 0; x < _x; ++x)
        {
            for (unsigned int y = 0; y < _y; ++y)
            {
                ret->pixel(x, y) = (unsigned char)(fppixel(x, y) * 255);
            }
        }
        return ret;
    }
    if (_type == grayscale && to == floatgrayscale)
    {
        for (unsigned int x = 0; x < _x; ++x)
        {
            for (unsigned int y = 0; y < _y; ++y)
            {
                ret->fppixel(x, y) = ((float)pixel(x, y)) / 255;
            }
        }
        return ret;
    }
    throw ImageException("Not Implemented (yet)");
    return nullptr;
}

// Cleans up the floating point range so that it is
// guarenteed to be 0-1, allowing some minor
// floating point errors and thus making sure they don't
// slip through.
void Image::clean()
{
    if (_type < 4)
        return;

    for (unsigned int x = 0; x < _x; ++x)
    {
        for (unsigned int y = 0; y < _y; ++y)
        {
            fppixel(x, y) = (fppixel(x, y) < 0) ? 0 : ((fppixel(x, y) > 1) ? 1 : fppixel(x, y));
        }
    }
}
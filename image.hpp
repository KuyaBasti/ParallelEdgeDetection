// Do not modify this file unless you LIKE merge conflicts...

#ifndef _IMAGE_HPP
#define _IMAGE_HPP

#include <exception>
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"
#include <memory>
#include <cassert>
#include <cstring>
#include <iostream>

// You can define _IMAGE_DEBUG when
// including this file to
// turn on additional error checking
// for inlined functions for accessing
// pixels

#ifdef _IMAGE_DEBUG
#define IMAGE_ASSERT(X) assert(x)
#else
#define IMAGE_ASSERT(X)
#endif

class ImageException : public std::exception
{
    std::string msg;

public:
    ImageException() : msg("Image Exception Error") {}

    ImageException(const std::string &in) : msg(in) {}

    // Override what because damnit, I prefer
    // std::string and think c_str shit needs to die
    virtual const char *what() const noexcept
    {
        return msg.c_str();
    }
};

// For types less that 4 it is assumed
// to be 8b per channel and than many channels
// For floating versions the pixels are 32b floating point
// rather than fixed.

// Currently only a subset of types are supported:
// grayscale, rgb, and floating-point grayscale but
// keeping things flexible/generic to enable supporting
// other types.
enum ImageType
{
    grayscale = 1,
    rgb = 3,
    rgba = 4,
    floatgrayscale = 5,
    floatrgb = 7,
    floatrgba = 8,
    invalidimage = 0
};

// We will subclass this to create parallel versions to do the
// various transformations.  So this way also we can keep correctness checks the same
// (just with a different subtype)

// The destructor for subclasses MAY need to change the allocator and thus the
// destructor.  Since there is a null-ptr check on the destructor, a subclass's
// destructor should just make sure _data is cleared if its a custom allocator
// (such as for the CUDA version).
class Image : public std::enable_shared_from_this<Image>
{
public:
    // This creator loads from a .png file
    Image(const std::string file);

    // And this one just creates an empty image.
    Image() : _x(0), _y(0), _type(invalidimage), _data(nullptr), _isstb(false)
    {
    }

    // Creates an image of the specified type and size with empty data
    Image(unsigned int x, unsigned int y, ImageType type) : _x(x), _y(y), _type(type), _data(nullptr), _isstb(false)
    {
        alloc_data();
    }

    // If you override the allocator and your deallocator
    // needs to free memory, have it just set _data to null
    // so that it is not touched here.
    inline ~Image()
    {
        if (!_type)
            return;
        if (_isstb)
        {
            if (_data)
            {
                stbi_image_free(_data);
            }
        }
        else if (_data)
        {
            free(_data);
        }
    }

    // Have a copy constructor...
    Image(const Image &cp)
    {
        _x = cp._x;
        _y = cp._y;
        _type = cp._type;
        _isstb = false;
        _data = malloc(pixel_size() * pixel_count());
        memcpy(_data, cp._data, pixel_count() * pixel_size());
    }

    // And a far more efficient move constructor...
    inline Image(Image &&mv)
    {
        _data = mv._data;
        _x = mv._x;
        _y = mv._y;
        _type = mv._type;
        _isstb = mv._isstb;
        mv._data = nullptr;
        mv._x = 0;
        mv._y = 0;
        mv._type = invalidimage;
    }

    // A very useful utility: write the file out as a png,
    // so one can look at the results and visually verify that
    // things make sense.  Its virtual so feel free to
    // override it diretly in your subclasses.
    virtual void write_png(const std::string &filename);

    // We have these functions all act/return a shared_ptr
    // so we can specialize/override/etc but still get
    // good memory behavior of everything getting freed when done.

    // It also means we have a natural pipeline where we
    // can have intermediaries freed automagically by
    // stacking things together:
    // eg im->shrink(32)->convert(floatgrayscale)->blur()
    // will end up having the intermediate smart pointers
    // freed prompty, but by doing a test flow where the pointers
    // hang around we can make sure to eliminate cache effects in
    // timing.
    virtual std::shared_ptr<Image> shrink(uint factor);
    virtual std::shared_ptr<Image> convert(ImageType to);
    virtual std::shared_ptr<Image> blur();
    virtual std::shared_ptr<Image> gradient();
    // Parameters are "weak edge" and "strong edge"
    virtual std::shared_ptr<Image> edges(float low, float high);
    // 
    virtual std::shared_ptr<Image> to_host() { return shared_from_this(); }

    // This, ONLY for floating point representations,
    // makes sure the range is explicitly between 0 and 1.
    // Useful because this allows us to have things truncate
    // out of range without a problem in intermediate results.
    // It doesn't end up needing to be used in the normal pipeline
    // but its here to enable intermediate-result-debugging by
    // outputing positions.
    virtual void clean();

    // Ye generic getter functions.
    unsigned int x() const { return _x; }
    unsigned int y() const { return _y; }
    ImageType type() const { return _type; }
    void *data() const { return _data; }

    // A very useful utility, asserting that
    // everthing is valid and if not throwing an exception
    void validate()
    {
        if (!_data || !_x || !_y || !_type)
            throw ImageException("Invalid Image");
    }

    size_t pixel_size() const noexcept
    {
        if (_type < 5)
            return (size_t)_type;
        return (((size_t)_type) - 4) * sizeof(float);
    }

    size_t pixel_count() const noexcept
    {
        return _x * _y;
    }

    // Need to have type, x, y set correct first,
    // and can't  be called if there is already data.
    inline void alloc_data() noexcept
    {
        assert(_x && _y && _type);
        assert(!_data && !_isstb);
        _data = malloc(pixel_size() * pixel_count());
    }

    // gets the pixel for bit representation.  Its bounds checknig
    // is optional and disabled by default.
    inline unsigned char &pixel(unsigned int x, unsigned int y, unsigned channel = 0) noexcept
    {
        IMAGE_ASSERT(channel < _type && _type < 5);
        IMAGE_ASSERT(x < _x && x >= 0 && y < _y && y >= 0);
        return ((unsigned char *)_data)[x * _type + y * _x * _type + channel];
    }

    // And for floating point representation
    inline float &fppixel(unsigned int x, unsigned int y, unsigned channel = 0) noexcept
    {
        IMAGE_ASSERT(channel < (_type - 4) && _type >= 5);
        IMAGE_ASSERT(x < _x && x >= 0 && y < _y && y >= 0);
        return ((float *)_data)[x * ((unsigned int)_type - 4) +
                                y * _x * ((unsigned int)_type - 4) + channel];
    }

    // This will be updated to make a version that is a BIT more
    // forgiving of comparisons, but for now it is
    // just a pure exact.
    bool operator==(Image &ref)
    {
        if (ref._x != _x || ref._y != _y || ref._type != _type)
            return false;

        for (unsigned int x = 0; x < _x; ++x)
        {
            for (unsigned int y = 0; y < _y; ++y)
            {
                if (pixel(x, y) != ref.pixel(x, y))
                    return false;
            }
        }
        return true;
    }

    // A very useful utility: FIND out where you are screwing up...
    void printdiff(Image &ref, int limit = 100)
    {
        int count = 0;
        if (ref._x != _x || ref._y != _y || ref._type != _type)
        {
            std::cout << "Uncomparable images" << "\n";
            return;
        }
        for (unsigned int x = 0; x < _x; ++x)
        {
            for (unsigned int y = 0; y < _y; ++y)
            {
                if (pixel(x, y) != ref.pixel(x, y))
                {
                    std::cout << "Disagreement: {" << x << "," << y << "}(" <<  (int) pixel(x, y) << "," << (int) ref.pixel(x, y) << ")\n";
                    if (count++ > limit)
                    {
                        std::cout << "Exceeding limit, exiting\n";
                        return;
                    }
                }
            }
        }
    }

    protected:
        unsigned int _x;
        unsigned int _y;
        ImageType _type;

        // Data is either created through malloc or
        // through stb library, the destructor acts as
        // appropriate...
    public:
        void *_data;

    protected:
        bool _isstb;
    };

#endif
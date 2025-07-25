#ifndef _CUDA_IMAGE
#define _CUDA_IMAGE

#include "image.hpp"
#include <memory>

class CudaImage : public Image {
    public:
    // We have a constructor here that will copy the image to 
    // CUDA image space.
    CudaImage(Image &from);     // -> CPU to GPU copy
    
    // Regular constructor
    CudaImage(unsigned int x, unsigned int y, ImageType type);  // -> create new empty GPU image
    
    // Destructor to free CUDA memory
    ~CudaImage();

    // A function that allows you to convert back to the host
    // image type.
    std::shared_ptr<Image> to_host();

    // override virtual functions from base class
    std::shared_ptr<Image> convert(ImageType to) override { return convert(to, 0); }
    std::shared_ptr<Image> blur() override { return blur(0); }
    std::shared_ptr<Image> gradient() override { return gradient(0); }
    std::shared_ptr<Image> edges(float low, float high) override { return edges(low, high, 0); }

    // CUDA implementations with mode parameter
    virtual std::shared_ptr<Image> convert(ImageType to, int mode);
    virtual std::shared_ptr<Image> blur(int mode);
    virtual std::shared_ptr<Image> gradient(int mode);
    virtual std::shared_ptr<Image> edges(float low, float high, int mode);
};

#endif
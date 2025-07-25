#include "cuda_image.hpp"

CudaImage::CudaImage(Image &from)
{
    // from parallel_image.hpp
    _x = from.x();
    _y = from.y();
    _type = from.type();
    _isstb = false;

    // Allocate GPU memory
    cudaMalloc(&_data, pixel_size() * pixel_count());

    // Copy data to GPU 
    cudaMemcpy(_data, from.data(), pixel_size() * pixel_count(), cudaMemcpyHostToDevice);
}

CudaImage::CudaImage(unsigned int x, unsigned int y, ImageType type) {
    _x = x;
    _y = y;
    _type = type;
    _isstb = false;
    
    // Allocate GPU memory
    cudaMalloc(&_data, pixel_size() * pixel_count());

    // Set memory to 0
    cudaMemset(_data, 0, pixel_size() * pixel_count());
}

CudaImage::~CudaImage()
{
    if (_data) {
        cudaFree(_data);
        _data = nullptr;
    }
}

std::shared_ptr<Image> CudaImage::to_host()
{
    validate();  
    auto new_image = std::make_shared<Image>(_x, _y, _type);
    cudaMemcpy(new_image->data(), _data, pixel_size() * pixel_count(), cudaMemcpyDeviceToHost);
    return new_image;
}

                                    // RGB DATA             GRAYSCALE     WIDTH   HEIGHT
__global__ void convertRGBtoGRAYSCALE(unsigned char* srcdata, float* rdata, int x, int y)
{
    // DISCUSSION RICO
    // 1. Calculate thread ID
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Check if thread is in bounds
    if (tidx < x && tidy < y)
    {
        // 3. Calculate 1D index
        int i = tidx + tidy * x;
        
        // 4. Do the work - convert RGB to grayscale
        rdata[i] = ((float)srcdata[i * 3] +
                    (float)srcdata[i * 3 + 1] +
                    (float)srcdata[i * 3 + 2]) /
                    (255.0f * 3.0f);
    }
}

__global__ void convertFLOATINGGRAYSCALEtoGRAYSCALE(float* srcdata, unsigned char* rdata, int x, int y)
{   // DISCUSSION RICO
    // 1. Calculate thread ID
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Check if thread is in bounds
    if (tidx < x && tidy < y)
    {
        // 3. Calculate 1D index
        int i = tidx + tidy * x;
        
        // 4. Do the work - convert floatingpioint grayscal to grayscale
        rdata[i] = (unsigned char)(srcdata[i] * 255.0f);
    }
}

std::shared_ptr<Image> CudaImage::convert(ImageType to, int mode)
{
    // which conversion we are doing?
    if (mode == 0) {
        // RGB -> floatgrayscale
        if (to == floatgrayscale && _type == rgb) {
            // RGB to GRASCALE conversion
            // 1. Create new CudaImage for result
            // 2. Set up CUDA grid/block dimensions
            // 3. Launch RGB to grayscale kernel
            // 4. Return result
            auto ret = std::make_shared<CudaImage>(_x, _y, to);

            dim3 gridDim(_x/32, _y/32,1);
            dim3 blockDim(32, 32, 1);

            convertRGBtoGRAYSCALE<<<gridDim, blockDim>>>((unsigned char*)_data, (float*)ret->_data, _x, _y);
            
            return ret;
        }
        // floatgrayscale -> grayscale 
        else if (to == grayscale && _type == floatgrayscale) {
            // GRAYSCALE to RGB conversion
            // _data is source which is floatgrayscale
            // ret->_data is destination where we want the RGB result
            auto ret = std::make_shared<CudaImage>(_x, _y, to);
            
            dim3 gridDim(_x/32, _y/32,1);
            dim3 blockDim(32, 32, 1);

            convertFLOATINGGRAYSCALEtoGRAYSCALE<<<gridDim, blockDim>>>((float*)_data, (unsigned char*)ret->_data, _x, _y);
            
            return ret;
        }
    }
    return Image::convert(to);
}

__constant__ float gaussian[5][5] = {
    {0.0002f, 0.0033f, 0.0081f, 0.0033f, 0.0002f},
    {0.0033f, 0.0479f, 0.1164f, 0.0479f, 0.0033f},
    {0.0081f, 0.1164f, 0.2831f, 0.1164f, 0.0081f},
    {0.0033f, 0.0479f, 0.1164f, 0.0479f, 0.0033f},
    {0.0002f, 0.0033f, 0.0081f, 0.0033f, 0.0002f}};

__global__ void blur_kernel(float* srcdata, float* rdata, int x, int y)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tidx < x && tidy < y)
    {
        float tmp = 0;
        for (int i = -2; i < 3; ++i)
        {
            for (int j = -2; j < 3; ++j)
            {
                int xcoord = tidx + i;
                int ycoord = tidy + j;
                if (xcoord < 0)
                    xcoord = 0;
                if (ycoord < 0)
                    ycoord = 0;
                if (xcoord >= x)
                    xcoord = x - 1;
                if (ycoord >= y)
                    ycoord = y - 1;
                // this is biggest hurdle for performance need to do direct mem access
                // how to get index of 2D image in 1D array bc its how its stored
                // index = row * width + column
                // but fppixel uses x + y * _x where x=column, y=row
                // _x is the width of image
                // _y is height 
                // READ from neighboring pixels
                // (x, y) is at index = x + y * _x in 1D array memory
                float pixel = srcdata[xcoord + ycoord * x];

                tmp += gaussian[i + 2][j + 2] * pixel;
            }
        }
        // WRITE to actual pixel target
        rdata[tidx + tidy * x] = tmp;
    }
    
}

std::shared_ptr<Image> CudaImage::blur(int mode)
{
    if (mode == 0) {
        validate();
        if (_type != floatgrayscale)
        {
            throw ImageException("Currently can only blur on floating point grayscale");
        }
        
        auto ret = std::make_shared<CudaImage>(_x, _y, _type);

        dim3 gridDim(_x/32, _y/32,1);
        dim3 blockDim(32, 32, 1);

        blur_kernel<<<gridDim, blockDim>>>((float*)_data, (float*)ret->_data, _x, _y);
        
        return ret;
    }
    return Image::blur();
}

__constant__ float ydir[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
__constant__ float xdir[3][3] = {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}};

__global__ void gradient_kernel(float* srcdata, float* rdata, int x, int y)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx < x && tidy < y)
    {
        float xgradient = 0;
        float ygradient = 0;
        for (int i = -1; i < 2; ++i)
        {
            for (int j = -1; j < 2; ++j)
            {
                int xcoord = tidx + i;
                int ycoord = tidy + j;
                if (xcoord < 0)
                    xcoord = 0;
                if (ycoord < 0)
                    ycoord = 0;
                if (xcoord >= x)
                    xcoord = x - 1;
                if (ycoord >= y)
                    ycoord = y - 1;
                // READ from neighboring pixels
                float pixel = srcdata[xcoord + ycoord * x];
                xgradient += xdir[i + 1][j + 1] * pixel;
                ygradient += ydir[i + 1][j + 1] * pixel;
            }
        }
        // WRITE to actual target pixel
       rdata[tidx + tidy * x] = sqrtf(xgradient * xgradient + ygradient * ygradient);
    }
    
}

std::shared_ptr<Image> CudaImage::gradient(int mode)
{
    if (mode == 0) {
        validate();
        if (_type != floatgrayscale)
        {
            throw ImageException("Currently can only gradiant on floating point grayscale");
        }
        
        auto ret = std::make_shared<CudaImage>(_x, _y, _type);

        dim3 gridDim(_x/32, _y/32,1);
        dim3 blockDim(32, 32, 1);

        gradient_kernel<<<gridDim, blockDim>>>((float*)_data, (float*)ret->_data, _x, _y);
       
        return ret;
    }
    return Image::gradient();
}

__global__ void edges_kernel(float* srcdata, float* rdata, int x, int y, float low, float high)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx < x && tidy < y)
    {
        bool nearstrong = false;
        for (int i = -1; i < 2; ++i)
        {
            for (int j = -1; j < 2; ++j)
            {
                int xcoord = tidx + i;
                int ycoord = tidy + j;
                if (xcoord < 0)
                    xcoord = 0;
                if (ycoord < 0)
                    ycoord = 0;
                if (xcoord >= x)
                    xcoord = x - 1;
                if (ycoord >= y)
                    ycoord = y - 1;
                // READ from neighboring pixels
                float pixel = srcdata[xcoord + ycoord * x];
                if (pixel > high)
                    nearstrong = true;
            }
        }
        // WRITE to actual target pixel
        float current_pixel = srcdata[tidx + tidy * x];
        rdata[tidx + tidy * x] = current_pixel > high ? 1 : ((current_pixel > low && nearstrong) ? 1 : 0);

        
    }
}

std::shared_ptr<Image> CudaImage::edges(float low, float high, int mode)
{
    if (mode == 0) {
        validate();
        if (_type != floatgrayscale)
        {
            throw ImageException("Currently can only gradiant on floating point grayscale");
        }
        auto ret = std::make_shared<CudaImage>(_x, _y, _type);

        dim3 gridDim(_x/32, _y/32,1);
        dim3 blockDim(32, 32, 1);

        edges_kernel<<<gridDim, blockDim>>>((float*)_data, (float*)ret->_data, _x, _y, low, high);
       
        return ret;
    }
    return Image::edges(low, high);
}
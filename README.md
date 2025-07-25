# Parallel Edge Detection

A high-performance image processing pipeline implementing edge detection algorithms with multiple parallelization strategies. This project demonstrates various parallel computing approaches including OpenMP, Intel SIMD intrinsics, and CUDA GPU acceleration.

## 🚀 Features

- **Sequential Implementation**: Baseline edge detection using traditional CPU processing
- **OpenMP Parallelization**: Multi-threaded CPU implementation with Intel SIMD intrinsics
- **CUDA GPU Acceleration**: High-performance GPU implementation for maximum throughput
- **Multiple Image Formats**: Support for common image formats via STB library
- **Comprehensive Testing**: Google Test framework with extensive test coverage
- **Cross-platform Build**: CMake-based build system

## 🏗️ Architecture

The project implements a class hierarchy with three main implementations:

```
Image (Base Class)
├── ParallelImage (OpenMP + Intel Intrinsics)
└── CudaImage (CUDA GPU)
```

### Core Pipeline

1. **Image Loading**: STB library for reading various image formats
2. **Gaussian Blur**: 5x5 Gaussian kernel for noise reduction
3. **Gradient Calculation**: Sobel operators for edge detection
4. **Edge Extraction**: Threshold-based edge extraction with hysteresis

## 📋 Prerequisites

### System Requirements
- **CPU**: Intel processor with AVX support (for SIMD intrinsics)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.5+ (optional)
- **RAM**: Minimum 4GB, recommended 8GB+

### Software Dependencies
- **CMake**: Version 3.10+
- **C++ Compiler**: C++20 compatible (GCC 9+, Clang 10+, MSVC 2019+)
- **CUDA Toolkit**: Version 11.0+ (for GPU acceleration)
- **OpenMP**: For parallel CPU implementation

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install cmake build-essential libomp-dev
# For CUDA (optional)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda
```

#### macOS
```bash
brew install cmake libomp
# CUDA not supported on macOS
```

#### Windows
- Install Visual Studio 2019+ with C++ support
- Install CMake from [cmake.org](https://cmake.org/download/)
- Install CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)

## 🔨 Building

### Quick Start
```bash
git clone https://github.com/KuyaBasti/ParallelEdgeDetection.git
cd ParallelEdgeDetection
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options
```bash
# Debug build (for development)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Disable CUDA (CPU-only)
cmake -DENABLE_CUDA=OFF ..
```

## 🧪 Testing

The project includes comprehensive unit tests using Google Test:

```bash
# Run all tests
make test

# Run specific test binary
./testbinary

# Run tests with verbose output
./testbinary --gtest_verbose=1
```

### Test Coverage
- Image loading and saving functionality
- Sequential edge detection accuracy
- OpenMP parallel implementation correctness
- CUDA GPU implementation validation
- Performance benchmarking

## 📊 Usage

### Command Line Interface
```bash
# Basic edge detection
./edgedetect input.png output.png

# With custom thresholds
./edgedetect input.png output.png --low 0.1 --high 0.3

# Specify implementation
./edgedetect input.png output.png --mode cuda    # GPU acceleration
./edgedetect input.png output.png --mode openmp  # CPU parallel
./edgedetect input.png output.png --mode seq     # Sequential
```

### Programmatic API

```cpp
#include "parallel_image.hpp"
#include "cuda_image.hpp"

// Load image
auto img = std::make_shared<Image>("input.png");

// OpenMP implementation
ParallelImage parallel_img(*img);
auto edges_omp = parallel_img.edges(0.1f, 0.3f);

// CUDA implementation  
CudaImage cuda_img(*img);
auto edges_cuda = cuda_img.edges(0.1f, 0.3f);

// Save results
edges_omp->save("output_omp.png");
edges_cuda->to_host()->save("output_cuda.png");
```

## ⚡ Performance

### Benchmark Results
*(Results on Intel i7-9750H, RTX 2060, 2048x2048 image)*

| Implementation | Processing Time | Speedup | Memory Usage |
|----------------|----------------|---------|--------------|
| Sequential     | 1.2s          | 1.0x    | 32MB        |
| OpenMP (6 cores) | 0.23s       | 5.2x    | 35MB        |
| CUDA GPU       | 0.08s         | 15.0x   | 48MB        |

### Optimization Features
- **SIMD Vectorization**: Intel AVX instructions for 8x float processing
- **Memory Coalescing**: Optimized memory access patterns for GPU
- **Thread Load Balancing**: Dynamic OpenMP scheduling
- **Zero-copy Operations**: Minimize CPU-GPU memory transfers

## 📁 Project Structure

```
├── README.md                 # This file
├── CMakeLists.txt           # Build configuration
├── main.cpp                 # Command-line interface
├── image.{hpp,cpp}          # Base image class (sequential)
├── parallel_image.{hpp,cpp} # OpenMP + SIMD implementation
├── cuda_image.{hpp,cu}      # CUDA GPU implementation
├── parallel_utils.{hpp,cpp} # Utility functions
├── stb_*.h                  # STB image library
├── stb_instantiation.cpp    # STB library compilation unit
├── *_tests.cpp              # Test files
└── testfiles/               # Sample images
    ├── Lena_2048.png        # Standard test image
    └── shrunk.png           # Smaller test image
```

## 🔬 Technical Details

### Edge Detection Algorithm
1. **Gaussian Blur** (5x5 kernel):
   ```
   [0.0002, 0.0033, 0.0081, 0.0033, 0.0002]
   [0.0033, 0.0479, 0.1164, 0.0479, 0.0033]
   [0.0081, 0.1164, 0.2831, 0.1164, 0.0081]
   [0.0033, 0.0479, 0.1164, 0.0479, 0.0033]
   [0.0002, 0.0033, 0.0081, 0.0033, 0.0002]
   ```

2. **Sobel Gradient Calculation**:
   - X-direction: `[[-1,0,1], [-2,0,2], [-1,0,1]]`
   - Y-direction: `[[-1,-2,-1], [0,0,0], [1,2,1]]`

3. **Hysteresis Thresholding**: Dual-threshold edge linking

### CUDA Implementation Details
- **Grid Configuration**: 2D blocks optimized for memory coalescing
- **Shared Memory**: Tile-based convolution for cache efficiency
- **Texture Memory**: For read-only image data access
- **Compute Capability**: Targets 7.5+ (RTX 20xx series and newer)

### OpenMP Features
- **Parallel Regions**: Multi-threaded image processing
- **SIMD Intrinsics**: AVX/AVX2 for vectorized operations
- **Load Balancing**: Dynamic scheduling for irregular workloads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow C++20 modern practices
- Add unit tests for new features
- Maintain backward compatibility
- Document public APIs
- Use clang-format for code formatting

## 🙏 Acknowledgments

- **STB Library**: [Sean Barrett's STB](https://github.com/nothings/stb) for image I/O
- **Test Image**: [Ethically sourced Lena](https://mortenhannemose.github.io/lena/) by Morten Rieger Hannemose
- **Google Test**: Testing framework by Google

## 📚 References

```bibtex
@misc{hannemoselena,
    author = {Morten Rieger Hannemose},
    title = {Recreated Lena Picture},
    year = {2019},
    url = {https://mortenhannemose.github.io/lena/}
}
```

---

*High-performance parallel computing for computer vision applications*
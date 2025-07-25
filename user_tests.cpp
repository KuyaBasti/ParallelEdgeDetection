#include <gtest/gtest.h> 
#include "cuda_image.hpp"
#include "image.hpp"
#include "parallel_image.hpp"
#include "parallel_utils.hpp"
#include <chrono>

TEST(TestCudaImage, TestEach)
{
    Image lena("../testfiles/Lena_2048.png");

    std::shared_ptr<Image> test_each;
    auto t00 = GetTiming([&]()
        { test_each = std::make_shared<CudaImage>(lena); });
    auto t01 = GetTiming([&]()
        { test_each = test_each->convert(floatgrayscale); });
    auto t02 = GetTiming([&]()
        { test_each = test_each->blur(); });
    auto t03 = GetTiming([&]()
        { test_each = test_each->gradient(); });
    auto t04 = GetTiming([&]()
        { test_each = test_each->edges(0.3f, 0.7f); });
    auto t05 = GetTiming([&]()
        { test_each = test_each->convert(grayscale); });
    auto last = std::dynamic_pointer_cast<CudaImage>(test_each);
    auto t06 = GetTiming([&]()
        { test_each = last->to_host(); });

    std::cout << "Part 0 (copy): " << t00 << "\n";
    std::cout << "Part 1 (convert floatgrayscale): " << t01 << "\n";
    std::cout << "Part 2 (blur): " << t02 << "\n";
    std::cout << "Part 3 (gradient): " << t03 << "\n";
    std::cout << "Part 4 (edges): " << t04 << "\n";
    std::cout << "Part 5 (convert grayscale): " << t05 << "\n";
    std::cout << "Part 6 (to host): " << t06 << "\n";
    std::cout << "Total: " << t00 + t01 + t02 + t03 + t04 + t05 + t06 << "\n";


    auto reference = lena.convert(floatgrayscale)->blur()->gradient()->edges(.3f, .7f)->convert(grayscale);
    EXPECT_TRUE(*reference == *test_each);
}
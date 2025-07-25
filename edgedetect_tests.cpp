// This test file may be updated, so do not modify unless you love merge conflicts!

#include <gtest/gtest.h>

#include "image.hpp"
#include "parallel_image.hpp"
#include <omp.h>
#include "parallel_utils.hpp"

int bogus;

void init_openmp()
{
#pragma omp parallel
    {
        bogus = omp_get_num_threads();
    }
}

// Demonstrate some basic assertions.
TEST(ImageTest, BasicTiming)
{
    // Expect two strings not to be equal.
    EXPECT_THROW(Image("badloc"), ImageException);
    auto lena = std::make_shared<Image>("../testfiles/Lena_2048.png");
    std::cout << "X: " << lena->x() << " Y: " << lena->y() << "\n";
    // Checks that self conversion works
    auto ret = lena->convert(lena->type())->shrink(16);
    ret->write_png("./shrunk.png");
    ret = ret->convert(floatgrayscale);
    ret = ret->convert(grayscale);
    ret->write_png("./grayscale.png");
    lena->write_png("./lena_writetest.png");
}

TEST(ImageTest, TestShrink)
{
    Image lena("../testfiles/Lena_2048.png");
    Image refshrunk("../testfiles/shrunk.png");
    // Checks that self conversion works
    auto ret = lena.shrink(16);
    ret->write_png("./shrunk.png");
    Image shrunk("./shrunk.png");
    EXPECT_TRUE(shrunk == refshrunk);
    shrunk.pixel(32, 32) = shrunk.pixel(32, 32) - 5;
    EXPECT_FALSE(shrunk == refshrunk);
}

TEST(ImageTest, TestConvert)
{
    Image grays(128, 128, grayscale);
    for (unsigned x = 0; x < grays.x(); ++x)
    {
        for (unsigned y = 0; y < grays.y(); ++y)
        {
            grays.pixel(x, y) = (unsigned char)(255 - (x + y));
        }
        grays.pixel(0, 0) = 0;
        grays.pixel(0, grays.y() - 1) = 255;
        grays.pixel(grays.x() - 1, 0) = 255;
    }
    grays.write_png("./grayscale-test.png");
    auto converted = grays.convert(floatgrayscale);
    auto converted2 = converted->convert(grayscale);
    converted2->write_png("./grascale-converted.png");
    EXPECT_TRUE(grays == *converted2);
}

TEST(ImageTest, TestBlur)
{
    Image lena("../testfiles/Lena_2048.png");
    auto blur = lena.convert(floatgrayscale);
    blur->write_png("./Lena_grayscale.png");
    blur = blur->blur();
    blur->clean();
    blur->write_png("./Lena_blurred.png");
    blur = blur->gradient();
    blur->clean();
    blur->write_png("./Lena_gradient.png");
    blur = blur->edges(.3f, .7f);
    blur->write_png("./Lena_edges.png");
}

TEST(ImageTest, MakeSmall)
{
    Image lena("../testfiles/Lena_2048.png");
    auto smaller = lena.shrink(4);
    smaller->write_png("Lena_smaller.png");
    auto gs = smaller->convert(floatgrayscale);
    gs->write_png("Lena_smaller_grayscale.png");
    auto blur = gs->blur();
    blur->clean();
    blur->write_png("./Lena_smaller_blurred.png");
    blur = blur->gradient();
    blur->clean();
    blur->write_png("./Lena_smaller_gradient.png");
    blur = blur->edges(.3f, .7f);
    blur->write_png("./Lena_smaller_edges.png");
}

TEST(ParallelTest, TimeCopy)
{

    std::shared_ptr<ParallelImage> p0, p1, p2;
    init_openmp();

    Image lena("../testfiles/Lena_2048.png");
    auto t0 = GetTiming([&]()
                        { p0 = make_shared<ParallelImage>(lena, 0); });

    Image lena2("../testfiles/Lena_2048.png");
    auto t1 = GetTiming([&]()
                        { p1 = make_shared<ParallelImage>(lena2, 1); });

    Image lena3("../testfiles/Lena_2048.png");
    auto t2 = GetTiming([&]()
                        { p2 = make_shared<ParallelImage>(lena3, 2); });

    EXPECT_TRUE(lena == *p0);
    EXPECT_TRUE(lena == *p1);
    EXPECT_TRUE(lena == *p2);
    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Mode 2: " << t2 << "\n";
    Image lena5("../testfiles/Lena_2048.png");

    t1 = GetTiming([&]()
                   { p1 = make_shared<ParallelImage>(lena5, 1); });
    Image lena6("../testfiles/Lena_2048.png");

    t0 = GetTiming([&]()
                   { p1 = make_shared<ParallelImage>(lena6, 0); });
    Image lena7("../testfiles/Lena_2048.png");

    t2 = GetTiming([&]()
                   { p1 = make_shared<ParallelImage>(lena7, 2); });
    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Mode 2: " << t2 << "\n";
}

TEST(ParallelTest, TimeConvert)
{
    Image lena("../testfiles/Lena_2048.png");

    auto p0 = make_shared<ParallelImage>(lena);
    auto t0 = GetTiming([&]()
                        { p0 = std::dynamic_pointer_cast<ParallelImage>(p0->convert(floatgrayscale, 0)); });

    auto p1 = make_shared<ParallelImage>(lena);
    auto t1 = GetTiming([&]()
                        { p1 = std::dynamic_pointer_cast<ParallelImage>(p1->convert(floatgrayscale, 1)); });

    auto p2 = make_shared<ParallelImage>(lena);
    auto t2 = GetTiming([&]()
                        { p2 = std::dynamic_pointer_cast<ParallelImage>(p2->convert(floatgrayscale, 2)); });

    auto p3 = make_shared<ParallelImage>(lena);
    auto t3 = GetTiming([&]()
                        { p3 = std::dynamic_pointer_cast<ParallelImage>(p3->convert(floatgrayscale, 3)); });

    auto p4 = make_shared<ParallelImage>(lena);
    auto t4 = GetTiming([&]()
                        { p4 = std::dynamic_pointer_cast<ParallelImage>(p4->convert(floatgrayscale, 4)); });

    auto p5 = make_shared<ParallelImage>(lena);
    auto t5 = GetTiming([&]()
                        { p5 = std::dynamic_pointer_cast<ParallelImage>(p5->convert(floatgrayscale, 5)); });

    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Mode 2: " << t2 << "\n";
    std::cout << "Mode 3: " << t3 << "\n";
    std::cout << "Mode 4: " << t4 << "\n";
    std::cout << "Mode 5: " << t5 << "\n";
    EXPECT_TRUE(*(p0->convert(grayscale)) == *(p1->convert(grayscale)));
    EXPECT_TRUE(*(p0->convert(grayscale)) == *(p2->convert(grayscale)));
    EXPECT_TRUE(*(p0->convert(grayscale)) == *(p3->convert(grayscale)));
    EXPECT_TRUE(*(p0->convert(grayscale)) == *(p4->convert(grayscale)));
    EXPECT_TRUE(*(p0->convert(grayscale)) == *(p5->convert(grayscale)));
}

TEST(ParallelTest, TimeBlur)
{
    Image lena("../testfiles/Lena_2048.png");
    auto converted = make_shared<ParallelImage>(lena);

    auto p0 = std::dynamic_pointer_cast<ParallelImage>(converted->convert(floatgrayscale));
    auto t0 = GetTiming([&]()
                        { p0 = p0->blur(0); });
    auto p1 = std::dynamic_pointer_cast<ParallelImage>(converted->convert(floatgrayscale));
    auto t1 = GetTiming([&]()
                        { p1 = p1->blur(1); });

    auto p2 = std::dynamic_pointer_cast<ParallelImage>(converted->convert(floatgrayscale));
    auto t2 = GetTiming([&]()
                        { p2 = p2->blur(2); });

    auto p3 = std::dynamic_pointer_cast<ParallelImage>(converted->convert(floatgrayscale));
    auto t3 = GetTiming([&]()
                        { p3 = p3->blur(3); });

    auto p4 = std::dynamic_pointer_cast<ParallelImage>(converted->convert(floatgrayscale));
    auto t4 = GetTiming([&]()
                        { p4 = p4->blur(4); });

    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Mode 2: " << t2 << "\n";
    std::cout << "Mode 3: " << t3 << "\n";
    std::cout << "Mode 4: " << t4 << "\n";

    auto reference = lena.convert(floatgrayscale)->blur()->convert(grayscale);
    EXPECT_TRUE(*reference == *(p0->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p1->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p2->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p3->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p4->convert(grayscale)));
    p4->write_png("VectorizedTest.png");
}

TEST(ParallelTest, TimeGradient)
{
    Image lena("../testfiles/Lena_2048.png");
    auto converted = make_shared<ParallelImage>(lena)->convert(floatgrayscale);

    auto p0 = std::dynamic_pointer_cast<ParallelImage>(converted->blur());
    auto t0 = GetTiming([&]()
                        { p0 = p0->gradient(0); });

    auto p1 = std::dynamic_pointer_cast<ParallelImage>(converted->blur());
    auto t1 = GetTiming([&]()
                        { p1 = p1->gradient(1); });

    auto p2 = std::dynamic_pointer_cast<ParallelImage>(converted->blur());
    auto t2 = GetTiming([&]()
                        { p2 = p2->gradient(2); });

    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Mode 2: " << t2 << "\n";

    auto reference = lena.convert(floatgrayscale)->blur()->gradient()->convert(grayscale);
    EXPECT_TRUE(*reference == *(p0->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p1->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p2->convert(grayscale)));
}

TEST(ParallelTest, TimeEdge)
{
    Image lena("../testfiles/Lena_2048.png");
    auto converted = make_shared<ParallelImage>(lena)->convert(floatgrayscale)->blur();

    auto p0 = std::dynamic_pointer_cast<ParallelImage>(converted->gradient());
    auto t0 = GetTiming([&]()
                        { p0 = p0->edges(0.3f, 0.7f, 0); });

    auto p1 = std::dynamic_pointer_cast<ParallelImage>(converted->gradient());
    auto t1 = GetTiming([&]()
                        { p1 = p1->edges(0.3f, 0.7f, 1); });

    auto p2 = std::dynamic_pointer_cast<ParallelImage>(converted->gradient());
    auto t2 = GetTiming([&]()
                        { p2 = p2->edges(0.3f, 0.7f, 2); });

    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Mode 2: " << t2 << "\n";

    auto reference = lena.convert(floatgrayscale)->blur()->gradient()->edges(.3f, .7f)->convert(grayscale);
    EXPECT_TRUE(*reference == *(p0->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p1->convert(grayscale)));
    EXPECT_TRUE(*reference == *(p2->convert(grayscale)));
    reference->printdiff(*(p2->convert(grayscale)));
    p2->write_png("EdgesParallel.png");
}

TEST(ParallelTest, FullPerformance)
{
    Image lena("../testfiles/Lena_2048.png");
    std::shared_ptr<Image> p0;
    auto t0 = GetTiming([&]()
                        { p0 = lena.convert(floatgrayscale)->blur()->gradient()->edges(0.3f, 0.7f)->convert(grayscale); });

    std::shared_ptr<Image> p1;
    Image lena2("../testfiles/Lena_2048.png");
    auto t1 = GetTiming([&]()
                        { p1 = std::make_shared<ParallelImage>(lena)->convert(floatgrayscale)->blur()->gradient()->edges(0.3f, 0.7f)->convert(grayscale); });

    std::cout << "Mode 0: " << t0 << "\n";
    std::cout << "Mode 1: " << t1 << "\n";
    std::cout << "Performance improvement: " << (t0 / t1) << "\n";
    EXPECT_TRUE(*p0 == *p1);
#pragma omp parallel 
{
    std::cout << "Number of threads: " << omp_get_num_threads() << "\n";
}
}

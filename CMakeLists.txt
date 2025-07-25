
# Please do not change this file except for the optimization level
# (changing it to -O0 when debugging) unless you like merge conflicts

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(edgedetect CUDA CXX C)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 89)
# This makes it print out usage information on each kenel compiled
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=\"-v \"")

set(DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
add_executable(edgedetect main.cpp stb_instantiation.cpp)

set_target_properties(edgedetect PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set (CMAKE_CXX_STANDARD 20)
# This is "Turn on a lot the hardening stuff.  There is a 
# minor performance impact but hey, better Safe than Sorry"
# Feel free to turn OFF the optimizer for better debugging however.
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g \
-O2 \
-Wall -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough \
-Werror=format-security \
-D_GLIBCXX_ASSERTIONS \
-fopenmp -mavx")

set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra --coverage -D_GLIBCXX_ASSERTIONS")


# This is geting gunit so you don't have to...
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  #DOWNLOAD_EXTRACT_TIMESTAMP true
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()


add_executable(testbinary edgedetect_tests.cpp stb_instantiation.cpp image.cpp parallel_image.cpp cuda_image.cu user_tests.cpp parallel_utils.cpp) 
target_link_libraries(
  testbinary
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(testbinary)

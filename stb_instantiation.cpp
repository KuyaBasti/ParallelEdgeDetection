// There is no need to modify this file, this is so that you have access
// to the STB libraries for image loading/storing.

// The STB libraries do introduce a couple of compiler warnings
// in my preferred paranoid compiler settings, so...
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wconversion"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma GCC diagnostic pop

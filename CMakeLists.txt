cmake_minimum_required(VERSION 3.10)
project(conv2d_library C)

# Dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3f)
find_package(OpenMP REQUIRED)

# Create the static library
add_library(conv2d STATIC src/conv2d.c)
target_include_directories(conv2d PUBLIC include ${FFTW3_INCLUDE_DIRS})
target_link_libraries(conv2d PUBLIC ${FFTW3_LIBRARIES} OpenMP::OpenMP_C m)

# Add a demo executable
add_executable(conv2d_test src/main.c)
target_link_libraries(conv2d_test PRIVATE conv2d)


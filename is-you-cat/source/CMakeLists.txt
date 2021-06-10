# generate project with with `-DCMAKE_PREFIX_PATH=<path to libtorch>`
# build with `--config Release`

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(is-you-cat)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined -O0")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined -O0")

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native") # -funsafe-math-optimizations?
set (CMAKE_LINKER_FLAGS_RELEASE "${CMAKE_LINKER_FLAGS_RELEASE} -O3 -march=native")

add_executable(is-you-cat main.cpp)

target_link_libraries(is-you-cat "${TORCH_LIBRARIES}")
target_link_libraries(is-you-cat "${OpenCV_LIBS}")
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

set_property(TARGET is-you-cat PROPERTY CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic)
endif()


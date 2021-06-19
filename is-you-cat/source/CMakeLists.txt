# generate project with with `-DCMAKE_PREFIX_PATH=<path to libtorch>` and `-DTENSORFLOW_SOURCE_DIR=<path to tensorflow>`
# build with `--config Release`

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(is-you-cat)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT DEFINED TENSORFLOW_SOURCE_DIR)
  message(FATAL_ERROR, "Set TENSORFLOW_SOURCE_DIR variable")
endif()

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined -O0")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined -O0")

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native") # -funsafe-math-optimizations?
set (CMAKE_LINKER_FLAGS_RELEASE "${CMAKE_LINKER_FLAGS_RELEASE} -O3 -march=native")

add_executable(is-you-cat main.cpp ONNXModel.cpp TFModel.cpp TorchModel.cpp CatModelFactory.cpp hsv.cpp rgb.cpp)

target_link_libraries(is-you-cat "${TORCH_LIBRARIES}")
target_link_libraries(is-you-cat "${OpenCV_LIBS}")
target_link_libraries(is-you-cat tensorflow-lite)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

set_property(TARGET is-you-cat PROPERTY CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic)
endif()


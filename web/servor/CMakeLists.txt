cmake_minimum_required(VERSION 3.13)

project("servor" VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_ROOT ${CMAKE_SOURCE_DIR}/external/)


add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ORT_ROOT}/lib/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include
)

add_executable(application "main.cpp")

target_link_libraries(application PRIVATE onnxruntime)

set_target_properties(application PROPERTIES
        BUILD_RPATH ${ORT_ROOT}/lib
)


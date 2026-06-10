cmake_minimum_required(VERSION 3.15)
project(client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(client_bindings
        pybind_module.cpp
        request_client.cpp
)
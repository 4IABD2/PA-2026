cmake_minimum_required (VERSION 3.13)

project ("client" VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Main target: create executable with its source file(s) directly
add_executable(application "main.cpp")

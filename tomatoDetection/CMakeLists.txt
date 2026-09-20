cmake_minimum_required(VERSION 3.10)
project(TomatoDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(TomatoDetection main.cpp)
target_link_libraries(TomatoDetection PRIVATE ${OpenCV_LIBS})
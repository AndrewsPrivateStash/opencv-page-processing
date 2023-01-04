# buillds the makefile for the project

cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(missal_image_proc VERSION 1.0)

find_package(OpenCV REQUIRED)
add_executable(mImgProc main.cpp)

target_include_directories(mImgProc PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_libraries(mImgProc PUBLIC ${OpenCV_LIBRARIES} ${OpenCV_DEPENDENCIES})
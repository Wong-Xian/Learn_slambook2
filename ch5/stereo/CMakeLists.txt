project(stereo)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_FLAGS "-std=c++14")

set(Pangolin_DIR /home/wx/文档/Pangolin/build)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

set(OpenCV_DIR /home/wx/下载/opencv3.4/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
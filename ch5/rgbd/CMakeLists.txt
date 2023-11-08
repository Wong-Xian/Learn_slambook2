find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(Pangolin_DIR /home/wx/文档/Pangolin/build)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

set(OpenCV_DIR /home/wx/下载/opencv3.4/opencv/build)
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
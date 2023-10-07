project(imageBasics)

set(OpenCV_DIR /home/wx/下载/opencv3.4/opencv/build)
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )


add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})

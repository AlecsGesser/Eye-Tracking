cmake_minimum_required(VERSION 2.8)
project( EyeTracking )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( tracker src/findContours.cpp )
target_link_libraries( tracker ${OpenCV_LIBS} )

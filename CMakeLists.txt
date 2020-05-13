cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(OpenCV_DIR /Users/eric/installation/OpenCV-master/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config      : ${OpenCV_DIR}")
message(STATUS "    version     : ${OpenCV_VERSION}")
message(STATUS "    libraries   : ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#Include src directory
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

file(GLOB src_folder
    "src/*.h"
    "src/*.cpp"
)

# Declare the executable target built from your sources
add_executable(stitch ${src_folder}
    main.cpp
)



# Link your application with OpenCV, TBB, OpenMP libraries
target_link_libraries(stitch PRIVATE ${OpenCV_LIBS})

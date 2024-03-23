cmake_minimum_required(VERSION 3.5.1)
project(trt_infer)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 如果找到了OpenCV，打印一些消息
if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()

# 查找CUDA库
find_package(CUDA REQUIRED)

# 如果找到了CUDA，打印一些消息
if(CUDA_FOUND)
    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    libraries: ${CUDA_LIBRARIES}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
endif()

# 设置TensorRT库的路径
set(TENSORRT_ROOT "/home/lifa/software/TensorRT-7.2.3.4")

# 添加OpenCV, TensorRT和CUDA头文件的搜索路径
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_ROOT}/include)

# 添加源代码文件并命名输出的可执行文件
add_executable(trt_infer trt_infer.cpp)

# 链接OpenCV库
target_link_libraries(trt_infer nvinfer ${OpenCV_LIBS} ${CUDA_LIBRARIES})

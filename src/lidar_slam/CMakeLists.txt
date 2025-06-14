cmake_minimum_required(VERSION 3.0.2)
project(lidar_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置构建模式为 Release
set(CMAKE_BUILD_TYPE Release)

# 查找 PCL 库
find_package(PCL 1.10 REQUIRED)
# 查找 OpenCV 库
find_package(OpenCV REQUIRED)
# 查找 catkin 及其组件
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  sensor_msgs
)
# 查找 SuiteSparse 库
find_package(SuiteSparse REQUIRED)
# 查找 g2o 库
find_package(g2o REQUIRED)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidar_slam
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)
# 包含目录
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${SuiteSparse_INCLUDE_DIRS}
  ${g2o_INCLUDE_DIRS}
  /opt/ros/noetic/include/g2o
)

# 声明可执行文件
# add_executable(${PROJECT_NAME}
#                src/main.cpp 
#                src/point_process.cpp)
add_executable(${PROJECT_NAME}
 src/main.cpp src/point_process.cpp src/ImuDataHandler.cpp)

# 显式设置 g2o_DIR
set(g2o_DIR "/opt/ros/noetic/share/g2o/cmake" CACHE PATH "g2o_DIR")

message(STATUS "G2O include directories: ${G2O_INCLUDE_DIRS}")

# 打印状态信息
message(STATUS "G2O include directories: ${G2O_INCLUDE_DIR}")
message(STATUS "SuiteSparse_INCLUDE_DIRS ${SuiteSparse_INCLUDE_DIRS}")

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${SuiteSparse_LIBRARIES}
  ${g2o_LIBRARIES}
  cxsparse
  g2o_csparse_extension  # 添加这个库
  g2o_core
  g2o_stuff
  g2o_solver_csparse
  g2o_types_slam3d  # 如果使用了 slam3d 类型
  # 其他可能需要的 g2o 库...
)
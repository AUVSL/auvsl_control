cmake_minimum_required(VERSION 3.0.2)
project(auvsl_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  geometry_msgs
  message_generation
  )

## Generate services in the 'srv' folder
add_service_files(
   FILES
   AnfisControl.srv
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
 )


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES auvsl_control
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

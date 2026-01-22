cmake_minimum_required(VERSION 3.0.2)
project(solution_prep)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  message_runtime
  lite6_labauto
  roscpp
  rospy
  std_msgs
)

add_service_files(
  FILES
  BeakerMani.srv
  pHMeasure.srv
  PipetteDo.srv
  PipetteDoTwice.srv
  TweezersDraw.srv
  )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  )

catkin_package(
  CATKIN_DEPENDS message_generation message_runtime std_msgs geometry_msgs
  )

###########
## Build ##
###########
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############
catkin_install_python(PROGRAMS
   scripts/beaker_manipulator.py
   scripts/IPC.py
   scripts/orchestrator.py
   scripts/pH_controller.py
   scripts/pH_serial.py
   scripts/pipette_controller.py
   scripts/pipette_twice_controller.py
   scripts/tweezers_controller.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

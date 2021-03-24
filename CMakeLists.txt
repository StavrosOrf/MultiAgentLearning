cmake_minimum_required(VERSION 2.8)
project(test_includes)

#There are lots of scripts with cmake
#for finding external libraries. 
#see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples
#find_package(Boost COMPONENTS system filesystem REQUIRED)
#include_directories( ${Boost_INCLUDE_DIRS} )

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories( ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${yaml-cpp_INCLUDE_DIR}
  include)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-g -Wall -I /usr/local/include/eigen3/")
add_subdirectory(include)

#link_directories(libraries)

set( LIBS_TO_LINK Utilities
                  Learning
                  Domains
                  Agents
                  POMDPs
                  Planning
                  yaml-cpp
                  ${CMAKE_THREAD_LIBS_INIT})# -llapack

add_executable(testNeuralNetEigen testNeuralNetEigen.cpp)
target_link_libraries( testNeuralNetEigen ${LIBS_TO_LINK} )

add_executable(testNeuroEvoEigen testNeuroEvoEigen.cpp)
target_link_libraries( testNeuroEvoEigen ${LIBS_TO_LINK} )

add_executable(testMAPElitesEigen testMAPElitesEigen.cpp)
target_link_libraries( testMAPElitesEigen ${LIBS_TO_LINK} )

add_executable(testPOMDP testPOMDP.cpp)
target_link_libraries( testPOMDP ${LIBS_TO_LINK} )

add_executable(testSearch testSearch.cpp)
target_link_libraries( testSearch ${LIBS_TO_LINK} )

add_executable(testWarehouse testWarehouse.cpp threadpool.cpp)
target_link_libraries( testWarehouse ${LIBS_TO_LINK} )

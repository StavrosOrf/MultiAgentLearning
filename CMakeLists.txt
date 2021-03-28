cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_includes)

#There are lots of scripts with cmake
#for finding external libraries. 
#see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples
#find_package(Boost COMPONENTS system filesystem REQUIRED)
#include_directories( ${Boost_INCLUDE_DIRS} )

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)

if(NOT Torch_FOUND)
	message(STATUS "FUCK MY LIFE")
endif()

include_directories( ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${yaml-cpp_INCLUDE_DIR}
  ${Torch_INCLUDE_DIRS}
  include)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-03")
#set(CMAKE_CXX_FLAGS "-g -Wall -I /usr/local/include/eigen3/")
add_subdirectory(include)

#link_directories(libraries)

set( LIBS_TO_LINK Utilities
                  Learning
                  Domains
                  Agents
                  Planning
                  yaml-cpp
                  ${TORCH_LIBRARIES}
                  ${CMAKE_THREAD_LIBS_INIT})# -llapack

add_executable(testSearch testSearch.cpp)
target_link_libraries( testSearch ${LIBS_TO_LINK})

add_executable(testWarehouse testWarehouse.cpp threadpool.cpp)
target_link_libraries( testWarehouse ${LIBS_TO_LINK})

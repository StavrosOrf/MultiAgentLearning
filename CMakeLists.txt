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
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)

SET( CMAKE_EXPORT_COMPILE_COMMANDS ON )

IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()

include_directories( ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${yaml-cpp_INCLUDE_DIR}
  ${Torch_INCLUDE_DIRS}
  include)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-03")
#set(CMAKE_CXX_FLAGS "-g -Wall -Wextra")
add_subdirectory(include)
add_subdirectory(tests)

#link_directories(libraries)

set( LIBS_TO_LINK
                  Domains
                  Agents
                  Planning
                  yaml-cpp
                  ${TORCH_LIBRARIES}
                  ${CMAKE_THREAD_LIBS_INIT})# -llapack

	  #add_executable(testSearch testSearch.cpp)
	  #target_link_libraries(testSearch ${LIBS_TO_LINK})

	  #add_executable(test_torch test_torch.cpp)
	  #target_link_libraries(test_torch ${LIBS_TO_LINK})

	  #add_executable(test_random test_random.cpp)
	  #target_link_libraries(test_random ${LIBS_TO_LINK})

add_executable(testWarehouse testWarehouse.cpp)
target_link_libraries(testWarehouse ${LIBS_TO_LINK})

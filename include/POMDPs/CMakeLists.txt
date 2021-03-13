set( SRCS POMDP.cpp POMDPEnvironment.cpp POMDPPolicy.cpp)
add_library( POMDPs SHARED ${SRCS} )
target_link_libraries(POMDPs)

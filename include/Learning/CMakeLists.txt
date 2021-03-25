set( SRCS NeuralNet.cpp)
add_library( Learning SHARED ${SRCS} )
target_link_libraries(Learning Utilities)

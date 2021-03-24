set( SRCS NeuralNet.cpp  NeuroEvo.cpp)
add_library( Learning SHARED ${SRCS} )
target_link_libraries(Learning Utilities)

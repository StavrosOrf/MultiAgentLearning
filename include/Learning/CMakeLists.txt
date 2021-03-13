set( SRCS NeuralNet.cpp  NeuroEvo.cpp MAPElites.cpp)
add_library( Learning SHARED ${SRCS} )
target_link_libraries(Learning Utilities)

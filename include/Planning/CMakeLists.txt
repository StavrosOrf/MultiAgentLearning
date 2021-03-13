set( SRCS Search.cpp Graph.cpp Queue.cpp Node.cpp Edge.cpp)
add_library( Planning STATIC ${SRCS} )
target_link_libraries(Planning)

set( SRCS DDPGAgent.cpp)
add_library( Agents STATIC ${SRCS} )
target_link_libraries(Agents Domains Learning)

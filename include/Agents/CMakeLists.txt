set( SRCS DDPGAgent.cpp ESAgent.cpp)
add_library( Agents STATIC ${SRCS} )
target_link_libraries(Agents Domains ${TORCH_LIBRARIES})

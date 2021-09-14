set(SRCS ESAgent.cpp COMAAgent.cpp)
set(SRCS ${SRCS} DDPGAgent.cpp)
add_library( Agents STATIC ${SRCS} )
target_link_libraries(Agents Domains ${TORCH_LIBRARIES})

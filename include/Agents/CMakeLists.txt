set(SRCS ESAgent.cpp DQNAgent.cpp DDPGAgent.cpp)
#set(SRCS ${SRCS} DDPGAgent.cpp)
add_library( Agents STATIC ${SRCS} )
target_link_libraries(Agents Domains ${TORCH_LIBRARIES})

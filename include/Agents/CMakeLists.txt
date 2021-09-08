set(SRCS ESAgent.cpp)
#set(SRCS ${SRCS} DDPGAgent.cpp COMAAgent.cpp)
add_library( Agents STATIC ${SRCS} )
target_link_libraries(Agents Domains ${TORCH_LIBRARIES})

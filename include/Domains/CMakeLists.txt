set(SRCS Warehouse.cpp AGV.cpp Warehouse_DDPG.cpp)
add_library(Domains STATIC ${SRCS})
target_link_libraries(Domains Agents)

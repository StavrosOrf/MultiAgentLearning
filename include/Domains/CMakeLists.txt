set(SRCS Warehouse.cpp AGV.cpp Warehouse_DDPG.cpp Warehouse_ES.cpp Warehouse_ES_container.cpp)
add_library(Domains STATIC ${SRCS})
target_link_libraries(Domains Agents)

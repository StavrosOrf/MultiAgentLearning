set(SRCS Warehouse.cpp AGV.cpp Warehouse_ES.cpp Warehouse_ES_container.cpp Warehouse_COMA.cpp)
#set(SRCS ${SRCS} Warehouse_DDPG.cpp)
add_library(Domains STATIC ${SRCS})
target_link_libraries(Domains Agents)

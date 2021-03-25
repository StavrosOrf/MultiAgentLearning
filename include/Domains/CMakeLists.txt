set( SRCS Warehouse.cpp AGV.cpp WarehouseCentralised.cpp )
add_library( Domains STATIC ${SRCS} )
target_link_libraries(Domains Learning Agents)

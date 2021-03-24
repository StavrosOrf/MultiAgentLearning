set( SRCS Warehouse.cpp AGV.cpp WarehouseLinks.cpp WarehouseLinksTime.cpp WarehouseIntersections.cpp WarehouseIntersectionsTime.cpp WarehouseCentralised.cpp WarehouseCentralisedTime.cpp)
add_library( Domains STATIC ${SRCS} )
target_link_libraries(Domains Learning Agents)

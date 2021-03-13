set( SRCS SingleRover.cpp MAPElitesRover.cpp Target.cpp MultiRover.cpp MultiNightBar.cpp Bar.cpp Warehouse.cpp AGV.cpp WarehouseLinks.cpp WarehouseLinksTime.cpp WarehouseIntersections.cpp WarehouseIntersectionsTime.cpp WarehouseCentralised.cpp WarehouseCentralisedTime.cpp)
add_library( Domains STATIC ${SRCS} )
target_link_libraries(Domains Learning Utilities Agents)

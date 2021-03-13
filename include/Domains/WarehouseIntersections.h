#ifndef WAREHOUSE_INTERSECTIONS_H_
#define WAREHOUSE_INTERSECTIONS_H_

#include <vector>
#include <list>
#include <Eigen/Eigen>
#include "Agents/Intersection.h"
#include "Warehouse.h"

using std::vector ;
using std::list ;

class WarehouseIntersections : public Warehouse {
  public:
    WarehouseIntersections(YAML::Node configs) : Warehouse(configs){}
    ~WarehouseIntersections(void) ;
    
    void SimulateEpoch(bool train = true) ;
    void SimulateEpoch(vector<size_t> team) ;
    
    void InitialiseMATeam() ; // create agents for each vertex in graph
    
  private:
    void QueryMATeam(vector<size_t>, vector<double>&, vector<size_t>&) ; // get current graph costs
    void GetJointState(vector<Edge *> e, vector<size_t> &eNum) ;
    size_t GetAgentID(int) ;
    
};

#endif // WAREHOUSE_INTERSECTIONS_H_

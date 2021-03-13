#ifndef WAREHOUSE_CENTRALISED_TIME_H_
#define WAREHOUSE_CENTRALISED_TIME_H_

#include <vector>
#include <list>
#include <Eigen/Eigen>
#include "Agents/Intersection.h"
#include "Warehouse.h"

using std::vector ;
using std::list ;

class WarehouseCentralisedTime : public Warehouse {
  public:
    WarehouseCentralisedTime(YAML::Node configs) : Warehouse(configs){}
    ~WarehouseCentralisedTime(void) ;
    
    void SimulateEpoch(bool train = true) ;
    void SimulateEpoch(vector<size_t> team) ;
    
    void InitialiseMATeam() ; // create agents for each vertex in graph
    
  private:
    void QueryMATeam(vector<size_t>, vector<double>&, vector<size_t>&) ; // get current graph costs
    void GetJointState(vector<Edge *> e, vector<size_t> &eNum, vector<double> &eTime) ;
    
};

#endif // WAREHOUSE_CENTRALISED_TIME_H_

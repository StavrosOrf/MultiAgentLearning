#ifndef WAREHOUSE_CENTRALISED_H_
#define WAREHOUSE_CENTRALISED_H_

#include <vector>
#include <list>
#include <Eigen/Eigen>
#include <algorithm>
#include "Warehouse.h"
#include "Agents/DDPGAgent.h"

using std::vector ;
using std::list ;

class WarehouseCentralised : public Warehouse {
	public:
		WarehouseCentralised(YAML::Node configs) : Warehouse(configs){}
		~WarehouseCentralised(void) ;

		void SimulateEpochDDPG();

		void InitialiseMATeam() ; // create agents for each vertex in graph
	private:
		void GetJointState(vector<Edge *> e, vector<size_t> &eNum) ;
};

#endif // WAREHOUSE_CENTRALISED_H_

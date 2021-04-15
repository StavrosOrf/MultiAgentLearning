#ifndef AGV_H_
#define AGV_H_

#include <vector>
#include <list>
#include <iostream>
#include <stdlib.h>
#include "Planning/Search.h"
#include "Planning/Edge.h"

class AGV{
	public:
		AGV(vertex_t, std::vector<vertex_t>, Graph *);
		~AGV();

		void ResetAGV();
		void Traverse();
		void EnterNewEdge();
		void CompareCosts(std::vector<float>);
		void PlanAGV(std::vector<float>);
		vertex_t GetNextVertex(){return nextVertex;}
		vertex_t get_cur_vertex(){assert(t2v==0); return nextVertex;}
		Edge * GetCurEdge(){return curEdge;}
		size_t GetT2V(){return t2v;}
		bool GetIsReplan(){return isReplan;}
		Edge * GetNextEdge(){return path.front();}

		size_t GetMoveTime(){return tMove;}
		size_t GetEnterTime(){return tEnter;}
		size_t GetWaitTime(){return tWait;}
		size_t GetNumCompleted(){return nsDel;}
		size_t GetNumCommanded(){return ncDel;}
		Search * GetAGVPlanner(){return agvPlanner;}
		bool is_on_graph(){return nextVertex!=-1;}
		bool is_on_edge(){return curEdge != NULL;}
		//bool is_on_edge(){return t2v > 0;}
		bool entered_edge_this_step(){return just_entered_edge;}
		vertex_t get_start_vertex(){return agvPlanner->GetSource();}
		std::vector<vertex_t> get_possible_goals() const {return agvGoals;}
		void reset_edge_enter(){just_entered_edge = false;}

		//vertex_t GetOriginVertex(){return origin;}
		vertex_t get_goal_vertex(){return goal;}

		void DisplayPath();
		void ResetPerformanceCounters(){nsDel=ncDel=tMove=tEnter=tWait=0;}

	private:
		Edge * curEdge;			// current edge
		vertex_t nextVertex;			// next vertex
		size_t t2v;			// time to next intersection
		vertex_t origin;			// origin vertex for initialisation
		vertex_t goal;			// goal vertex
		size_t nsDel;			// number of successful deliveries
		size_t ncDel;			// number of commanded deliveries
		size_t tMove;			// moving time
		size_t tEnter;			// time waiting to enter graph
		size_t tWait;			// time waiting to cross intersections
		std::vector<vertex_t> agvGoals;	// vector of valid goal vertices
		bool just_entered_edge;

		bool isReplan;			// true if replanning is needed

		Search * agvPlanner;		// planning routine
		std::vector<float> costs;	// graph costs used to generate current plan
		std::list<Edge *> path;		// current path as ordered list of edges
		void SetNewGoal();		// set new goal vertex
};

#endif // AGV_H_

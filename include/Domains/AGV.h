#ifndef AGV_H_
#define AGV_H_

#include <vector>
#include <list>
#include <stdlib.h>
#include "Planning/Search.h"
#include "Planning/Edge.h"

using std::vector ;
using std::list ;

class AGV{
  public:
    AGV(int, vector<int>, Graph *) ;
    ~AGV() ;
    
    void ResetAGV() ;
    void Traverse() ;
    void EnterNewEdge() ;
    void CompareCosts(vector<double>) ;
    void PlanAGV(vector<double>) ;
    int GetNextVertex(){return nextVertex ;}
    Edge * GetCurEdge(){return curEdge ;}
    size_t GetT2V(){return t2v ;}
    bool GetIsReplan(){return isReplan ;}
    Edge * GetNextEdge(){return path.front() ;}
    
    size_t GetMoveTime(){return tMove ;}
    size_t GetEnterTime(){return tEnter ;}
    size_t GetWaitTime(){return tWait ;}
    size_t GetNumCompleted(){return nsDel ;}
    size_t GetNumCommanded(){return ncDel ;}
    Search * GetAGVPlanner(){return agvPlanner ;}
    
    void DisplayPath() ;
    
  private:
    Edge * curEdge ;      // current edge
    int nextVertex ;      // next vertex
    size_t t2v ;          // time to next intersection
    int origin ;          // origin vertex for initialisation
    int goal ;            // goal vertex
    size_t nsDel ;        // number of successful deliveries
    size_t ncDel ;        // number of commanded deliveries
    size_t tMove ;        // moving time
    size_t tEnter ;       // time waiting to enter graph
    size_t tWait ;        // time waiting to cross intersections
    vector<int> agvGoals ;// vector of valid goal vertices
    bool isReplan ;       // true if replanning is needed
    
    Search * agvPlanner ; // planning routine
    vector<double> costs ;// graph costs used to generate current plan
    list<Edge *> path ;   // current path as ordered list of edges
    void SetNewGoal() ;   // set new goal vertex
} ;

#endif // AGV_H_

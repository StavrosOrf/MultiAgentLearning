#ifndef BAR_H_
#define BAR_H_

#include <iostream>
#include <vector>
#include <math.h>

using std::vector ;

class Bar{
  public:
    Bar(size_t c): capacity(c){}
    ~Bar(){}
    
    double GetReward(size_t nAgents, bool update=false){
      numAgents = nAgents ;
      useUpdated = update ;
      if (update)
        EnjoymentFunction = &Bar::UpdatedCongestion ;
      else
        EnjoymentFunction = &Bar::ClassicCongestion ;
      
      (this->*EnjoymentFunction)() ;
      return reward ;
    }
    
    size_t GetCapacity(){return capacity ;}
  private:
    size_t capacity ;
    bool useUpdated ;
    size_t numAgents ;
    double reward ;
    
    void (Bar::*EnjoymentFunction)() ;
    void ClassicCongestion(){
      reward = (double)numAgents * exp(-pow((double)numAgents-(double)capacity,2)) ;
//      std::cout << "Attendance: " << numAgents << ", total enjoyment: " << r << "\n" ;
    }
    
    void UpdatedCongestion(){ // currently the same, will update when we get to including "celebrity" agents
      reward = (double)numAgents * exp(-pow((double)numAgents-(double)capacity,2)) ;
    }
} ;

#endif // BAR_H_

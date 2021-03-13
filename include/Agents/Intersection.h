#ifndef INTERSECTION_H_
#define INTERSECTION_H_

#include "Agent.h"

class Intersection : public Agent{
  public:
    Intersection(size_t nPop, size_t nIn, size_t nOut, size_t nHidden) : 
      Agent(nPop, nIn, nOut, nHidden){}
};

#endif // INTERSECTION_H_

#ifndef LINK_H_
#define LINK_H_

#include "Agent.h"

class Link : public Agent {
  public:
    Link(size_t nPop, size_t nIn, size_t nOut, size_t nHidden) :
      Agent(nPop, nIn, nOut, nHidden){}
};

#endif // LINK_H_

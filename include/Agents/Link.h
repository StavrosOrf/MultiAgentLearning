#ifndef LINK_H_
#define LINK_H_

#include "NeuroEvoAgent.h"

class Link : public NeuroEvoAgent {
	public:
		Link(size_t nPop, size_t nIn, size_t nOut, size_t nHidden) :
			NeuroEvoAgent(nPop, nIn, nOut, nHidden){}
};

#endif // LINK_H_

#ifndef INTERSECTION_H_
#define INTERSECTION_H_

#include "NeuroEvoAgent.h"

class Intersection : public NeuroEvoAgent{
	public:
		Intersection(size_t nPop, size_t nIn, size_t nOut, size_t nHidden) :
			NeuroEvoAgent(nPop, nIn, nOut, nHidden){}
};

#endif // INTERSECTION_H_

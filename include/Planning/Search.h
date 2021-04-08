#ifndef SEARCH_H_
#define SEARCH_H_

#include <vector> // std::vector, std::cout
#include <math.h> // pow, abs, sqrt
#include <cassert>
#include <stddef.h>
#include "Edge.h"
#include "Graph.h"
#include "Node.h"
#include "Queue.h"


// Dijkstra's path search class to store and search a graph
class Search{
	public:
		Search(Graph * graph, int source, int goal):itsGraph(graph), itsQueue(0), itsSource(source), itsGoal(goal){}

		~Search(){
			delete itsQueue;
			itsQueue = 0;
		}

		Graph * GetGraph() const {return itsGraph;}
		Queue * GetQueue() const {return itsQueue;}
		void SetQueue(Queue * queue) {itsQueue = queue;}
		int GetSource() const {return itsSource;}
		void SetSource(int s){itsSource = s;}
		int GetGoal() const {return itsGoal;}
		void SetGoal(int g){itsGoal = g;}
		Node * PathSearch();
		float PathSearchLenght(); //return the cost/lenght of the path returned by PathSearch()
		void ResetSearch();
	private:
		Graph * itsGraph;
		Queue * itsQueue;
		int itsSource;
		int itsGoal;

		size_t FindSourceID();
};

#endif // SEARCH_H_

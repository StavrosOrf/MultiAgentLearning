#pragma once

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
		Search(Graph * graph, vertex_t source, vertex_t goal):itsGraph(graph), itsQueue(0), itsSource(source), itsGoal(goal){
			itsQueue = new Queue();	
		}


		~Search(){
			delete itsQueue;
			itsQueue = 0;
		}

		Graph * GetGraph() const {return itsGraph;}
		Queue * GetQueue() const {return itsQueue;}
		//void SetQueue(Queue * queue) {itsQueue = queue;}
		vertex_t GetSource() const {return itsSource;}
		void SetSource(vertex_t s){itsSource = s;}
		vertex_t GetGoal() const {return itsGoal;}
		void SetGoal(vertex_t g){itsGoal = g;}
		Node * PathSearch();
		float PathSearchLenght(); //return the cost/lenght of the path returned by PathSearch()
		void ResetSearch();
	private:
		Graph * itsGraph;
		Queue * itsQueue;
		vertex_t itsSource;
		vertex_t itsGoal;

		size_t FindSourceID() [[deprecated("Serves no purpose, use getSource instead")]];
};
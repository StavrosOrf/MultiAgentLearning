#pragma once

#include <vector> // std::vector
#include <queue> // std::priority_queue
#include "Node.h"
#include <boost/heap/priority_queue.hpp>

struct CompareNode{
	bool operator() (const Node * n1, const Node * n2) const{
		float n1Cost = n1->GetCost();
		float n2Cost = n2->GetCost();
		return (n2Cost < n1Cost);
	}
};

// Custom queue type to perform priority queue updates
class Queue{
	public:
		//TODO(@Kallinteris) Move to boost::heap priority_queue for perfomance?
		typedef std::priority_queue<Node *, std::vector<Node *>, CompareNode> QUEUE;
		//typedef boost::heap::priority_queue<Node *> QUEUE;
		Queue(Node * source){
			itsPQ.push(source);
			//closed.reserve(100);
		}
		Queue(){}

		~Queue(){
			while (!itsPQ.empty()){
				Node * temp = itsPQ.top();
				delete temp;
				temp = 0;
				itsPQ.pop();
			}
			for (size_t i = 0; i < closed.size(); i ++){
				if (closed[i]){
					delete closed[i];
					closed[i] = 0;
				}
			}
		}

		std::vector<Node *> GetClosed() const {return closed;}
		bool EmptyQueue() const {return itsPQ.empty();}
		size_t SizeQueue() const {return itsPQ.size();}
		void UpdateQueue(Node * newNode);
		Node * PopQueue();
		void reset(){
			delete_queue();
			delete_closed();
			itsPQ = QUEUE(); 
			//itsPQ.clear();
			closed.clear(); 
			assert(itsPQ.empty() && closed.empty());
		}
	private:
		QUEUE itsPQ;
		std::vector<Node *> closed;
		void delete_queue() {
			while(!itsPQ.empty()){
				assert(itsPQ.top());
				Node * temp = itsPQ.top();
				delete temp;
				temp = 0;
				itsPQ.pop();
			}
		}
		void delete_closed() {
			for (size_t i = 0; i < closed.size(); i ++){
				assert (closed[i]);
				delete closed[i];
				closed[i] = 0;
			}
		}
};
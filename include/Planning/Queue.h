#ifndef QUEUE_H_
#define QUEUE_H_

#include <vector> // std::vector
#include <queue> // std::priority_queue
#include "Node.h"

using std::vector ;
using std::priority_queue ;

struct CompareNode{
  bool operator() (const Node * n1, const Node * n2) const{
    double n1Cost = n1->GetCost() ;
    double n2Cost = n2->GetCost() ;
    return (n2Cost < n1Cost) ;
  }
} ;

// Custom queue type to perform priority queue updates
class Queue
{
	public:
  	typedef priority_queue<Node *, vector<Node *>, CompareNode> QUEUE ;
		Queue(Node * source){
	    itsPQ = new QUEUE ;
	    itsPQ->push(source) ;
    }
    
		~Queue(){
			while (!itsPQ->empty()){
				Node * temp = itsPQ->top() ;
				delete temp ;
				temp = 0 ;
				itsPQ->pop() ;
			}
	    delete itsPQ ;
	    itsPQ = 0 ;
	    for (size_t i = 0; i < closed.size(); i ++){
		    delete closed[i] ;
		    closed[i] = 0 ;
	    }
    }
		
		vector<Node *> GetClosed() const {return closed ;}
		bool EmptyQueue() const {return itsPQ->empty() ;}
		size_t SizeQueue() const {return itsPQ->size() ;}
		void UpdateQueue(Node * newNode) ;
		Node * PopQueue() ;
    
	private:
		QUEUE * itsPQ ;
		vector<Node *> closed ;
} ;

#endif //QUEUE_H_

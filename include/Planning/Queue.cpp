#include "Queue.h"

void Queue::UpdateQueue(Node * newNode)
{
  itsPQ->push(newNode) ;
}

Node * Queue::PopQueue()
{
  closed.push_back(itsPQ->top()) ;
  itsPQ->pop() ;
  return closed[closed.size()-1] ;
}

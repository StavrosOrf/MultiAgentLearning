#ifndef AGENT_H_
#define AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class Agent{
	public:
		Agent(size_t nIn, size_t nOut, size_t nHidden);
		~Agent();


		void TrainAgent();

		size_t GetNumIn(){return numIn;}
		size_t GetNumHidden(){return numHidden;}
		size_t GetNumOut(){return numOut;}

	protected:
		size_t numIn;
		size_t numOut;
		size_t numHidden;

};

#endif // AGENT_H_

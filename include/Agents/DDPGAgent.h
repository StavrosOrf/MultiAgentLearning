#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Eigen>

using std::vector ;
using std::string ;


class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space);
		~DDPGAgent();
	protected:
};

#endif // DDPG_AGENT_H_

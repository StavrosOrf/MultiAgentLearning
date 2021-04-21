#include "Warehouse_ES_container.hpp"


Warehouse_ES_container::Warehouse_ES_container(YAML::Node configs){
	epoch = configs["ES"]["epochs"].as<int>();

	for(int i = 0 ; i< POP_SIZE; i++){
		population.push_back(new Warehouse_ES(configs));
		population[i]->InitialiseMATeam();
	}

}

void Warehouse_ES_container::evolution_strategy(){

	//get initial random policy //TODO check that each inital policy is different
	std::vector<esNN*> team =  population[0]->produce_random_team_NNs();
	std::vector<epoch_resultsES> results(POP_SIZE);
	float sum = 0,scalar;

	for (int i = 0; i < epoch; i++)
	{
		for (int j = 0; j < POP_SIZE; j++)
		{
			population[j]->setTeamNNs(team);
		}		

		//TODO multi-thread
		for (int j = 0; j < POP_SIZE; j++)
		{
			results[j] = population[j]->SimulateEpochES();	
		}		

		//calculate scalar step
		sum = 0;
		for (int j = 0; j < POP_SIZE; j++)
			sum += results[j].totalDeliveries * results[j].sample;

		scalar = LEARNING_RATE * sum / ( POP_SIZE * STD_DEV);

		for (int j = 0 ; j < team.size(); j ++){
			for (size_t k = 0; k < team[j]->parameters().size(); k++ ){
				torch::Tensor t = team[j]->parameters()[k].detach().clone();
				team[j]->parameters()[k].set_data(t + scalar);
			}		
		}	
		
	}
}


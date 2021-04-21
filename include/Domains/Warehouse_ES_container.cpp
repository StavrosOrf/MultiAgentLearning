#include "Warehouse_ES_container.hpp"


Warehouse_ES_container::Warehouse_ES_container(YAML::Node configs){
	epoch = configs["ES"]["epochs"].as<int>();

	for(int i = 0 ; i< POP_SIZE; i++){
		population.push_back(new Warehouse_ES(configs));
		population[i]->InitialiseMATeam();
	}

}

void Warehouse_ES_container::evolution_strategy(bool verbose){

	//get initial random policy //TODO check that each inital policy is different
	std::vector<esNN*> team =  population[0]->produce_random_team_NNs();
	std::vector<epoch_resultsES> results(POP_SIZE);
	float sum = 0,scalar;

	for (int i = 0; i < epoch; i++)
	{	
		float max_deliveries = -1;
		
		//Multithread this for loop
		for (int j = 0; j < POP_SIZE; j++)
		{
			population[j]->setTeamNNs(team);
			results[j] = population[j]->SimulateEpochES();	
		}		

		// synchronize here
		for (int j = 0; j < POP_SIZE; j++){
			if(max_deliveries < results[j].totalDeliveries){
				max_deliveries = results[j].totalDeliveries;
			}
		}

		if(verbose)
			std::cout <<"=== Epoch: "<<i<<" =========================================================== max G:"<<max_deliveries<< std::endl;


		//calculate scalar step
		sum = 0;
		for (int j = 0; j < POP_SIZE; j++){
			sum += results[j].totalDeliveries * results[j].sample;
			// std::cout<<"Sample: " <<results[j].sample<<std::endl;
		}



		scalar = LEARNING_RATE * sum / ( POP_SIZE * STD_DEV);
		if(verbose)
			std::cout <<scalar<<std::endl;

		for (int j = 0 ; j < team.size(); j ++){
			for (size_t k = 0; k < team[j]->parameters().size(); k++ ){
				torch::Tensor t = team[j]->parameters()[k].detach().clone();
				team[j]->parameters()[k].set_data(t + scalar);
			}		
		}	
		
	}
}


#include "Warehouse_ES_container.hpp"
#define MULTITHREADED

Warehouse_ES_container::Warehouse_ES_container(YAML::Node configs){
	epoch = configs["ES"]["epochs"].as<int>();
	learning_rate = configs["ES"]["learning_rate"].as<float>();
	const size_t population_size = configs["ES"]["population_size"].as<int>();
	assert(population_size > 0 && epoch > 0);

	for(size_t i = 0 ; i < population_size; i++){
		population.push_back(new Warehouse_ES(configs));
		population[i]->InitialiseMATeam();
	}

}

uint Warehouse_ES_container::evolution_strategy(size_t n_threads, bool verbose){
	//get initial random policy
	std::vector<esNN*> team = population[0]->produce_random_team_NNs(); //TODO RENAME team?
	std::vector<epoch_resultsES> results(population.size());

	uint max_deliveries_intra = 0;
	for (int i = 0; i < epoch; i++){	
		uint max_deliveries = 0;
		
#ifdef MULTITHREADED
		boost::asio::thread_pool simulator_pool(n_threads);
#endif
		for (size_t j = 0; j < population.size(); j++){
#ifdef MULTITHREADED
			boost::asio::post(simulator_pool, [i, j, team, &results, this](){
#endif
				population[j]->set_team_NNs(team);
				results[j] = population[j]->SimulateEpochES();	
#ifdef MULTITHREADED
			});
#endif
		}
#ifdef MULTITHREADED
		simulator_pool.join();
#endif

		for (epoch_resultsES r : results)
			max_deliveries = std::max<uint>(max_deliveries, r.totalDeliveries);
		max_deliveries_intra = std::max<uint>(max_deliveries, max_deliveries_intra);

		if(verbose)
			std::cout << "=== Epoch: "<<i<<" =========================================================== max G:"<<max_deliveries<< std::endl;


		//calculate scalar step
		float sum = 0;
		for (epoch_resultsES r : results)
			sum += r.totalDeliveries * r.sample;
		const float scalar = learning_rate * sum / ( population.size() * STD_DEV);
		//if(verbose)
			//std::cout << "scalar:" << scalar << std::endl;

		for (esNN* nn: team)
			for (auto p : nn->parameters())
				p.set_data(p.detach().clone() + scalar);
	}
	return max_deliveries_intra;
}
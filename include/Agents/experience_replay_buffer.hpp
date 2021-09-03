#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <stdlib.h>
#include "experience_replay.hpp"

class experience_replay_buffer{
	public:
		experience_replay_buffer(size_t buffer_capacity_arg){
			replay_buffer.reserve(buffer_capacity_arg);
			buffer_capacity = buffer_capacity_arg;
		}
		
		size_t get_size(){return replay_buffer.size();}
		void clear(){replay_buffer.clear();}

		/************************************************************************************************
		**Input:[size] of batch to return								*
		**Method:Selects a non-inclusive (with unique items) minibanch from the experience_replay buffer*
		**Output:Returns a non-inclusive miniBatch of [size]						*
		*************************************************************************************************/
		std::vector<experience_replay> get_batch(size_t size){
			assert(replay_buffer.size() >= size);
			std::vector<experience_replay> to_return;
			to_return.reserve(size);
			
			std::ranges::sample(replay_buffer, std::back_inserter(to_return), size, std::mt19937{std::random_device{}()});
			/*
			//generate a list of rand indexes (non inclusive)
			std::vector<int> rand_list;
			rand_list.reserve(size);
			for (size_t i = 0; i != size; i++){
				int r = rand() % replay_buffer.size();
				while (std::find(rand_list.begin(), rand_list.end(),r)!=rand_list.end())
					r = rand() % replay_buffer.size();
				rand_list.push_back(r);
			}

			for (int i : rand_list)
				to_return.push_back(replay_buffer[i]);

			assert(to_return.size() == size);
			*/
			return to_return;
		}

		/************************************************************************************************
 		**Input: A experience_replay [r]								*
 		**Method:Adds [r] to the [replay_buffer], if [replay_buffer] is full it evicts a random tuple	*
 		*************************************************************************************************/
		void add_random(experience_replay r){
			assert(r.next_state.size() == r.current_state.size());
		
			if (replay_buffer.size() < buffer_capacity)
				replay_buffer.push_back(r);
			else
				replay_buffer[std::rand() % buffer_capacity] = r;
		}
		
		void add_sequential();
		
	private:
		std::vector<experience_replay> replay_buffer;
		size_t buffer_capacity;
};

#ifndef NN_MODULE_H
#define NN_MODULE_H

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

struct Net : torch::nn::Module {
	Net(int numIn, int numOut, int numHid, const size_t hid_count=1) {
		assert(hid_count >= 1);
		//first = register_parameter("inputW", torch::rand({numIn, numHid}))/numHid;
		first = register_parameter("inputW", torch::randn({numIn, numHid}));
		parameters()[0].set_data((parameters()[0]-0.5)/numHid*2);
		middle = new torch::Tensor[hid_count-1];
		for (size_t i = 1; i != hid_count; i++){
			middle[i] = register_parameter("hidW"+std::to_string(i), torch::randn({numHid, numHid}));
			parameters()[i].set_data((parameters()[i]-0.5)/numHid*2);
		}
		//last = register_parameter("outputW", torch::rand({numHid, numOut}))/numOut;
		last = register_parameter("outputW", torch::randn({numHid, numOut}));
		parameters()[hid_count].set_data((parameters()[hid_count]-0.5)/numHid*2);
		h_c = hid_count;
	}
	torch::Tensor forward(const torch::Tensor input) {
		assert(torch::sum(first == parameters()[0]).item<float>() == first.numel());
		torch::Tensor output_layer,h;
		// const torch::Tensor r_input = (input-0.5)*2;
		h = torch::tanh(torch::mm(input, first));
		for (size_t i = 1; i != h_c; i++)
			h = (torch::mm(h, middle[i]));
		output_layer = torch::tanh(torch::mm(h, last));
		return output_layer;
	}
	torch::Tensor first, last, *middle;
	size_t h_c;
};

struct ActorNN : torch::nn::Module {
	ActorNN (int numIn, int numOut, int numHid) {		
		fc1 = register_module("fc1",torch::nn::Linear(numIn,numHid));
		fc2 = register_module("fc2",torch::nn::Linear(numHid,numHid));
		fc3 = register_module("fc3",torch::nn::Linear(numHid,numOut));
	}

	torch::Tensor forward(torch::Tensor x) {

		// std::cout<<std::isnan(x.item<float>())<<std::end;
		//Normalize [0,1] x after each layer		
		// auto m = torch::min(x);
		// auto s = torch::max(x);		
		// if(torch::is_nonzero(torch::sum(x)) && x.numel() != 1)
		// 	x = (x-m)/(s-m);				
		// x = torch::relu(fc1->forward(x));
		// x = torch::relu(fc2->forward(x));
		x = torch::relu(fc1->forward(x));
		// m = torch::min(x);
		// s = torch::max(x);		
		// if(torch::is_nonzero(torch::sum(x)) || x.numel() != 1)
		// 	x = (x-m)/(s-m);	
		x = torch::relu(fc2->forward(x));
		x = torch::tanh(fc3->forward(x));

		// return -x;
		// m = torch::min(x);
		// s = torch::max(x);		
		// if(torch::is_nonzero(torch::sum(x)) || x.numel() != 1 )
		// 	x = (x-m)/(s-m);				
		// assert(!std::isnan(x.item<float>()));
		return x;
	}

	torch::nn::Linear fc1{nullptr},fc2{nullptr},fc3{nullptr};	
};

struct CriticNN : torch::nn::Module {
	CriticNN (int numIn, int numOut, int numHid) {
		// n1 = register_module("n1",torch::nn::LayerNorm(numIn));
		fc1 = register_module("fc1",torch::nn::Linear(numIn,numHid));
		fc2 = register_module("fc2",torch::nn::Linear(numHid,numHid));
		fc3 = register_module("fc3",torch::nn::Linear(numHid,numOut));
	}

	torch::Tensor forward(torch::Tensor x) {	
		// auto m = torch::min(x);
		// auto s = torch::max(x);		
		// if(torch::is_nonzero(torch::sum(x)))
		// 	x = (x-m)/(s-m);		

		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = (fc3->forward(x));
		
		// m = torch::min(x);
		// s = torch::max(x);		
		// if(torch::is_nonzero(torch::sum(x)))
		// 	x = (x-m)/(s-m);								
		return x;
	}

	torch::nn::Linear fc1{nullptr},fc2{nullptr},fc3{nullptr};		
};

struct esNN : torch::nn::Module {
	esNN (int numIn, int numOut, int numHid=0) {
		fc1 = register_module("fc1",torch::nn::Linear(numIn,numOut));
	}

	torch::Tensor forward(torch::Tensor x) {	
		x = torch::sigmoid(fc1->forward(x));							
		return x;
	}

	torch::nn::Linear fc1{nullptr};		
};

#endif // NN_MODULE_H

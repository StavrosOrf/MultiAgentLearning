#include "NeuralNet.h"

// Constructor: Initialises NN given layer sizes, also initialises NN activation function, currently has hardcoded mutation rates, mutation value std and bias node value
NeuralNet::NeuralNet(size_t numIn, size_t numOut, size_t numHidden, actFun afType, nnOut bOut){
  bias = 1.0 ;
  MatrixXd A(numIn, numHidden) ;
  weightsA = A ;
  MatrixXd B(numHidden+1, numOut) ;
  weightsB = B ;
  mutationRate = 0.5 ;
  mutationStd = 1.0 ;
  
  if (afType == TANH){
    ActivationFunction = &NeuralNet::HyperbolicTangent ;
  }
  else if (afType == LOGISTIC){
    ActivationFunction = &NeuralNet::LogisticFunction ;
  }
  else{
    std::cout << "ERROR: Unknown activation function type! Using default hyperbolic tangent function.\n" ;
    ActivationFunction = &NeuralNet::HyperbolicTangent ;
  }
  
  if (bOut == BOUNDED){
    layerActivation.push_back(0) ;
    layerActivation.push_back(1) ;
  }
  else{
    layerActivation.push_back(0) ;
    layerActivation.push_back(2) ;
  }
    
  InitialiseWeights(weightsA) ;
  InitialiseWeights(weightsB) ;
  
  eta = 0.0001 ; // learning rate for backprop
}

// Evaluate NN output given input vector
VectorXd NeuralNet::EvaluateNN(VectorXd inputs){
  VectorXd hiddenLayer = (this->*ActivationFunction)(inputs, layerActivation[0]) ;
  VectorXd outputs = (this->*ActivationFunction)(hiddenLayer, layerActivation[1]) ;
  return outputs ;
}

// Evaluate NN output given input vector
VectorXd NeuralNet::EvaluateNN(VectorXd inputs, VectorXd & hiddenLayer){
  hiddenLayer = (this->*ActivationFunction)(inputs, layerActivation[0]) ;
  VectorXd outputs = (this->*ActivationFunction)(hiddenLayer, layerActivation[1]) ;
  return outputs ;
}

// Mutate the weights of the NN according to the mutation rate and mutation value std
void NeuralNet::MutateWeights(){
  for (int i = 0; i < weightsA.rows(); i++)
    for (int j = 0; j < weightsA.cols(); j++)
      weightsA(i,j) += RandomMutation() ;
  
  for (int i = 0; i < weightsB.rows(); i++)
    for (int j = 0; j < weightsB.cols(); j++)
      weightsB(i,j) += RandomMutation() ;
}

// Adds random amount mutationRate% of the time
double NeuralNet::RandomMutation() {
  if (rand_interval(0, 1) > mutationRate)
    return 0.0;
  else {
    // FOR MUTATION
    std::default_random_engine generator;
    generator.seed(static_cast<size_t>(time(NULL)));
    std::normal_distribution<double> distribution(0.0, mutationStd);
    return distribution(generator);
  }
}

// Assign weight matrices
void NeuralNet::SetWeights(MatrixXd A, MatrixXd B){
  weightsA = A ;
  weightsB = B ;
}

// Wrapper for writing NN weight matrices to specified files
void NeuralNet::OutputNN(const char * A, const char * B){
  // Write NN weights to txt files
  // File names stored in A and B
	std::stringstream NNFileNameA ;
	NNFileNameA << A ;
	std::stringstream NNFileNameB ;
	NNFileNameB << B ;

  WriteNN(weightsA, NNFileNameA) ;
  WriteNN(weightsB, NNFileNameB) ;
}

void NeuralNet::BackPropagation(vector<VectorXd> trainInputs, vector<VectorXd> trainTargets){
  double sumSquaredError = DBL_MAX ;
  double threshold = 0.001 ;
  
  size_t step = 0 ;
  while (sumSquaredError > threshold*trainTargets.size() && step < 10000){
    // Feedforward NN evaluation to compute targets
    vector<VectorXd> trainOutputs ;
    vector<VectorXd> hiddenLayers ;
    for (size_t i = 0; i < trainInputs.size(); i++){
      VectorXd hidden(weightsA.cols()) ;
      VectorXd tt = EvaluateNN(trainInputs[i], hidden) ;
      hiddenLayers.push_back(hidden) ;
      trainOutputs.push_back(tt) ;
    }
    
    for (size_t t = 0; t < trainInputs.size(); t++){
      // Calculate error terms for weight matrix B(hidden neurons, output neurons)
      MatrixXd deltaB(weightsB.rows(),weightsB.cols()) ;
      VectorXd deltaL(weightsB.cols()) ; // store for calculating error terms for weight matrix A
      // Hidden layer neurons
      VectorXd hidden(hiddenLayers[t].size()+1) ;
      hidden.head(hiddenLayers[t].size()) = hiddenLayers[t] ;
      hidden(hiddenLayers[t].size()) = bias ;
      MatrixXd net = hidden.transpose()*weightsB ;
      for (int i = 0; i < weightsB.rows(); i++){
        double oi = hidden(i) ;
        for (int j = 0; j < weightsB.cols(); j++){
          double netj = net(j) ;
          double oj = trainOutputs[t](j) ;
          double denom = pow((exp(-netj) + exp(netj)),2) ;
          deltaL(j) = (oj - trainTargets[t](j))*4.0/denom ;
          deltaB(i,j) = -eta*oi*deltaL(j) ;
        }
      }
      
      // Calculate error terms for weight matrix A(input neurons, hidden neurons)
      MatrixXd deltaA(weightsA.rows(),weightsA.cols()) ;
      net = trainInputs[t].transpose()*weightsA ;
      for (int i = 0; i < weightsA.rows(); i++){
        double oi = trainInputs[t](i) ;
        for (int j = 0; j < weightsA.cols(); j++){
          double deltaJ  = 0.0 ;
          for (int l = 0; l < weightsB.cols(); l++)
            deltaJ += deltaL(l)*weightsB(j,l) ;
          double netj = net(j) ;
          double denom = pow((exp(-netj) + exp(netj)),2) ;
          deltaA(i,j) = -eta*oi*deltaJ*4.0/denom ;
        }
      }
      
      weightsA += deltaA ;
      weightsB += deltaB ;
      
//      for (int i = 0; i < weightsA.rows(); i++){
//        for (int j = 0; j < weightsA.cols(); j++){
//          std::cout << deltaA(i,j) << " " ;
//        }
//        std::cout << "\n" ;
//      }
//      
//      for (int i = 0; i < weightsB.rows(); i++){
//        for (int j = 0; j < weightsB.cols(); j++){
//          std::cout << deltaB(i,j) << " " ;
//        }
//        std::cout << "\n" ;
//      }
    }
    sumSquaredError = 0.0 ;
    for (size_t i = 0; i < trainInputs.size(); i++){
      VectorXd tt = EvaluateNN(trainInputs[i]) ;
      VectorXd d = tt - trainTargets[i] ;
      for (int j = 0; j < d.size(); j++){
        d(j) = pow(d(j),2) ;
        sumSquaredError += d(j) ;
      }
    }
    if (step % 1000 == 0)
      std::cout << "SSE step" << step << ": " << sumSquaredError << "\n" ;
    step++ ;
  }
}

// Write weight matrix values to file
void NeuralNet::WriteNN(MatrixXd A, std::stringstream &fileName){
  std::ofstream NNFile ;
  NNFile.open(fileName.str().c_str()) ;
  for (int i = 0; i < A.rows(); i++){
	  for (int j = 0; j < A.cols(); j++)
	    NNFile << A(i,j) << "," ;
    NNFile << "\n" ;
	}
	NNFile.close() ;
}

// Initialise NN weight matrices to random values
void NeuralNet::InitialiseWeights(MatrixXd & A){
  double fan_in = A.rows() ;
  for (int i = 0; i < A.rows(); i++){
    for (int j = 0; j< A.cols(); j++){
      // For initialization of the neural net weights
      double rand_neg1to1 = rand_interval(-1, 1)*0.1;
      double scale_factor = 100.0;
      A(i,j) = scale_factor*rand_neg1to1 / sqrt(fan_in);
    }
  }
}

// Hyperbolic tan activation function
VectorXd NeuralNet::HyperbolicTangent(VectorXd input, size_t layer){
  VectorXd output ;
  if (layer == 0){
    output = input.transpose()*weightsA ;
    for (int i = 0; i < output.size(); i++)
      output(i) = tanh(output(i)) ;
  }
  else if (layer == 1){
    VectorXd hidden(input.size()+1) ;
    hidden.head(input.size()) = input ;
    hidden(input.size()) = bias ;
    output = hidden.transpose()*weightsB ;
    for (int i = 0; i < output.size(); i++)
      output(i) = tanh(output(i)) ;
  }
  else if (layer == 2){
    VectorXd hidden(input.size()+1) ;
    hidden.head(input.size()) = input ;
    hidden(input.size()) = bias ;
    output = hidden.transpose()*weightsB ;
  }
  else{
    std::printf("Error: second argument must be in {0,1,2}!\n") ;
  }
  
  return output ;
}

// Logistic function activation function
VectorXd NeuralNet::LogisticFunction(VectorXd input, size_t layer){
  VectorXd output ;
  if (layer == 0){
    output = input.transpose()*weightsA ;
    for (int i = 0; i < output.size(); i++)
      output(i) = 1/(1+exp(-output(i))) ;
  }
  else if (layer == 1){
    VectorXd hidden(input.size()+1) ;
    hidden.head(input.size()) = input ;
    hidden(input.size()) = bias ;
    output = hidden.transpose()*weightsB ;
    for (int i = 0; i < output.size(); i++)
      output(i) = 1/(1+exp(-output(i))) ;
  }
  else if (layer == 2){
    VectorXd hidden(input.size()+1) ;
    hidden.head(input.size()) = input ;
    hidden(input.size()) = bias ;
    output = hidden.transpose()*weightsB ;
  }
  else{
    std::printf("Error: second argument must be in {0,1,2}!\n") ;
  }
  
  return output ;
}

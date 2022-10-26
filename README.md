# Analyzing the Performance Impact of Combining Agent Factorization with Different Learning Algorithms for Multiagent Coordination.

This project implements various algorithms on a warehouse domain using multiple
agent definitions with each learning algorithm, for purpose of comparing it to
[CCEA](https://github.com/JenJenChung/multiagent_learning).

It contains implementations of the following algorithms on the warehouse domain/framework
 - ES
 - (MA)DDPG
 - I-DQN

The Analysis of the results can be found in our [paper](https://dl.acm.org/doi/10.1145/3549737.3549773).

## Installation ##

### Linux ###

Install system dependencies (on ubuntu):
```
sudo apt install libboost-dev libyaml-cpp-dev
```

In the working folder of your choice, clone the project code
```
git clone git@github.com:stavrosgreece/multiagentlearning.git
```


install CTorch/LibTorch (C++ version of PyTorch):
```
cd multiagentlearning
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch-shared-with-deps-1.10.1+cpu.zip
rm libtorch-shared-with-deps-1.10.1+cpu.zip
```
Note: you can download and install LibTorch with different computer platform (e.g. CUDA/RocM) on https://pytorch.org/, be sure to instal the c++11+ ABI version



Build the code:
```
cd multiagentlearning
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

## Running preconfigured projects

Run the project configured by `config.yaml` using six threads:
```
./testWarehouse -c ../config.yaml -t 6
```

## Debug
```
cmake -DCMAKE_BUILD_TYPE=Debug .. && make && gdb --args ./testWarehouse -c ../config.yaml
```
```
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make && valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./testWarehouse -c ../config.yaml
```
## Profiling
With Perf (Recommended if you are not made out RAM)
```
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make && perf record -g ./testWarehouse -c ../config.yaml
```
With Callgrind
```
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make && valgrind --tool=callgrind ./testWarehouse -c ../config.yaml
```

# Authors
 - [Kallinteris Andreas](https://github.com/kallinteris-andreas)
 - [Starvos Orfanoudakis](https://github.com/stavrosgreece/)
 
This is repo is a fork of https://github.com/JenJenChung/multiagent_learning

# Cite
```
    @inproceedings{10.1145/3549737.3549773,
    author = {Kallinteris, Andreas and Orfanoudakis, Stavros and Chalkiadakis, Georgios},
    title = {The Performance Impact of Combining Agent Factorization with Different Learning Algorithms for Multiagent Coordination},
    year = {2022},
    isbn = {9781450395977},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3549737.3549773},
    doi = {10.1145/3549737.3549773},
    abstract = {Factorizing a multiagent system refers to partitioning the state-action space to individual agents and defining the interactions between those agents. This so-called agent factorization is of much importance in real-world industrial settings, and is a process that can have significant performance implications. In this work, we explore if the performance impact of agent factorization is different when using different learning algorithms in multiagent coordination settings. We evaluated six different agent factorization instances—or agent definitions—in the warehouse traffic management domain, comparing the performance of (mainly) two learning algorithms suitable for learning coordinated multiagent policies: the Evolutionary Strategies (ES), and a genetic algorithm (CCEA) previously used in this setting. Our results demonstrate that different learning algorithms are affected in different ways by alternative agent definitions. Given this, we can deduce that many important multiagent coordination problems can potentially be solved by an appropriate agent factorization in conjunction with an appropriate choice of a learning algorithm. Moreover, our work shows that ES is an effective learning algorithm for the warehouse traffic management domain; while, interestingly, celebrated policy gradient methods do not fare well in this complex real-world problem setting.},
    booktitle = {Proceedings of the 12th Hellenic Conference on Artificial Intelligence},
    articleno = {32},
    numpages = {10},
    keywords = {Agent Factorization, Multiagent Coordination, Evolutionary Strategies, Warehouse Traffic Management},
    location = {Corfu, Greece},
    series = {SETN '22}
    }
```

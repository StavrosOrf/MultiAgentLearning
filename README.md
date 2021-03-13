# multiagent_learning

Top level project files for multiagent learning.

## Installation ##

### Linux ###

Install system dependencies:
```
sudo apt install libboost-dev libeigen3-dev libyaml-cpp-dev
```

In the working folder of your choice, clone the project code:
```
git clone git@github.com:stavrosgreece/multiagentlearning.git
```

Build the code:
```
cd multiagentlearning
mkdir build && cd build
cmake .. && make
```

## Running preconfigured projects

Run the project configured by `config.yaml` using six threads:
```
./testWarehouse -c ../config.yaml -t 6
```

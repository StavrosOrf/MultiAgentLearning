changelog format:
## v [semver][k/o depending on who pushed the update]
- change #0
- change #1 blablablablablablablablablablablablablablabla
blablablablablablabla (no tabs after newline)
[newline after each version]

## v 0.00000
-Added (structure for) new Learning Algorithm IterRL

## v 0.00001k
- added create_warehouse function
- fixed typos in testWarehouse.cpp
- added config.yaml to main directory
- added Domains/* to main dir
- added include as to the dir (and removed it from .gitignore)

## v 0.00002k
- updated build instructions in readme.md

## v 0.00003k / prep for DDPG
- changed indent to tabs
- added new skeleton classes: AgentDDPG

## v 0.00004o
- Undocumented

## v 0.00005k
- Refactored Agent to NeuroEvoAgent
- Deprecated Nightbar Agent Night Bar Domain and Night Bar Test (Not Relevant
to our code base)
- General Code Cleanup
- Made Progress on DDPG init
- General Control Flow Fix up

## v 0.00006k
- Deprecated Rovers class (and it's associated domains and tests)
- Implemented DDPG's deconstructor

## v 0.00007o
- Implented critic back propagations
- Finished DDPG constructor

## v 0.00008k / they call me the mop cause i CLEAN UP
- Deprecated bar.cpp, Target.cpp, MAPElites.cpp, POMDP.cpp (kinda),
GaussianProcess.h, all POMDPs, testLapackInverse.cpp, testMultiRobotROS.cpp
- Cleaned up CMAKE files
- Added a few utility functions in warehouse.cpp
- General Cleanup of Wahehouse::simulateEpochDDPG()
- assert!, assert?, assert is love assert life, join the cult of asserts
- Streamlined the results output process in testwarehouse.cpp
- Found and commented a bunch of bugs

## v 0.00009o
- Fixed Warehouse Simulation ( Now all edges properly count edges ),
might need to restore some basic ware house functions
- Minor print fixes

## v 0.00010k
- sed all the trailing white spaces away
- minor pefomance tweak
- added changelog format (top of this file)
- added testNeuralNet.cpp to cmake list
- deprecated testNeuralNet.cpp (it is bugged and uses unfinished code)

## v 0.00011k
- deprecated NEUROEVO and everything associated with it
- deprecated allmulti agent structures (will bring back later)
- Changed source formating now all lines that end with ';' do not contain
white space characters between the last character and ';'
- added a bunch of function comments

## v 0.00012k / Light up your (c)torches
- imported the c++ frontent for PyTorch's backend also updated cmake and
README.md and tested it

## v 0.00013o
- fixed compile errors
- implemented critic update( might need detailed testing after completing the algorithm)
- implemented actor update ( requires tests and bounds )

## v 0.00014o
- fixed and tested "update actor policy"
- from_blob method doesnt work properly, an other method is proposed in the update actor section
- TODO fix update Critic, all Evaluate functions
- TODO implement Update Targe NN weights

## v 0.00015o
- fixed the logic errors resulting from the use of "from_blob"
- started working on update weights

## v 0.00016k
- Minor performance tweaks (memory system optimizations)
- General code clean up
- Deprecated NeuralNet.cpp & Utilities.cpp (all of them exist in std::), any
trace if Eigen, Threadpool.cpp
- Implemented a Reward Heuristic Function (it is shit though)
- Changed all double floats to single float (for runtime performance reasons)
- Added Debug instructions in README.md
- Fixed Various Bugs
- Added test_torch.cpp, test_random.cpp, and created new ./result Directory
- Added cleancode.sh in the main directory

## v 0.00017o
- added some metrics ( time, totalMove ...)
- TODO more metrics can be added like Best G of Run etc
- TODO add metrics about loss functions of NNs
- TODO Store Results for each step/Epoch/Run in files so we can get Graphs for our report/paper

## v 0.00018k
- Huge Runtime Performance Tweaks
- Minor Tweaks to reward function (needs more work)
- testWarehouse now writes Evaluation parameters to results folder
- Streamlined Domains
- Now supports centralized warehouse with provided time

## v 0.00019k
- bunch of micro optimizations
- Finalized Reward Function (this is not true)
- Improved Reward Function

## v 0.00020o
- some additions for Link Agent

## v 0.00021k
- Initialize Intersection Multi Agent Team Implemented

## v 0.00022o ( yes we have changelog!!!)
- Implemented and tested MADDPG link agents
- Added Training step to reduce computational requirements
  (will do tests to prove that is not necessary to train in each step)
- TODO link_t doesn't reads state correctly ( the time part is always 0)

## v 0.00023k
- Renamed WarehouseCentraised Class to Warehouse_DDPG
- Added options for number of AGVs
- Deprecated Learning library
- Added Tiny Warehouse
- Added a bool verbose argument to all function that print
- Moved some constant DDPG arguments from #define to config.yaml
- General Code Cleanup

## v 0.00024k
- Wrote Skeleton for DDPG_merged_step (need to be renamed)

## v 0.00025k
- Code Cleanup and performance optimizations
- Fix Bug related to AGV routing

## v 0.00026o
- Optimizers are now correctly initialized and used
- Changed the way of exploration -> I believe that with this style of 
  "randomness" we better explore the action space
- Swapped to Register_Module NNs ( now we are sure everything works as intended -bias nodes etc..)    
  Disclaimer: We, might still need to try some small varainces of Layers,Normalizations etc.
  but this is not a priority
- Added an other reward function that calculates the amount of total Deliveries in a given time
  (inspired by the lectures)
- I cannot say that the algorithm works now, but it has potential and 2 more days of tests!!!

## v 0.00027k
- ChangeLog (this file) is now a markdown document
- torch::nn::modules are all in a different file (nn_modules.h)
- minor code cleanup
- minor bug fixing
- deprecated merged_step.h/.cpp
- renamed .h header files to .hpp
- Reduced ddpgAgent::REPLAY_BUFFER_SIZE to 1024*1024 (which results to sub 1 gigabyte of ram needed for one run)
- Reward Method 2 is now likely broken 
- Reward Method 1 Now runs Faster and provides a higher accuracy of the value

## v 0.00028 o
- Tried to have Q with more than 1 output(not working)
- Coppied the NN parameters from another DDPG implementation
- tested with these parameters and failed successfully 

## v 0.00029 o (k)
- Added full functionality for ES (not fully tested)

## v 0.00030k
- implemented multithreading to ES
- moved ES parameters from #define to config.yaml
- minor code cleanup on ES

## v 0.00031 o (k) it works!?
- Updated Noise addition(and tested centralized)

## v 0.00032 o
- Added print to file functionality for ES algorrithm
- Added the option to save - load team NNs
- Discovered major memory leak Issues!!! 

## v 0.00032 k perfomance imporvements
- Improved memory layout of graph
- Gotten rid of useless state
- Reduced heap allocations on Search's queue
- Added Caching of Edge's references

## v 0.00033 o
- fixed copy best team_policy
- experimented with different NNs / fixed typos / Reduction of the hidden node number lead to better memory performance(256->64 nodes)
- 

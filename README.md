# Bayesian-Network-Sampling-methods-2022
This project serves for Sampling methods of Bayesian Network.

In simple.py file I try to sample some random variables with the following distributions from uniform distribution.

    1- 0.3*gaussian(4,2) + 0.3*gaussian(3,2) + 0.4*exponential(0.01)
  
    2- 0.2*gaussian(0,10) + 0.2*gaussian(20,15) + 0.3*gaussian(-10,8) + 0.3*gaussian(50,25)
  
    3- 0.2*geometric(0.1) + 0.2*geometric(0.5) + 0.2*geometric(0.3) + 0.4*geometric(0.04)
In  sample.py file I try to sampling a given Bayesian-Network with these methods:
    
    Prior sampling
    Rejection sampling
    Likelihood sampling
    Gibs sampling

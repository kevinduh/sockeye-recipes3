## Hyperparameter Optimization with Asynchronous Successive Halving Algorithm (ASHA)

This example implements the ASHA paper ([A system for Massively Parallel Hyperparameter Tuning](https://arxiv.org/pdf/1810.05934.pdf)) for Sockeye. 
ASHA is a bandit learning algorithm that looks at the learning curves of multiple training runs. Not-so-promising runs are terminated early, in order to allocate more computational resources to promising runs. It balances exploration (trying new configurations) with exploitation (training promising configurations for longer). 
ASHA is the asychronous version of the Successive Halving algorithm. 
For more info, refer to the paper or to the [blog post](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/) by the original authors, Liam Li et. al. 


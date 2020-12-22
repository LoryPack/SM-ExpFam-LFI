# SM-ExpFam-LFI

Code for the paper: __Score Matched Conditional Exponential Families for Likelihood-Free Inference__, which can be found [here](https://arxiv.org/abs/2012.10903). 

## Instructions

All experiments and plots from the paper can be reproduced using the code provided here. Notice however that some scripts are computationally intensive and may need to be run on a parallel cluster (specifically those performing ABC, SL and RE inference).

The content of this repository is as follows: 

- `src` contains source code for performing NN training and inference
- `scripts` contains Python scripts to run the different experiments
- `plot_scripts` contains Python scripts to reproduce the figures starting from results
- `tests` contains some tests for our source code; see below

### Reproducing the experiments

We provide bash scripts calling the Python scripts with the options we used in the paper and in the right order. 

For instance, in order to run the experiments on the beta model, it suffices to do: 
    
    chmod +x run_beta_experiments.sh  # to make it runnable 
    ./run_beta_experiments.sh

However, it may be wiser to take the different Python commands and run them independently one by one; in fact, some of them have very different computational complexities, specifically some may require a large number of MPI tasks to run efficiently. Alternatively, you can also use GPUs to train the neural networks.

### Requirements
Listed in the `requirements.txt` file. Please install with: 

    pip install -r requirements.txt


### Tests
We test some source code we introduce. To run them, do:
     
    python -m unittest tests


## Citation
Please use the following `.bib` entry:

    @misc{pacchiardi2020score,
      title={Score Matched Conditional Exponential Families for Likelihood-Free Inference}, 
      author={Lorenzo Pacchiardi and Ritabrata Dutta},
      year={2020},
      eprint={2012.10903},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
    }

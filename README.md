# JPMC & QCWare Deep Hedging

A collaborative research project with source code to be released as an open source archived repo on paper publication.

Open Source @ JPMorgan Chase 2022
## Overview

### Part 1

- 'train_and_test.ipynb':  Notebook to train and test models
- 'hardware.ipynb': Notebook to test models on hardware
- 'data': Folder containing data used to reproduce results in the paper
- 'params': Folder containing model parameters used to reproduce results in the paper
- 'source':  Folder containing source code for the models
- 'source/models.py': Source code for the models: simple, recurrent, lstm and attention
- 'source/qnn.py': Necessary functions to build quantum orthogonal layers
- 'source/train.py': Main script to run an experiment. Includes function to generate GBM paths, loss metric, train and test functions.
- 'source/utils.py': Contains utility functions
- 'source/hardware_utils.py' : Contains utility functions for hardware experiments  
- 'source/config.py': Contains variables to hold data during hardware executions

## Reproducing the results in the manuscript

- Table 1:  `Part_1/train_and_test.ipynb`
- Table 2: `Part_1/hardware.ipynb`
- Table 3: `Part_1/hardware.ipynb`
- Table 4: 'Part_2/t'
- Table 5:  `Part_2/hardware.ipynb`
- Table 6: `Part_2/hardware.ipynb`

## OmniQ setup (JPMC) 

When creating an instance to use the code on OmniQ, *do not* run the standard `setup.sh` script. Instead, use the `omniq_setup.sh` script provided in this repo.

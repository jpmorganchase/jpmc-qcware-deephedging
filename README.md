# JPMC & QCWare Deep Hedging

A collaborative research project with source code to be released as an open source archived repo on paper publication.


## File structure
- `models.py` : Deep hedging model configurations; simple, recurrent, lstm, attention
- `main.py` : Main script for running an experiment
- `train.py`: Deep hedging model training
- `utils.py`: Specification of all Hyperparameters and util functions
- `data.py`: Implementation of module generating paths for geometric Brownian motion.


Open Source @ JPMorgan Chase 2022

## OmniQ setup (JPMC) 

When creating an instance to use the code on OmniQ, *do not* run the standard `setup.sh` script. Instead, use the `omniq_setup.sh` script provided in this repo.

from sklearn import model_selection

import pickle
from dataclasses import dataclass

from jax import numpy as jnp
from jax.tree_util import tree_map

@dataclass
class HyperParams:
  S0: int = 100
  n_steps: int = 30
  strike_price: int = 100
  epsilon: float = 0.0
  sigma: float = 0.2 
  risk_free: float = 0.0
  dividend: float = 0.0
  model_type: str = 'simple'
  layer_type: str = 'linear'
  hidden_dim: int = 16
  n_layers: int = 3
  noise_scale: float = 0.01
  loss_param: float = 1.0
  batch_size = 256
  share_strategy: bool = False

def train_test_split(data=None, test_size=None):
    """Split simulated data into training and testing sample."""
    xtrain = []
    xtest = []
    for x in data:
        tmp_xtrain, tmp_xtest = model_selection.train_test_split(
            x, test_size=test_size, shuffle=False)
        xtrain += [tmp_xtrain]
        xtest += [tmp_xtest]
    return xtrain, xtest

def get_batches(data, batch_size):
  k = batch_size
  num_batches = len(data) // k
  batches = [ data[i*k:(i+1)*k] for i in range(num_batches) ]
  return num_batches, jnp.array(batches)

def save_params(file_name, params):
  with open(file_name, "wb") as f:
    pickle.dump(params, f)

def load_params(file_name):
  with open(file_name, "rb") as f:
    params = pickle.load(f)
    # convert NP arrays to Jax arrays
    return tree_map(lambda param: jnp.array(param), params)
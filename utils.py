from sklearn import model_selection
from dataclasses import dataclass
from jax import numpy as jnp
from jax.tree_util import tree_map
import pickle
import optax

OptimizerFn = optax.GradientTransformation

OPTIMIZERS = [
    'adagrad',
    'adam',
    'radam',
    'adamw',
    'fromage',
    'lamb',
    'lars',
    'noisy_sgd',
    'sgd',
    'rmsprop',
    'yogi',
]


@dataclass
class HyperParams:
    S0: int = 100
    n_steps: int = 30
    n_paths: int = 120000
    strike_price: int = 100
    epsilon: float = 0.0
    sigma: float = 0.2
    risk_free: float = 0.0
    dividend: float = 0.0
    model_type: str = 'simple'
    layer_type: str = 'linear'
    n_features: int = 16
    n_layers: int = 3
    noise_scale: float = 0.01
    loss_param: float = 1.0
    batch_size = 256
    test_size = float = 0.2
    optimizer: str = 'adam'
    learning_rate: float = 1E-3
    num_epochs: int = 100


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
    batches = [data[i*k:(i+1)*k] for i in range(num_batches)]
    return num_batches, jnp.array(batches)


def save_params(file_name, params):
    with open(file_name, "wb") as f:
        pickle.dump(params, f)


def load_params(file_name):
    with open(file_name, "rb") as f:
        params = pickle.load(f)
        # convert NP arrays to Jax arrays
        return tree_map(lambda param: jnp.array(param), params)


def make_optimizer(optimizer: str = 'adam',
                   learning_rate: float = 1E-3) -> OptimizerFn:
    """ Creates an optimizer with the specified parameters. """
    assert optimizer in OPTIMIZERS, f'{optimizer} is not a valid optimizer. Choose from {OPTIMIZERS}'
    opt_cls = getattr(optax, optimizer)
    opt = opt_cls(learning_rate)
    return opt

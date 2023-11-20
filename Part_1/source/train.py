# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co and QC Ware
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import jax
import numpy as np
import optax
import tqdm
from jax import numpy as jnp
from source import utils
from source.qnn import orthogonalize_params
from source.utils import HyperParams
from tqdm import tqdm, trange


def gen_paths(hps):
    """Generate paths for geometric Brownian motion.
    Args:
        hps: HyperParams
    Returns:
        paths: (n_paths, n_steps + 1) array of paths
    """
    dt = 1 / 252
    paths = np.zeros((hps.n_steps + 1, hps.n_paths), np.float64)
    paths[0] = hps.S0
    for t in tqdm(range(1, hps.n_steps + 1)):
        rand = np.random.standard_normal(hps.n_paths)
        rand = (rand - rand.mean()) / rand.std()
        if hps.discrete_path:
            rand = np.asarray([2 * int(i > 0) - 1 for i in rand])
            paths[t] = paths[t - 1] + rand
        else:
            paths[t] = paths[t - 1] * np.exp(
                (hps.risk_free - 0.5 * hps.sigma**2) * dt
                + hps.sigma * np.sqrt(dt) * rand
            )
    return paths.T


def entropy_loss(hps, wealths):
    entropy_scale = hps.loss_param
    return (1 / entropy_scale) * jnp.log(jnp.mean(jnp.exp(-entropy_scale * wealths)))


def build_train_fn(
    hps,
    net,
    opt,
    loss_metric,
    epsilon=0.0,
    wealth_init=0.0,
    strike_price=100.0,
):
    """Build a train function for the given hyperparameters.

    Args:

        hps: HyperParams
        net: ModuleFn
        opt: OptimizerFn
        loss_metric: MetricFn
        epsilon: float
        wealth_init: float
        strike_price: float

    Returns:
        train_fn: train function
        loss_fn: loss function

    """

    def loss_fn(params, state, key, inputs):
        if hps.discrete_path:
            I = inputs - 100
        else:
            I = jnp.log(inputs / 100)
        outputs, state = net.apply(params, state, key, I[:, :-1, :])
        outputs = jnp.concatenate(
            (outputs, jnp.zeros_like(outputs[:, [0], :])),
            axis=1,
        )
        deltas = jnp.concatenate(
            (
                outputs[:, [0], :],
                outputs[:, 1:, :] - outputs[:, :-1, :],
            ),
            axis=1,
        )
        wealths = wealth_init
        wealths -= jnp.einsum("ijk,ijk->ik", jnp.abs(deltas), inputs) * hps.epsilon
        wealths -= jnp.einsum("ijk,ijk->ik", deltas, inputs)
        wealths += jnp.einsum("ijk,ijk->ik", outputs[:, [-1], :], inputs[:, [-1], :])
        wealths -= jnp.maximum(inputs[:, -1] - strike_price, 0.0)
        loss = loss_metric(hps, wealths)
        return loss, (state, wealths, deltas, outputs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_fn(params, state, opt_state, key, inputs):
        (loss, (state, wealths, deltas, outputs)), grads = grad_fn(
            params, state, key, inputs
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if hps.layer_type == "linear_svb":
            params = orthogonalize_params(params)
        return params, state, opt_state, loss, (wealths, deltas, outputs)

    return train_fn, loss_fn


if __name__ == "__main__":

    seed = 42
    key = jax.random.PRNGKey(seed)
    hps = HyperParams(
        S0=100,
        n_steps=30,
        n_paths=120000,
        discrete_path=False,
        strike_price=100,
        epsilon=0,
        sigma=0.2,
        risk_free=0,
        dividend=0,
        model_type="lstm",
        layer_type="linear",
        n_features=8,
        n_layers=1,
        loss_param=1.0,
        batch_size=256,
        test_size=0.2,
        optimizer="adam",
        learning_rate=1e-3,
        num_epochs=100,
    )

    # Data
    # Generate paths for geometric Brownian motion using the specified hyperparameters.
    # Create batches of data for training.
    # The batch_size parameter specifies the number of data points in each batch.
    # The resulting train_batches object is an iterable containing the batches.
    S = gen_paths(hps)
    [S_train, S_test] = utils.train_test_split([S], test_size=hps.test_size)
    _, train_batches = utils.get_batches(
        jnp.array(S_train[0]), batch_size=hps.batch_size
    )

    # Model
    # Define a neural network with the specified hyperparameters.

    # Create a layer function based on the specified layer type.
    layer_func = utils.make_layer(layer_type=hps.layer_type)

    # Create a neural network model based on the specified model type.
    # The model takes in hyperparameters, hps, and the layer function as input.
    net = utils.make_model(hps.model_type)(hps=hps, layer_func=layer_func)

    # Create an optimizer with the specified settings.
    opt = utils.make_optimizer(optimizer=hps.optimizer, learning_rate=hps.learning_rate)

    # Initialize the network parameters, optimizer state, and loss metric.
    # The key is used for random initialization.
    key, init_key = jax.random.split(key)
    params, state, _ = net.init(init_key, (1, hps.n_steps, 1))
    opt_state = opt.init(params)
    loss_metric = entropy_loss

    # Training  the neural network model on the generated data.

    # Define the training function and loss function for the neural network.
    train_fn, loss_fn = build_train_fn(
        hps=hps, net=net, opt=opt, loss_metric=loss_metric
    )
    # Initialize the loss variable.
    loss = 0.0

    # Train the model for the specified number of epochs.
    # Iterate over the batches of training data.
    # Prepare the input data for the neural network.
    # Generate a new random key for each training step.
    # Update the neural network parameters and optimizer state using the training function.
    # The function returns the updated parameters, state, optimizer state, loss, and other metrics.
    # Update the progress bar with the current loss and hyperparameters.
    with trange(1, hps.num_epochs + 1) as t:
        for epoch in t:
            for i, inputs in enumerate(train_batches):
                inputs = inputs[..., None]
                key, train_key = jax.random.split(key)
                params, state, opt_state, loss, (wealths, deltas, outputs) = train_fn(
                    params, state, opt_state, train_key, inputs
                )
            t.set_postfix(
                loss=loss, model=hps.model_type, layer=hps.layer_type, eps=hps.epsilon
            )

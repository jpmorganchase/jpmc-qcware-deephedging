import jax
import optax
from jax import numpy as jnp

from qnn import orthogonalize_params


def build_train_fn(
    hps,
    net,
    opt,
    loss_metric,
    epsilon=0.,
    wealth_init=0.,
    strike_price=100.,
):
    """ Build a train function for the given hyperparameters.
    
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
        outputs, state = net.apply(params, state, key, I[:,:-1,:])
        outputs = jnp.concatenate( 
            (
                outputs,
             jnp.zeros_like(outputs[:, [0], :])
        ), axis=1,
            )
        deltas = jnp.concatenate(
            (
                outputs[:, [0], :],
                outputs[:, 1:, :] - outputs[:, :-1, :],
            ),
            axis=1,
        )
        wealths = wealth_init
        wealths -= jnp.einsum('ijk,ijk->ik',
                              jnp.abs(deltas), inputs) * hps.epsilon
        wealths -= jnp.einsum('ijk,ijk->ik', deltas, inputs)
        wealths += jnp.einsum('ijk,ijk->ik',
                              outputs[:, [-1], :], inputs[:, [-1], :])
        wealths -= jnp.maximum(inputs[:, -1] - strike_price, 0.0)
        loss = loss_metric(hps, wealths)
        return loss, (state, wealths, deltas, outputs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_fn(params, state, opt_state, key, inputs):
        (loss, (state, wealths, deltas,
                outputs)), grads = grad_fn(params, state, key, inputs)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if hps.layer_type == 'linear_svb':
            params = orthogonalize_params(params)
        return params, state, opt_state, loss, (wealths, deltas, outputs)

    return train_fn, loss_fn

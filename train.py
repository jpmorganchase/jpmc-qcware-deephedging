import jax
import optax
from jax import numpy as jnp
from qnn import orthogonalize_params


def build_train_fn(
    hps,
    net,
    opt,
    loss_metric,
):
    def loss_fn(params, state, key, inputs):
        log_inputs = jnp.log(inputs / hps.strike_price)
        outputs, state = net.apply(params, state, key, log_inputs)
        deltas = jnp.concatenate(
            (
                outputs[:, [0], :],
                outputs[:, 1:, :] - outputs[:, :-1, :],
            ),
            axis=1,
        )
        wealths = hps.wealth_init
        # transaction fee
        wealths -= jnp.einsum('ijk,ijk->ik', jnp.abs(deltas), inputs) * hps.epsilon
        # raw cost of buying hedging instruments
        wealths -= jnp.einsum('ijk,ijk->ik', deltas, inputs)
        # value of the hedges at maturity
        wealths += jnp.einsum('ijk,ijk->ik', outputs[:, [-1], :], inputs[:, [-1], :])
        # value of the short call option
        wealths -= jnp.maximum(inputs[:, -1] - hps.strike_price, 0.0)
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

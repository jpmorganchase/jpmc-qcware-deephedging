from jax import numpy as jnp

def entropy_loss(hps, wealths):
    entropy_scale = hps.loss_param
    return (1 / entropy_scale) * jnp.log(
        jnp.mean(jnp.exp(-entropy_scale * wealths)))
from jax import numpy as jnp

def entropy_loss(wealths, entropy_scale=1.):
    return (1 / entropy_scale) * jnp.log(
        jnp.mean(jnp.exp(-entropy_scale * wealths)))
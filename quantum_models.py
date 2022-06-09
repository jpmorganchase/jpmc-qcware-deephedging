import jax
from jax import numpy as jnp
import qnn
from qnn import ModuleFn, elementwise, ortho_linear, linear, sequential

relu = elementwise(jax.nn.relu)
gelu = elementwise(jax.nn.gelu)
log_softmax = elementwise(jax.nn.log_softmax)
sigmoid = elementwise(jax.nn.sigmoid)


def ortho_shared_simple_network(n_features: int = 16, n_layers: int = 3, **kwargs):
    preprocessing = [linear(n_features), sigmoid]
    features = n_layers * [ortho_linear(n_features), relu]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)
    return net


def _tile(tree, n_timesteps):
    return jax.tree_map(
        lambda param: jnp.tile(param, (n_timesteps, *param.ndim * (1, ))),
        tree)


def ortho_simple_network(n_features: int = 16, n_layers: int = 3, **kwargs):
    preprocessing = [linear(n_features), sigmoid]
    features = n_layers * [ortho_linear(n_features), relu]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)

    def init_fn(key, inputs_shape):
        params, state, outputs_shape = net.init(key, inputs_shape)
        params = _tile(params, inputs_shape[-2])
        state = _tile(state, inputs_shape[-2])
        return params, state, (*inputs_shape[:-1], outputs_shape[-1])

    def apply_fn(params, state, key, inputs, **kwargs):
        key = jax.random.split(key, inputs.shape[-2])
        outputs, state = jax.vmap(net.apply,
                                  in_axes=(0, 0, 0, 1))(params, state, key,
                                                        inputs)
        outputs = outputs.transpose(1, 0, 2)
        return outputs, state

    return ModuleFn(apply_fn, init=init_fn)

from typing import NamedTuple

class LSTMState(NamedTuple):
  hidden : jnp.ndarray
  cell: jnp.ndarray

def ortho_lstm_cell(n_features=16):

    _linear = ortho_linear(n_features=n_features, with_bias=True)
    
    def init_fn(key, inputs_shape):
        keys = jax.random.split(key, num = 4)    
        params = {}
        params['i'] = _linear.init(keys[0],  (inputs_shape[0],inputs_shape[1],2*inputs_shape[2]))[0]
        params['g'] = _linear.init(keys[1],  (inputs_shape[0],inputs_shape[1],2*inputs_shape[2]))[0]
        params['f'] = _linear.init(keys[2],  (inputs_shape[0],inputs_shape[1],2*inputs_shape[2]))[0]
        params['o'] = _linear.init(keys[3],  (inputs_shape[0],inputs_shape[1],2*inputs_shape[2]))[0]

        return params, None, inputs_shape
    

    def apply_fn(params, state, key, inputs):
        def cell_fn(prev_state, inputs):
            prev_hidden, prev_cell = prev_state
            x_and_h = jnp.concatenate([inputs, prev_hidden], axis=-1)
            i = _linear.apply(params['i'], None, None, x_and_h)[0]
            g = _linear.apply(params['g'], None, None, x_and_h)[0]
            f = _linear.apply(params['f'], None, None, x_and_h)[0]
            o = _linear.apply(params['o'], None, None, x_and_h)[0]
            # i = input, g = cell_gate, f = forget_gate, o = output_gate
            f = jax.nn.sigmoid(f + 1) 
            c = f * prev_cell + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return jnp.stack( [h,c],axis=0), h
        
        prev_state = jnp.zeros((2,inputs.shape[0],inputs.shape[-1]))
        inputs = inputs.transpose(1,0,2)
        _, outputs = jax.lax.scan(cell_fn, prev_state, inputs)
        outputs = outputs.transpose(1,0,2)
        return outputs, state
    return qnn.ModuleFn(apply_fn, init_fn)

def ortho_lstm_network(n_features=16, **kwargs):
    preprocessing = [linear(n_features), sigmoid]
    features = [ortho_lstm_cell()]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)
    return net



def ortho_attention_v0(
    n_features: int,
    layout: str = 'butterfly',
) -> ModuleFn:
    """ Create an attention layer.
    
    Args:
        n_features: The number of features.
        layout: The layout of the RBS gates.
    """
    norm = qnn.layer_norm()
    to_w = ortho_linear(n_features,
                        layout=layout,
                        with_scale=False,
                        with_bias=False,
                        normalize_inputs=False,
                        normalize_outputs=False)
    
    to_v = ortho_linear(n_features,
                        layout=layout,
                        with_scale=True,
                        with_bias=False,
                        normalize_inputs=True,
                        normalize_outputs=False)
    def apply_fn(params,state, key, inputs, **kwargs):
        
        n_params = qnn.get_params_by_scope('norm', params)
        w_params = qnn.get_params_by_scope('weights', params)
        v_params = qnn.get_params_by_scope('value', params)
        #norm_outputs = norm.apply(n_params, None, None, inputs)[0]
        causal_mask = jnp.tril(jnp.ones((inputs.shape[1], inputs.shape[1])))
        case = 1
        dots = jnp.matmul(to_w.apply(w_params,None, None, inputs)[0],
                        inputs.transpose(0, 2, 1))
        if case == 0:
            dots = jnp.where(causal_mask, dots, -jnp.inf)
            w = jax.nn.softmax(dots/ jnp.sqrt(causal_mask.sum(axis=-1)), axis=-1)
        elif case == 1:
            w = jnp.square(dots) * causal_mask 
            w /= jnp.sum(w, axis=-1, keepdims=True)
        elif case == 2:
            w = causal_mask / causal_mask.sum(axis=-1)
        v = to_v.apply(v_params,None, None, inputs)[0]
        outputs = jnp.einsum('...ij,...jk->...ik',w, v)
        outputs += inputs
        #outputs = norm.apply(n_params, None, None, outputs)[0]
         # Residue
        return outputs, state

    def init_fn(key, inputs_shape):
        params = {}
        w_key, v_key = jax.random.split(key)
        w_params = to_w.init(w_key, inputs_shape)[0]
        v_params = to_v.init(v_key, inputs_shape)[0]
        n_params = norm.init(v_key, inputs_shape)[0]
        params.update(qnn.add_scope_to_params('norm', n_params))
        params.update(qnn.add_scope_to_params('weights', w_params))
        params.update(qnn.add_scope_to_params('value', v_params))
        return params, None, inputs_shape[:-1] + (n_features, )


    return ModuleFn(apply_fn, init=init_fn)



def ortho_attention_network(n_features: int = 16, n_layers: int = 3, **kwargs):
    n_features = 8
    preprocessing = [linear(n_features), sigmoid]
    features = n_layers * [ortho_linear(n_features), sigmoid,] + [ortho_attention_v0(n_features)]
    postprocessing =  [linear(1), sigmoid] 
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)
    return net
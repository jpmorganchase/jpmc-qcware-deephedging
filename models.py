import itertools
from typing import Any, TypeVar

import jax
from jax import numpy as jnp

import qnn
from qnn import (ModuleFn, _get_butterfly_idxs, _get_pyramid_idxs, elementwise,
                 linear, make_general_orthogonal_fn, sequential)

HyperParams = TypeVar('HyperParams')

relu = elementwise(jax.nn.relu)
gelu = elementwise(jax.nn.gelu)
log_softmax = elementwise(jax.nn.log_softmax)
sigmoid = elementwise(jax.nn.sigmoid)


def simple_network(hps: HyperParams, layer_func: ModuleFn = linear, **kwargs) -> ModuleFn:
    """ Create a Simple Network.

    Args:
        n_features: The number of features.
        n_layers: The number of layers.
        layer_func: The type of layers to use.
    """
    preprocessing = [linear(hps.n_features), sigmoid]
    features = hps.n_layers * [layer_func(hps.n_features), relu]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)

    def apply_fn(params, state, key, inputs, **kwargs):
        batch_size = inputs.shape[0]
        T = jnp.arange(0, inputs.shape[-2], 1, dtype='float32')
        T = T[..., None]
        T = jnp.array([T]*batch_size)
        inputs = jnp.concatenate((inputs, T), axis=2)
        outputs, state = net.apply(params, state, key, inputs)
        return outputs, state

    def init_fn(key, inputs_shape):
        params = net.init(
            key, (inputs_shape[0], inputs_shape[1], 2*inputs_shape[2]))[0]
        return params, None, inputs_shape

    return ModuleFn(apply_fn, init_fn)


def recurrent_network(hps: HyperParams, layer_func: ModuleFn = linear, **kwargs) -> ModuleFn:
    """ Create a Recurrent Network.

    Args:
        n_features: The number of features.
        n_layers: The number of layers.
        layer_func: The type of layers to use.
    """

    preprocessing = [linear(hps.n_features), sigmoid]
    features = hps.n_layers * [layer_func(hps.n_features), relu]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)

    def init_fn(key, inputs_shape):
        params = net.init(
            key, (inputs_shape[0], inputs_shape[1], 2*inputs_shape[2]))[0]
        return params, None, inputs_shape

    def apply_fn(params, state, key, inputs):
        def cell_fn(prev_outputs, inputs):
            inp = inputs[None, ...]
            inp = jnp.concatenate([prev_outputs, inp], axis=-1)
            delta = net.apply(params, None, key, inp)[0]
            return delta, delta

        prev_state = jnp.zeros((1, inputs.shape[0], inputs.shape[-1]))
        inputs = inputs.transpose(1, 0, 2)
        _, outputs = jax.lax.scan(cell_fn, prev_state, inputs)
        outputs = jnp.squeeze(outputs, 1)
        outputs = outputs.transpose(1, 0, 2)
        return outputs, state
    return qnn.ModuleFn(apply_fn, init_fn)


def lstm_cell(hps: HyperParams,  layer_func: ModuleFn = linear, **kwargs) -> ModuleFn:
    """ Create an LSTM Cell.

    Args:
        n_features: The number of features.
        layer_func: The type of layers to use.
    """

    _linear = layer_func(n_features=hps.n_features, with_bias=True)

    def init_fn(key, inputs_shape):
        keys = jax.random.split(key, num=4)
        params = {}
        layer_idx = ['i', 'g', 'f', 'o']
        _shape = (inputs_shape[0], inputs_shape[1], 2*inputs_shape[2])
        _init_params = {}
        for i, id in enumerate(layer_idx):
            _init_params[id] = _linear.init(keys[i],  _shape)[0]
            params.update(qnn.add_scope_to_params(id, _init_params[id]))
        return params, None, inputs_shape

    def apply_fn(params, state, key, inputs):
        def cell_fn(prev_state, inputs):
            prev_hidden, prev_cell = prev_state
            x_and_h = jnp.concatenate([inputs, prev_hidden], axis=-1)
            layer_idx = ['i', 'g', 'f', 'o']
            _apply_params = {}
            for i, id in enumerate(layer_idx):
                _apply_params[id] = qnn.get_params_by_scope(id, params)
            i = _linear.apply(_apply_params['i'], None, key, x_and_h)[0]
            g = _linear.apply(_apply_params['g'], None, key, x_and_h)[0]
            f = _linear.apply(_apply_params['f'], None, key, x_and_h)[0]
            o = _linear.apply(_apply_params['o'], None, key, x_and_h)[0]
            # i = input, g = cell_gate, f = forget_gate, o = output_gate
            f = jax.nn.sigmoid(f + 1)
            c = f * prev_cell + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return jnp.stack([h, c], axis=0), h

        prev_state = jnp.zeros((2, inputs.shape[0], inputs.shape[-1]))
        inputs = inputs.transpose(1, 0, 2)
        _, outputs = jax.lax.scan(cell_fn, prev_state, inputs)
        outputs = outputs.transpose(1, 0, 2)
        return outputs, state
    return qnn.ModuleFn(apply_fn, init_fn)


def lstm_network(hps: HyperParams, layer_func: ModuleFn = linear, **kwargs) -> ModuleFn:
    """ Create an LSTM Network.

    Args:
        n_features: The number of features.
        layer_func: The type of layers to use.
    """
    preprocessing = [linear(hps.n_features), sigmoid]
    features = [lstm_cell(hps=hps, layer_func=layer_func)]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)
    return net


def attention_layer(
    hps: HyperParams,
    layer_func: ModuleFn = linear,
) -> ModuleFn:
    """ Create an Attention layer.

    Args:
        hps.n_features: The number of features.
        layout: The layout of the RBS gates.
        layer_func: The type of layers to use.
    """
    norm = qnn.layer_norm()
    to_w = layer_func(hps.n_features, with_bias=False)
    to_v = layer_func(hps.n_features, with_bias=True)

    def apply_fn(params, state, key, inputs, **kwargs):

        n_params = qnn.get_params_by_scope('norm', params)
        w_params = qnn.get_params_by_scope('weights', params)
        v_params = qnn.get_params_by_scope('value', params)
        #norm_outputs = norm.apply(n_params, None, None, inputs)[0]
        causal_mask = jnp.tril(jnp.ones((inputs.shape[1], inputs.shape[1])))
        case = 0
        dots = jnp.matmul(to_w.apply(w_params, None, key, inputs)[0],
                          inputs.transpose(0, 2, 1))
        if case == 0:
            dots = jnp.where(causal_mask, dots, -jnp.inf)
            w = jax.nn.softmax(
                dots / jnp.sqrt(causal_mask.sum(axis=-1)), axis=-1)
        elif case == 1:
            w = jnp.square(dots) * causal_mask
            w /= jnp.sum(w, axis=-1, keepdims=True)
        elif case == 2:
            w = causal_mask / causal_mask.sum(axis=-1)
        v = to_v.apply(v_params, None, key, inputs)[0]
        outputs = jnp.einsum('...ij,...jk->...ik', w, v)
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
        return params, None, inputs_shape[:-1] + (hps.n_features, )

    return ModuleFn(apply_fn, init=init_fn)


def timestep_layer():
    """ Creates positional embedding of timesteps.
    """

    def init_fn(key, inputs_shape):
        params = {}
        num_timesteps = inputs_shape[1]
        num_features = inputs_shape[2]
        params['pos_embed'] = jax.random.normal(
            key, shape=(1, num_timesteps, num_features))

        return params, None, inputs_shape

    def apply_fn(params, state, key, inputs, **kwargs):
        return inputs + params['pos_embed'], state

    return ModuleFn(apply_fn, init=init_fn)


def attention_network(hps: HyperParams, layer_func=linear,  **kwargs) -> ModuleFn:
    """ Create a Attention Network.

    Args:
        n_features: The number of features.
        n_layers: The number of layers.
        layer_func: The type of layers to use.
    """
    preprocessing = [linear(hps.n_features), sigmoid, timestep_layer()]
    features = hps.n_layers * [layer_func(hps.n_features), sigmoid, ] + [
        attention_layer(hps=hps, layer_func=layer_func)]
    postprocessing = [linear(1), sigmoid]
    layers = preprocessing + features + postprocessing
    net = sequential(*layers)
    return net

def quantum_network(hps: HyperParams, **kwargs) -> ModuleFn:
  
    if hps.layer_type == 'butterfly':
      rbs_idxs_fn = _get_butterfly_idxs
    else:
      rbs_idxs_fn = _get_pyramid_idxs
    def encoding_fn(inputs):
        T = inputs.shape[-1] - 1
        deltas = inputs[:,1:]-inputs[:,:-1]
        jumps = (deltas+1)/2
        encodings = [jnp.ones((inputs.shape[0],2**T))/2**(T/2)]
        for t in range(1,T+1):
          past_jumps = jumps[...,:t]
          future_jumps = jnp.array(list(itertools.product(*((T-t)*[[0,1]]))))
          past_jumps = jnp.repeat(past_jumps[...,None,:], 2**(T-t), -2)
          future_jumps = jnp.repeat(future_jumps[None,...], past_jumps.shape[0], 0)
          superposition = jnp.concatenate([past_jumps, future_jumps],-1)
          basis = 2**jnp.arange(T)
          idxs = (superposition @ basis)
          idxs = jax.lax.convert_element_type(idxs, jnp.int32)
          encoding = jnp.zeros((idxs.shape[0], 2**T))
          batch_idxs = jnp.repeat(jnp.arange(past_jumps.shape[0])[...,None],2**(T-t),-1)
          idxs = jnp.stack([batch_idxs,idxs],-1).reshape(-1,2)
          encoding = encoding.at[tuple(idxs.T)].set(1.)/2.**((T-t)/2)
          encodings.append(encoding)
        encodings = jnp.stack(encodings,1)
        return encodings
    def measurement_fn(probs):
        T = probs.shape[-2] - 1
        results = []
        for t in range(0,T):
          idxs = [sum(binary)
                                for binary in itertools.product(*( [0, 2**q]
                                                        for q in range(t,T)))
                      ]
          map_measurement = [ list(jnp.arange(2**t)+i) for i in idxs]
          marginal =  probs[t][jnp.asarray(sum(map_measurement,[]))].reshape(2**(T-t),2**t).sum(-1)
          support = jnp.linspace(0,1,2**(T-t))
          result = jnp.dot(support,marginal)
          results.append(result)
        results.append(jnp.zeros_like(results[0]))
        return jnp.stack(results)

    def apply_fn(params, state, key, inputs, **kwargs):
        inputs = inputs[...,0]
        num_qubits = inputs.shape[-1] - 1
        rbs_idxs = rbs_idxs_fn(num_qubits,num_qubits)
        rbs_idxs = [list(map(list, rbs_idx)) for rbs_idx in rbs_idxs]
        encodings = encoding_fn(inputs)
        unitary_fn = make_general_orthogonal_fn(rbs_idxs, num_qubits)
        unitary = jax.vmap(unitary_fn)(params)
        output_state = jnp.einsum('ijk,lij->lik',unitary,encodings)
        probs = jnp.einsum('ijk,ijk->ijk',output_state,output_state)
        outputs = jax.vmap(measurement_fn)(probs)
        outputs = outputs[...,None]
        return outputs, state

    def init_fn(key, inputs_shape):
      key, init_key = jax.random.split(key)
      num_qubits = inputs_shape[-2] - 1
      rbs_idxs = rbs_idxs_fn(num_qubits,num_qubits)
      rbs_idxs = [list(map(list, rbs_idx)) for rbs_idx in rbs_idxs]
      num_params = len(sum(rbs_idxs, []))
      params = jax.random.uniform(init_key, (inputs_shape[-2],num_params),
                                                  minval=-jnp.pi,
                                                  maxval=jnp.pi)
      return params, None, inputs_shape

    return ModuleFn(apply_fn, init_fn)

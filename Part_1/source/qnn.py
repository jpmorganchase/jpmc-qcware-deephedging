import sys
from typing import Callable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

sys.path.append("..")

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from quantum import make_ortho_fn

# Typing
# -----------------------------------------------------------------------------

Array = jnp.ndarray
Shape = Sequence[int]
Dtype = Union[jnp.float32, jnp.float64]
PRNGKey = Array
Params = Mapping[str, Mapping[str, jnp.ndarray]]
State = Mapping[str, Mapping[str, jnp.ndarray]]
InitializerFn = Callable[[PRNGKey, Shape, Dtype], Array]
Initializer = Callable[..., InitializerFn]
Module = Callable[..., InitializerFn]


class ModuleFn(NamedTuple):
    """A named tuple of apply and init functions."""

    apply: Callable[..., Tuple[Array, State]]
    init: Optional[Callable[..., Tuple[Params, State, Array]]] = None


def add_scope_to_params(scope, params):
    return dict((f"{scope}/{key}", array) for key, array in params.items())


def get_params_by_scope(scope, params):
    return dict(
        (key[len(scope) + 1 :], array)
        for key, array in params.items()
        if key.startswith(scope + "/")
    )


# Initializers
# -----------------------------------------------------------------------------


def constant(
    val: float,
) -> InitializerFn:
    """Initialize with a constant value.

    Args:
        val: The value to initialize with.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        return jnp.broadcast_to(val, shape).astype(dtype)

    return init_fn


def zeros() -> InitializerFn:
    """Initialize with zeros."""
    return constant(0.0)


def ones() -> InitializerFn:
    """Initialize with ones."""
    return constant(1.0)


def uniform(
    minval: float = 0.0,
    maxval: float = 1.0,
) -> InitializerFn:
    """Initialize with a uniform distribution.

    Args:
        minval: The minimum value of the uniform distribution.
        maxval: The maximum value of the uniform distribution.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval, maxval)

    return init_fn


def normal(
    mean: float = 0.0,
    std: float = 1.0,
) -> InitializerFn:
    """Initialize with a normal distribution.

    Args:
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        _mean = lax.convert_element_type(mean, dtype)
        _std = lax.convert_element_type(std, dtype)
        return _mean + _std * jax.random.normal(key, shape, dtype)

    return init_fn


def truncated_normal(
    mean: float = 0.0,
    std: float = 1.0,
) -> InitializerFn:
    """Initialize with a truncated normal distribution.

    Args:
        mean: The mean of the truncated normal distribution.
        std: The standard deviation of the truncated normal distribution.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        _mean = lax.convert_element_type(mean, dtype)
        _std = lax.convert_element_type(std, dtype)
        return _mean + _std * jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype)

    return init_fn


# Modules
# -----------------------------------------------------------------------------


def elementwise(
    elementwise_fn: Callable[[Array], Array],
) -> ModuleFn:
    """Create an elementwise layer from a JAX function.

    Args:
        elementwise_fn: The JAX function to apply to each element.
    """
    return ModuleFn(apply=elementwise_fn)


def linear(
    n_features: int,
    with_bias: bool = True,
    w_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """Create a linear layer.

    Args:
        n_features: The number of features in the output.
        with_bias: Whether to include a bias term.
        w_init: The initializer for the weights.
        b_init: The initializer for the bias.
    """

    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = jnp.dot(inputs, params["w"])

        if with_bias:
            outputs += params["b"]
        return outputs, None

    def init_fn(key, inputs_shape):
        params, state = {}, None
        key, w_key, b_key = jax.random.split(key, 3)
        w_init_ = w_init or truncated_normal(std=1.0 / inputs_shape[-1])
        w_shape = (inputs_shape[-1], n_features)
        params["w"] = w_init_(w_key, w_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features,)
            params["b"] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features,)
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def layer_norm(
    with_scale: bool = True,
    with_bias: bool = True,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """Create a normalization layer.

    Args:
        with_scale: Whether to use a scale parameter.
        with_bias: Whether to include a bias term.
        s_init: The initializer for the scale.
        b_init: The initializer for the bias.
    """

    def init_fn(key, inputs_shape):
        params = {}
        state = None
        s_key, b_key = jax.random.split(key)
        n_features = inputs_shape[-1]
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features,)
            params["s"] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features,)
            params["b"] = b_init_(b_key, b_shape)
        return params, state, inputs_shape

    def apply_fn(params, state, key, inputs, **kwargs):
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        var = jnp.var(inputs, axis=-1, keepdims=True) + 1e-5
        outputs = params["s"] * (inputs - mean) / jnp.sqrt(var) + params["b"]
        return outputs, state

    return ModuleFn(apply_fn, init=init_fn)


def sequential(
    *modules: List[ModuleFn],
) -> ModuleFn:
    """Create a sequential module from a list of modules.

    Args:
        modules: A list of modules.
    """

    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = inputs
        if key is not None:
            key = jax.random.split(key, len(modules))
        else:
            key = len(modules) * [None]
        new_state = dict(("layer_{}".format(idx), None) for idx in range(len(modules)))
        if state is None:
            state = new_state
        for idx, module in enumerate(modules):
            if module.init is not None:
                outputs, new_module_state = module.apply(
                    params["layer_{}".format(idx)],
                    state["layer_{}".format(idx)],
                    key[idx],
                    outputs,
                    **kwargs,
                )
                new_state["layer_{}".format(idx)] = new_module_state
            else:
                outputs = module.apply(outputs)

        state = new_state
        return outputs, state

    def init_fn(key, inputs_shape):
        params = dict(("layer_{}".format(idx), None) for idx in range(len(modules)))
        state = dict(("layer_{}".format(idx), None) for idx in range(len(modules)))
        key = jax.random.split(key, len(modules))
        shape = inputs_shape
        for idx, module in enumerate(modules):
            if module.init is not None:
                module_params, module_state, shape = module.init(key[idx], shape)
                params["layer_{}".format(idx)] = module_params
                state["layer_{}".format(idx)] = module_state
            else:
                shape = module.apply(jnp.zeros(shape)).shape

        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def orthogonalize_weights(weights):
    """Take the current weight matrices for each layer, apply SVD decomposition on each one,
    then transform the singular values, and finally recompose to make the weight matrix orthogonal.
    U,s,V = SVD(W). then all singular values must be ~1.
    Output : update the self.weights matrices.
    Reference : Orthogonal Deep Neural Networks, K.Juia et al. 2019"""
    epsilon = 0.5
    U, s, V = jnp.linalg.svd(weights, full_matrices=False)
    s = jnp.clip(s, 1 / (1 + epsilon), 1 + epsilon)
    # reform with the new singular values
    weights = jnp.dot(U, jnp.dot(jnp.diag(s), V))
    return weights


def orthogonalize_params(params):
    """Take a dictionary of params and orthogonalize the weights"""
    for k1 in params.keys():
        if params[k1] != None:
            for k2 in params[k1].keys():
                if k2.split("/")[-1] == "w":
                    params[k1][k2] = orthogonalize_weights(params[k1][k2])

    return params


def get_pyramid_idxs(num_inputs, num_outputs):
    num_max = max(num_inputs, num_outputs)
    num_min = min(num_inputs, num_outputs)
    if num_max == num_min:
        num_min -= 1
    end_idxs = np.concatenate(
        [np.arange(1, num_max - 1), num_max - np.arange(1, num_min + 1)]
    )
    start_idxs = np.concatenate(
        [
            np.arange(end_idxs.shape[0] + num_min - num_max) % 2,
            np.arange(num_max - num_min),
        ]
    )
    if num_inputs < num_outputs:
        start_idxs = start_idxs[::-1]
        end_idxs = end_idxs[::-1]
    rbs_idxs = [
        np.arange(start_idxs[i], end_idxs[i] + 1).reshape(-1, 2)
        for i in range(len(start_idxs))
    ]
    return rbs_idxs


def get_butterfly_idxs(num_inputs, num_outputs):
    def _get_butterfly_idxs(n):
        if n == 2:
            return np.array([[[0, 1]]])
        else:
            rbs_idxs = _get_butterfly_idxs(n // 2)
            first = np.concatenate([rbs_idxs, rbs_idxs + n // 2], 1)
            last = np.arange(n).reshape(1, 2, n // 2).transpose(0, 2, 1)
            rbs_idxs = np.concatenate([first, last], 0)
            return rbs_idxs

    circuit_dim = int(2 ** np.ceil(np.log2(max(num_inputs, num_outputs))))
    rbs_idxs = _get_butterfly_idxs(circuit_dim)
    if num_inputs < num_outputs:
        rbs_idxs = rbs_idxs[::-1]
    return rbs_idxs


def ortho_linear(
    n_features: int,
    layout: Union[str, List[List[Tuple[int, int]]]] = "butterfly",
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """Create an orthogonal layer from a layout of RBS gates.

    Args:
        n_features: The number of features in the output.
        layout: The layout of the RBS gates.
        normalize_inputs: Whether to normalize the inputs.
        normalize_outputs: Whether to normalize the outputs.
        normalize_stop_gradient: Whether to stop the gradient of the norm.
        with_scale: Whether to use a scale parameter.
        with_bias: Whether to include a bias term.
        t_init: The initializer for the angles.
        s_init: The initializer for the scale.
        b_init: The initializer for the bias.
    """

    def apply_fn(params, state, key, inputs, **kwargs):
        if layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(inputs.shape[-1], n_features)
            circuit_dim = int(2 ** np.ceil(np.log2(max(inputs.shape[-1], n_features))))
        elif layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(inputs.shape[-1], n_features)
            make_unitary = get_pyramid_idxs(inputs.shape[-1], n_features)
            circuit_dim = max(inputs.shape[-1], n_features)
        else:
            rbs_idxs = layout
            circuit_dim = max([max(idxs) for moment in layout for idxs in moment])
        make_unitary = make_ortho_fn(rbs_idxs[::-1], circuit_dim)
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        if inputs.shape[-1] < circuit_dim:
            zeros = jnp.zeros(
                (*inputs.shape[:-1], circuit_dim - inputs.shape[-1]),
            )
            inputs = jnp.concatenate([zeros, inputs], axis=-1)
        unitary = make_unitary(params["t"][::-1])
        outputs = jnp.dot(inputs, unitary.T)[..., -n_features:]
        if normalize_outputs:
            norm = jnp.linalg.norm(outputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            outputs /= norm
        if with_scale:
            outputs *= params["s"]
        if with_bias:
            outputs += params["b"]
        return outputs, None

    def init_fn(key, inputs_shape):
        if layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(inputs_shape[-1], n_features)
        elif layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(inputs_shape[-1], n_features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles,)
        params["t"] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features,)
            params["s"] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features,)
            params["b"] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features,)
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def ortho_linear_noisy(
    n_features: int,
    noise_scale: float = 0.01,
    layout: Union[str, List[List[Tuple[int, int]]]] = "butterfly",
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """Create an orthogonal layer from a layout of RBS gates.
    Args:
        n_features: The number of features in the output.
        layout: The layout of the RBS gates.
        normalize_inputs: Whether to normalize the inputs.
        normalize_outputs: Whether to normalize the outputs.
        normalize_stop_gradient: Whether to stop the gradient of the norm.
        with_scale: Whether to use a scale parameter.
        with_bias: Whether to include a bias term.
        t_init: The initializer for the angles.
        s_init: The initializer for the scale.
        b_init: The initializer for the bias.
    """

    def apply_fn(params, state, key, inputs, **kwargs):
        if layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(inputs.shape[-1], n_features)
            circuit_dim = int(2 ** np.ceil(np.log2(max(inputs.shape[-1], n_features))))
        elif layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(inputs.shape[-1], n_features)
            make_unitary = get_pyramid_idxs(inputs.shape[-1], n_features)
            circuit_dim = max(inputs.shape[-1], n_features)
        else:
            rbs_idxs = layout
            circuit_dim = max([max(idxs) for moment in layout for idxs in moment])
        make_unitary = make_ortho_fn(rbs_idxs[::-1], circuit_dim)
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        if inputs.shape[-1] < circuit_dim:
            zeros = jnp.zeros(
                (*inputs.shape[:-1], circuit_dim - inputs.shape[-1]),
            )
            inputs = jnp.concatenate([zeros, inputs], axis=-1)
        unitary = make_unitary(params["t"][::-1])
        outputs = jnp.dot(inputs, unitary.T)
        if normalize_outputs:
            norm = jnp.linalg.norm(outputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            outputs /= norm
        outputs = jnp.einsum("...i,...i->...i", outputs, outputs)
        outputs = outputs[..., -n_features:]
        if with_scale:
            outputs *= params["s"]
        if with_bias:
            outputs += params["b"]
        key, _ = jax.random.split(key)
        outputs += noise_scale * jax.random.normal(key, outputs.shape)
        return outputs, state

    def init_fn(key, inputs_shape):
        if layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(inputs_shape[-1], n_features)
        elif layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(inputs_shape[-1], n_features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles,)
        params["t"] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features,)
            params["s"] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features,)
            params["b"] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features,)
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def ortho_linear_hardware(
    prepare_circuit,
    run_circuit,
    n_features: int,
    layout: Union[str, List[List[Tuple[int, int]]]] = "butterfly",
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """Create an orthogonal layer from a layout of RBS gates which can be executed on hardware. This layer
    replaces `ortho_linear()` and runs the same operation but on real hardware using `prepare_circuit()` and `run_circuit()` functions.
    Args:
        prepare_circuit: A function to create qiskit circuits given parameters.
        run_circuit: A function which accepts a list of qiskit circuits and returns the result after executing on compatible backend.
        n_features: The number of features in the output.
        layout: The layout of the RBS gates.
        normalize_inputs: Whether to normalize the inputs.
        normalize_outputs: Whether to normalize the outputs.
        normalize_stop_gradient: Whether to stop the gradient of the norm.
        with_scale: Whether to use a scale parameter.
        with_bias: Whether to include a bias term.
        t_init: The initializer for the angles.
        s_init: The initializer for the scale.
        b_init: The initializer for the bias.
    """

    def apply_fn(params, state, key, inputs, **kwargs):
        # Step 1: preprocess the inputs
        if layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(inputs.shape[-1], n_features)
            circuit_dim = int(2 ** np.ceil(np.log2(max(inputs.shape[-1], n_features))))
        elif layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(inputs.shape[-1], n_features)
            make_unitary = get_pyramid_idxs(inputs.shape[-1], n_features)
            circuit_dim = max(inputs.shape[-1], n_features)
        else:
            rbs_idxs = layout
            circuit_dim = max([max(idxs) for moment in layout for idxs in moment])
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        if inputs.shape[-1] < circuit_dim:
            zeros = jnp.zeros(
                (*inputs.shape[:-1], circuit_dim - inputs.shape[-1]),
            )
            inputs = jnp.concatenate([zeros, inputs], axis=-1)
        # Step 2: generate the circuits
        circs = []
        out_shape = inputs.shape[:-1] + (n_features,)
        for input in inputs.reshape(-1, circuit_dim):
            circs.append(prepare_circuit(input, params["t"]))
        # run circuits and truncate to desired number of outputs
        outputs = jnp.array(run_circuit(circs, circuit_dim))[..., -n_features:]
        outputs = outputs.reshape(out_shape)
        if with_scale:
            outputs *= params["s"]
        if with_bias:
            outputs += params["b"]
        return outputs, state

    def init_fn(key, inputs_shape):
        if layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(inputs_shape[-1], n_features)
        elif layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(inputs_shape[-1], n_features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles,)
        params["t"] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features,)
            params["s"] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features,)
            params["b"] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features,)
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)

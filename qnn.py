from typing import (Callable, List, Mapping, NamedTuple, Optional, Sequence,
                    Tuple, Union)

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

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
    apply: Callable[..., Tuple[Array, State]]
    init: Optional[Callable[..., Tuple[Params, State, Array]]] = None


def add_scope_to_params(scope, params):
    return dict((f"{scope}/{key}", array) for key, array in params.items())


def get_params_by_scope(scope, params):
    return dict((key[len(scope) + 1:], array) for key, array in params.items()
                if key.startswith(scope + '/'))


# Initializers
# -----------------------------------------------------------------------------


def constant(val: float, ) -> InitializerFn:
    """ Initialize with a constant value. 
    
    Args:
        val: The value to initialize with.
    """
    def init_fn(key, shape, dtype=jnp.float32):
        return jnp.broadcast_to(val, shape).astype(dtype)

    return init_fn


def zeros() -> InitializerFn:
    """ Initialize with zeros."""
    return constant(0.)


def ones() -> InitializerFn:
    """ Initialize with ones."""
    return constant(1.)


def uniform(
    minval: float = 0.,
    maxval: float = 1.,
) -> InitializerFn:
    """ Initialize with a uniform distribution.

    Args:
        minval: The minimum value of the uniform distribution. 
        maxval: The maximum value of the uniform distribution.
    """
    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval, maxval)

    return init_fn


def normal(
    mean: float = 0.,
    std: float = 1.,
) -> InitializerFn:
    """ Initialize with a normal distribution.
    
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
    mean: float = 0.,
    std: float = 1.,
) -> InitializerFn:
    """ Initialize with a truncated normal distribution.
    
    Args:
        mean: The mean of the truncated normal distribution.
        std: The standard deviation of the truncated normal distribution.
    """
    def init_fn(key, shape, dtype=jnp.float32):
        _mean = lax.convert_element_type(mean, dtype)
        _std = lax.convert_element_type(std, dtype)
        return _mean + _std * jax.random.truncated_normal(
            key, -2., 2., shape, dtype)

    return init_fn


# Modules
# -----------------------------------------------------------------------------


def quax_wrapper(layer_fn):
    """ Create a module from a quax layer. """
    def module(*args, **kwargs):
        init_fn, apply_fn = layer_fn(*args, **kwargs)

        def _apply_fn(params, state, key, inputs, **kwargs):
            outputs = apply_fn(params, inputs, **kwargs)
            return outputs, state

        def _init_fn(key, inputs_shape):
            shape, params = init_fn(key, inputs_shape)
            state = None
            return params, state, shape

        return ModuleFn(_apply_fn, init=_init_fn)

    return module


def haiku_wrapper(layer_fn):
    """ Create a module from a Haiku layer. """
    def module(*args, **kwargs):
        import haiku as hk
        layer = hk.transform_with_state(layer_fn(*args, **kwargs))

        def _apply_fn(params, state, key, inputs, **kwargs):
            outputs, state = layer.apply(params, state, key, inputs, **kwargs)
            return outputs, state

        def _init_fn(key, inputs_shape):
            params, state = layer.init(key, inputs_shape)
            outputs, _ = layer.apply(params, state, key, inputs_shape,
                                     **kwargs)
            shape = outputs.shape
            return params, state, shape

        return ModuleFn(_apply_fn, init=_init_fn)

    return module


def elementwise(elementwise_fn: Callable[[Array], Array], ) -> ModuleFn:
    """ Create an elementwise layer from a JAX function. 
        
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
    """ Create a linear layer.

    Args:
        n_features: The number of features in the output.
        with_bias: Whether to include a bias term.
        w_init: The initializer for the weights.
        b_init: The initializer for the bias.
    """
    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = jnp.dot(inputs, params['w'])

        if with_bias:
            outputs += params['b']
        return outputs, None

    def init_fn(key, inputs_shape):
        params, state = {}, None
        key, w_key, b_key = jax.random.split(key, 3)
        w_init_ = w_init or truncated_normal(std=1. / inputs_shape[-1])
        w_shape = (inputs_shape[-1], n_features)
        params['w'] = w_init_(w_key, w_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features, )
            params['b'] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features, )
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def conv(
    n_spatial_dims: int,
    n_channels: int,
    kernel_shape: Union[int, Shape],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    with_bias: bool = True,
    w_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create a convolutional layer.

    Args:
        n_spatial_dims: The number of spatial dimensions.
        n_channels: The number of channels in the output.
        kernel_shape: The shape of the convolutional kernel.
        stride: The stride of the convolutional kernel.
        padding: The padding of the convolutional kernel.
        with_bias: Whether to include a bias term.
        w_init: The initializer for the weights.
        b_init: The initializer for the bias.
    """

    if isinstance(padding, str):
        padding = padding.upper()
        assert padding in ['VALID',
                           'SAME'], 'padding must be either "VALID" or "SAME"'

    n_dims = n_spatial_dims + 2
    spatial_dims = tuple(range(1, n_dims - 1))

    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape, ) * n_spatial_dims

    if isinstance(stride, int):
        stride = (stride, ) * n_spatial_dims

    def apply_fn(params, state, key, inputs, **kwargs):
        n_dims_input = (0, n_dims - 1) + spatial_dims
        n_dims_kernel = (n_dims - 1, n_dims - 2) + tuple(range(n_dims - 2))
        conv_dims = lax.ConvDimensionNumbers(n_dims_input, n_dims_kernel,
                                             n_dims_input)
        outputs = lax.conv_general_dilated(inputs,
                                           params['w'],
                                           window_strides=stride,
                                           padding=padding,
                                           dimension_numbers=conv_dims)
        if with_bias:
            outputs += params['b']
        return outputs, None

    def init_fn(key, inputs_shape):
        params, state = {}, None
        key, w_key, b_key = jax.random.split(key, 3)
        n_inputs_units = inputs_shape[-1] * np.prod(kernel_shape)
        w_init_ = w_init or truncated_normal(std=1. / n_inputs_units)
        w_shape = kernel_shape + (inputs_shape[-1], n_channels)
        params['w'] = w_init_(w_key, w_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = ((*(len(kernel_shape) * (1, )), n_channels))
            params['b'] = b_init_(b_key, b_shape)
        n_dims_input = ('N', ) + spatial_dims + ('C', )
        n_dims_kernel = spatial_dims + ('I', 'O')
        conv_dims = lax.ConvDimensionNumbers(n_dims_input, n_dims_kernel,
                                             n_dims_input)
        shape = lax.conv_general_shape_tuple(inputs_shape, w_shape, stride,
                                             padding, conv_dims)
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def conv_1D(
    n_channels: int,
    kernel_shape: Union[int, Shape],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    with_bias: bool = True,
    w_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create a 1D convolutional layer.

    Args:
        n_channels: The number of channels in the output.
        kernel_shape: The shape of the convolutional kernel.
        stride: The stride of the convolutional kernel.
        padding: The padding of the convolutional kernel.
        with_bias: Whether to include a bias term.
        w_init: The initializer for the weights.
        b_init: The initializer for the bias.
    """
    return conv(1, n_channels, kernel_shape, stride, padding, with_bias,
                w_init, b_init)


def conv_2D(
    n_channels: int,
    kernel_shape: Union[int, Shape],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    with_bias: bool = True,
    w_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create a 2D convolutional layer.

    Args:
        n_channels: The number of channels in the output.
        kernel_shape: The shape of the convolutional kernel.
        stride: The stride of the convolutional kernel.
        padding: The padding of the convolutional kernel.
        with_bias: Whether to include a bias term.
        w_init: The initializer for the weights.
        b_init: The initializer for the bias.
    """
    return conv(2, n_channels, kernel_shape, stride, padding, with_bias,
                w_init, b_init)


def _pool(
    window_shape: Sequence[int],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    pool_fn: str = 'SUM',
) -> ModuleFn:
    assert pool_fn in ['SUM', 'MAX',
                       'AVG'], "pool_fn must be one of 'SUM', 'MAX', 'AVG'"
    if pool_fn == 'MAX':
        computation = lax.max
        init_value = -jnp.inf
    else:
        computation = lax.add
        init_value = 0.
    if isinstance(stride, int):
        stride = (stride, ) * len(window_shape)

    window_dimensions = (1, ) + window_shape + (1, )
    window_strides = (1, ) + stride + (1, )

    def apply_fn(inputs, **kwargs):

        outputs = lax.reduce_window(inputs, init_value, computation,
                                    window_dimensions, window_strides, padding)
        if pool_fn == 'AVG':
            spatial_shape = inputs.shape[1:-1]
            one = jnp.ones(spatial_shape, dtype=inputs.dtype)
            window_sizes = lax.reduce_window(one, 0., lax.add, window_shape,
                                             stride, padding)
            window_sizes = jnp.expand_dims(window_sizes, [0, -1])
            outputs /= window_sizes
        return outputs

    return ModuleFn(apply_fn)


def avg_pool(
    n_spatial_dims: int,
    window_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
) -> ModuleFn:
    """ Create an average pooling layer.

    Args:
        n_spatial_dims: The number of spatial dimensions.
        window_shape: The shape of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding of the pooling window.
    """
    if isinstance(window_shape, int):
        window_shape = (window_shape, ) * n_spatial_dims
    return _pool(window_shape, stride, padding, 'AVG')


def avg_pool_2D(
    window_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
) -> ModuleFn:
    """ Create a 2D average pooling layer.

    Args:
        window_shape: The shape of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding of the pooling window.
    """
    return avg_pool(2, window_shape, stride, padding)


def max_pool(
    n_spatial_dims: int,
    window_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
) -> ModuleFn:
    """ Create a maximum pooling layer.

    Args:
        n_spatial_dims: The number of spatial dimensions.
        window_shape: The shape of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding of the pooling window.
    """
    if isinstance(window_shape, int):
        window_shape = (window_shape, ) * n_spatial_dims
    return _pool(window_shape, stride, padding, 'MAX')


def max_pool_2D(
    window_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
) -> ModuleFn:
    """ Create a 2D maximum pooling layer.

    Args:
        window_shape: The shape of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding of the pooling window.
    """
    return max_pool(2, window_shape, stride, padding)


def sum_pool(
    n_spatial_dims: int,
    window_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
) -> ModuleFn:
    """ Create a 2D sum pooling layer.

    Args:
        n_spatial_dims: The number of spatial dimensions.
        window_shape: The shape of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding of the pooling window.
    """
    if isinstance(window_shape, int):
        window_shape = (window_shape, ) * n_spatial_dims
    return _pool(n_spatial_dims, window_shape, stride, padding, 'SUM')


def sum_pool_2D(
    window_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
) -> ModuleFn:
    """ Create a 2D sum pooling layer.

    Args:
        window_shape: The shape of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding of the pooling window.
    """
    if isinstance(window_shape, int):
        window_shape = (window_shape, ) * 2
    return _pool(2, window_shape, stride, padding, 'SUM')


def global_pool(n_spatial_dims: int, ) -> ModuleFn:
    """ Create a global average pooling layer.
    
    Args:
        n_spatial_dims: The number of spatial dimensions.
    """
    def apply_fn(inputs, **kwargs):
        return jnp.mean(inputs, axis=tuple(range(1, n_spatial_dims + 1)))

    return ModuleFn(apply_fn)


def global_pool_2D() -> ModuleFn:
    """ Create a 2D global average pooling layer."""
    return global_pool(2)


def flatten() -> ModuleFn:
    """ Create a flattening layer. """
    def apply_fn(inputs, **kwargs):
        return jnp.reshape(inputs, (inputs.shape[0], -1))

    return ModuleFn(apply_fn)


def extract_patches(patch_size: int) -> ModuleFn:
    """ Create a layer that extract patches from an NHWC input. 
    
    Args:
        patch_size: The size of the patch.
    """
    def init_fn(key, inputs_shape):
        n_patches = (inputs_shape[1] * inputs_shape[2]) // (patch_size**2)
        params, state = {}, None
        shape = (
            inputs_shape[0],
            n_patches,
            (patch_size**2) * inputs_shape[-1],
        )
        return params, state, shape

    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = inputs.reshape(
            inputs.shape[0],
            inputs.shape[1] // patch_size,
            patch_size,
            inputs.shape[2] // patch_size,
            patch_size,
            inputs.shape[-1],
        )
        outputs = outputs.swapaxes(2, 3)
        outputs = outputs.reshape(inputs.shape[0], -1,
                                  patch_size * patch_size * inputs.shape[-1])
        return outputs, None

    return ModuleFn(apply_fn, init=init_fn)


def attention(n_features: int, n_heads: int = 1) -> ModuleFn:
    """ Create an attention layer.
    
    Args:
        n_features: The number of features.
        n_heads: The number of heads.
        n_features_per_head: The number of features per head.
    """
    n_features_per_head = n_features // n_heads
    to_qk = linear(n_features, False)
    to_v = linear(n_features, False)

    def apply_fn(params, state, key, inputs, **kwargs):
        q_params = get_params_by_scope('query', params)
        q, _ = to_qk.apply(q_params, None, None, inputs)
        k_params = get_params_by_scope('key', params)
        k, _ = to_qk.apply(k_params, None, None, inputs)
        v_params = get_params_by_scope('value', params)
        v, _ = to_v.apply(v_params, None, None, inputs)
        q = q.reshape(*q.shape[:-1], n_heads, n_features_per_head)
        k = k.reshape(*k.shape[:-1], n_heads, n_features_per_head)
        v = v.reshape(*v.shape[:-1], n_heads, n_features_per_head)
        dots = jnp.einsum("...thd,...Thd->...htT", q, k)
        w = jax.nn.softmax(dots / np.sqrt(n_features_per_head), axis=-1)
        outputs = jnp.einsum("...htT,...Thd->...thd", w, v)
        outputs = outputs.reshape(*outputs.shape[:2], -1)
        return outputs, state

    def init_fn(key, inputs_shape):
        state = None
        params = {}
        q_key, k_key, v_key = jax.random.split(key, 3)
        q_params, _, _ = to_qk.init(q_key, inputs_shape)
        k_params, _, _ = to_qk.init(k_key, inputs_shape)
        v_params, _, shape = to_v.init(v_key, inputs_shape)
        params.update(add_scope_to_params('query', q_params))
        params.update(add_scope_to_params('key', k_params))
        params.update(add_scope_to_params('value', v_params))
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def layer_norm(
    with_scale: bool = True,
    with_bias: bool = True,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create a normalization layer. 
    
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
            s_shape = (n_features, )
            params['s'] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features, )
            params['b'] = b_init_(b_key, b_shape)
        return params, state, inputs_shape

    def apply_fn(params, state, key, inputs, **kwargs):
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        var = jnp.var(inputs, axis=-1, keepdims=True) + 1e-5
        outputs = params['s'] * (inputs - mean) / jnp.sqrt(var) + params['b']
        return outputs, state

    return ModuleFn(apply_fn, init=init_fn)


def transformer(
    n_features: int,
    n_heads: int,
    depth: int,
    n_features_hidden: Optional[int] = None,
) -> ModuleFn:
    """ Create a transformer layer.
    
    Args:
        n_features: The number of features.
        n_heads: The number of heads.
        depth: The number of layers.
        n_features_per_head: The number of features per head.
        n_features_hidden: The number of features in the hidden layer.    
    """
    n_features_hidden = n_features_hidden or n_features
    gelu = elementwise(jax.nn.gelu)
    mlp_layers = []
    attn_layers = []
    mlp_norm_layers = []
    attn_norm_layers = []
    for _ in range(depth):
        mlp_layers.append(
            sequential(linear(n_features_hidden, True), gelu,
                       linear(n_features, True)))
        attn_layers.append(attention(n_features, n_heads))
        mlp_norm_layers.append(layer_norm(with_scale=True, with_bias=True))
        attn_norm_layers.append(layer_norm(with_scale=True, with_bias=True))

    def init_fn(key, inputs_shape):
        params, state = {}, None
        for i in range(depth):
            key, attn_key, attn_norm_key, mlp_key, mlp_norm_key = jax.random.split(
                key, 5)
            attn_params, _, inputs_shape = attn_layers[i].init(
                attn_key,
                inputs_shape,
            )
            attn_norm_params, _, inputs_shape = attn_norm_layers[i].init(
                attn_norm_key, inputs_shape)
            params.update(add_scope_to_params('attn_{}'.format(i),
                                              attn_params))
            params.update(
                add_scope_to_params('attn_norm_{}'.format(i),
                                    attn_norm_params))

            mlp_params, _, inputs_shape = mlp_layers[i].init(
                mlp_key, inputs_shape)
            params.update(add_scope_to_params('mlp_{}'.format(i), mlp_params))
            mlp_norm_params, _, inputs_shape = mlp_norm_layers[i].init(
                mlp_norm_key, inputs_shape)
            params.update(
                add_scope_to_params('mlp_norm_{}'.format(i), mlp_norm_params))
        return params, state, inputs_shape

    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = inputs
        for i in range(depth):
            attn_params = get_params_by_scope('attn_{}'.format(i), params)
            attn_norm_params = get_params_by_scope('attn_norm_{}'.format(i),
                                                   params)
            mlp_params = get_params_by_scope('mlp_{}'.format(i), params)
            mlp_norm_params = get_params_by_scope('mlp_norm_{}'.format(i),
                                                  params)
            outputs = attn_norm_layers[i].apply(attn_norm_params, None, None,
                                                outputs)[0]
            outputs = attn_layers[i].apply(attn_params, None, None,
                                           outputs)[0] + outputs
            outputs = mlp_norm_layers[i].apply(mlp_norm_params, None, None,
                                               outputs)[0]
            outputs = mlp_layers[i].apply(mlp_params, None, None,
                                          outputs)[0] + outputs
        return outputs, None

    return ModuleFn(apply_fn, init=init_fn)


def visual_transformer(
    n_features: int,
    patch_size: int = 7,
    n_heads: int = 1,
    depth: int = 8,
    n_features_hidden: Optional[int] = None,
) -> ModuleFn:
    """ Create a visual transformer layer.
    
    Args:
        n_features: The number of features.
        patch_size: The size of the patch.
        n_heads: The number of heads.
        depth: The number of layers.
        n_features_hidden: The number of features in the hidden layer.
    """
    n_features_hidden = n_features_hidden or n_features
    embed = linear(n_features, True)
    transform = transformer(n_features, n_heads=n_heads, depth=depth)
    to_patch = extract_patches(patch_size)

    def init_fn(key, inputs_shape):
        n_patches = (inputs_shape[1] * inputs_shape[2]) // (patch_size**2)
        params, state = {}, None
        embed_key, transformer_key, pos_key, cls_key = jax.random.split(key, 4)
        _, _, patches_shape = to_patch.init(None, inputs_shape)
        embed_params, _, embed_shape = embed.init(embed_key, patches_shape)
        transform_params, _, shape = transform.init(transformer_key,
                                                    embed_shape)
        params.update(add_scope_to_params('embed', embed_params))
        params.update(add_scope_to_params('transformer', transform_params))
        params['pos_embed'] = normal()(pos_key, (1, n_patches + 1, n_features))
        params['cls_token'] = normal()(cls_key, (1, 1, n_features))
        return params, state, shape

    def apply_fn(params, state, key, inputs, **kwargs):
        patches, _ = to_patch.apply(None, None, None, inputs)
        embed_params = get_params_by_scope('embed', params)
        patches, _ = embed.apply(embed_params, None, None, patches)
        tokens = jnp.tile(params['cls_token'], [inputs.shape[0], 1, 1])
        patches = jnp.concatenate([tokens, patches], axis=1)
        patches += params['pos_embed']
        transform_params = get_params_by_scope('transformer', params)
        patches = transform.apply(transform_params, None, None, patches)[0]
        return patches[:, 0], None

    return ModuleFn(apply_fn, init=init_fn)


def sequential(*modules: List[ModuleFn], ) -> ModuleFn:
    """ Create a sequential module from a list of modules.

    Args:
        modules: A list of modules.
    """
    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = inputs
        if key is not None:
            key = jax.random.split(key, len(modules))
        else:
            key = len(modules) * [None]
        new_state = dict(
            ('layer_{}'.format(idx), None) for idx in range(len(modules)))
        if state is None:
            state = new_state
        for idx, module in enumerate(modules):
            if module.init is not None:
                outputs, new_module_state = module.apply(
                    params['layer_{}'.format(idx)],
                    state['layer_{}'.format(idx)],
                    key[idx],
                    outputs,
                    **kwargs,
                )
                new_state['layer_{}'.format(idx)] = new_module_state
            else:
                outputs = module.apply(outputs)

        state = new_state
        return outputs, state

    def init_fn(key, inputs_shape):
        params = dict(
            ('layer_{}'.format(idx), None) for idx in range(len(modules)))
        state = dict(
            ('layer_{}'.format(idx), None) for idx in range(len(modules)))
        key = jax.random.split(key, len(modules))
        shape = inputs_shape
        for idx, module in enumerate(modules):
            if module.init is not None:
                module_params, module_state, shape = module.init(
                    key[idx], shape)
                params['layer_{}'.format(idx)] = module_params
                state['layer_{}'.format(idx)] = module_state
            else:
                shape = module.apply(jnp.zeros(shape)).shape

        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def parallel(*modules: List[ModuleFn], ) -> ModuleFn:
    """ Create a parallel module from a list of modules.

    Args:
        modules: A list of modules.
    """
    def apply_fn(params, state, key, inputs, **kwargs):
        assert isinstance(
            inputs,
            (list, tuple)), "Parallel module requires a list of inputs."
        outputs = inputs
        if key is not None:
            key = jax.random.split(key, len(modules))
        else:
            key = len(modules) * [None]
        new_state = len(modules) * [None]
        for idx, module in enumerate(modules):
            if module.init is not None:
                outputs, new_module_state = module.apply(
                    params[idx],
                    state[idx],
                    key[idx],
                    outputs,
                    **kwargs,
                )
                new_state[idx] = new_module_state
            else:
                outputs = module.apply(outputs)

        state = new_state
        return outputs, state

    def init_fn(key, inputs_shape):
        params, state = len(modules) * [None], len(modules) * [None]
        key = jax.random.split(key, len(modules))
        outputs_shape = []
        for idx, module in enumerate(modules):
            if module.init is not None:
                module_params, module_state, shape = module.init(
                    key[idx], inputs_shape)
                params[idx] = module_params
                state[idx] = module_state
            else:
                shape = inputs_shape
            outputs_shape.append(shape)
        shape = outputs_shape
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def _make_orthogonal_fn(rbs_idxs, size):
    num_thetas = sum(map(len, rbs_idxs))
    rbs_idxs = [list(map(list, rbs_idx)) for rbs_idx in rbs_idxs]
    len_idxs = np.cumsum([0] + list(map(len, rbs_idxs)))

    def _get_rbs_unitary(theta):
        """ Returns the unitary matrix for a single RBS gate. """
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        unitary = jnp.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ])
        unitary = unitary.transpose(*[*range(2, unitary.ndim), 0, 1])
        return unitary

    def _get_rbs_unitary_grad(theta):
        """ Returns the unitary matrix for a single RBS gate. """
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        unitary = jnp.array([
            [-sin_theta, -cos_theta],
            [cos_theta, -sin_theta],
        ])
        unitary = unitary.transpose(*[*range(2, unitary.ndim), 0, 1])
        return unitary

    @jax.custom_jvp
    def _get_parallel_rbs_unitary(thetas):
        """ Returns the unitary matrix for parallel RBS gates. """
        unitaries = []
        for i, idxs in enumerate(rbs_idxs):
            idxs = sum(idxs, [])
            sub_thetas = thetas[len_idxs[i]:len_idxs[i + 1]]
            rbs_blocks = _get_rbs_unitary(sub_thetas)
            eye_block = jnp.eye(size - len(idxs), dtype=thetas.dtype)
            permutation = idxs + [i for i in range(size) if i not in idxs]
            permutation = np.argsort(permutation)
            unitary = jax.scipy.linalg.block_diag(*rbs_blocks, eye_block)
            unitary = unitary[permutation][:, permutation]
            unitaries.append(unitary)
        unitaries = jnp.stack(unitaries)
        return unitaries

    @_get_parallel_rbs_unitary.defjvp
    def get_parallel_rbs_unitary_jvp(primals, tangents):
        thetas, = primals
        thetas_dot, = tangents
        unitaries = []
        unitaries_dot = []
        for i, idxs in enumerate(rbs_idxs):
            idxs = sum(idxs, [])
            sub_thetas = thetas[len_idxs[i]:len_idxs[i + 1]]
            sub_thetas_dot = thetas_dot[len_idxs[i]:len_idxs[i + 1]]
            rbs_blocks = _get_rbs_unitary(sub_thetas)
            rbs_blocks_grad = _get_rbs_unitary_grad(sub_thetas)
            rbs_blocks_dot = sub_thetas_dot[..., None, None] * rbs_blocks_grad
            eye_block = jnp.eye(size - len(idxs), dtype=thetas.dtype)
            zero_block = jnp.zeros_like(eye_block)
            permutation = idxs + [i for i in range(size) if i not in idxs]
            permutation = np.argsort(permutation)
            unitary = jax.scipy.linalg.block_diag(*rbs_blocks, eye_block)
            unitary_dot = jax.scipy.linalg.block_diag(*rbs_blocks_dot,
                                                      zero_block)
            unitary = unitary[permutation][:, permutation]
            unitary_dot = unitary_dot[permutation][:, permutation]
            unitaries.append(unitary)
            unitaries_dot.append(unitary_dot)
        primal_out = jnp.stack(unitaries)
        tangent_out = jnp.stack(unitaries_dot)
        return primal_out, tangent_out

    def orthogonal_fn(thetas, precision=None):
        """ Returns the unitary matrix for a sequence of parallel RBS gates. """
        assert thetas.shape[0] == num_thetas, "Wrong number of thetas."
        unitaries = _get_parallel_rbs_unitary(thetas)
        unitary = jnp.linalg.multi_dot(unitaries[::-1], precision=precision)
        return unitary

    return orthogonal_fn


def _get_pyramid_idxs(num_inputs, num_outputs):
    num_max = max(num_inputs, num_outputs)
    num_min = min(num_inputs, num_outputs)
    if num_max == num_min:
        num_min -= 1
    end_idxs = np.concatenate(
        [np.arange(1, num_max - 1), num_max - np.arange(1, num_min + 1)])
    start_idxs = np.concatenate([
        np.arange(end_idxs.shape[0] + num_min - num_max) % 2,
        np.arange(num_max - num_min)
    ])
    if num_inputs < num_outputs:
        start_idxs = start_idxs[::-1]
        end_idxs = end_idxs[::-1]
    rbs_idxs = [
        np.arange(start_idxs[i], end_idxs[i] + 1).reshape(-1, 2)
        for i in range(len(start_idxs))
    ]
    return rbs_idxs


def _get_butterfly_idxs(num_inputs, num_outputs):
    def _get_butterfly_idxs(n):
        if n == 2:
            return np.array([[[0, 1]]])
        else:
            rbs_idxs = _get_butterfly_idxs(n // 2)
            first = np.concatenate([rbs_idxs, rbs_idxs + n // 2], 1)
            last = np.arange(n).reshape(1, 2, n // 2).transpose(0, 2, 1)
            rbs_idxs = np.concatenate([first, last], 0)
            return rbs_idxs

    circuit_dim = int(2**np.ceil(np.log2(max(num_inputs, num_outputs))))
    rbs_idxs = _get_butterfly_idxs(circuit_dim)
    if num_inputs < num_outputs:
        rbs_idxs = rbs_idxs[::-1]
    return rbs_idxs


def ortho_linear(
    n_features: int,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create an orthogonal layer from a layout of RBS gates.

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
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(inputs.shape[-1], n_features)
            circuit_dim = int(2**np.ceil(
                np.log2(max(inputs.shape[-1], n_features))))
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(inputs.shape[-1], n_features)
            make_unitary = _get_pyramid_idxs(inputs.shape[-1], n_features)
            circuit_dim = max(inputs.shape[-1], n_features)
        else:
            rbs_idxs = layout
            circuit_dim = max(
                [max(idxs) for moment in layout for idxs in moment])
        make_unitary = _make_orthogonal_fn(rbs_idxs, circuit_dim)
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        if inputs.shape[-1] < circuit_dim:
            zeros = jnp.zeros(
                (*inputs.shape[:-1], circuit_dim - inputs.shape[-1]), )
            inputs = jnp.concatenate([zeros, inputs], axis=-1)
        unitary = make_unitary(params['t'])
        outputs = jnp.dot(inputs, unitary.T)[..., -n_features:]
        if normalize_outputs:
            norm = jnp.linalg.norm(outputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            outputs /= norm
        if with_scale:
            outputs *= params['s']
        if with_bias:
            outputs += params['b']
        return outputs, None

    def init_fn(key, inputs_shape):
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(inputs_shape[-1], n_features)
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(inputs_shape[-1], n_features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles, )
        params['t'] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features, )
            params['s'] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features, )
            params['b'] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features, )
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)

def ortho_linear_noisy(
    n_features: int,
    noise_scale: float=0.1,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create an orthogonal layer from a layout of RBS gates.

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
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(inputs.shape[-1], n_features)
            circuit_dim = int(2**np.ceil(
                np.log2(max(inputs.shape[-1], n_features))))
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(inputs.shape[-1], n_features)
            make_unitary = _get_pyramid_idxs(inputs.shape[-1], n_features)
            circuit_dim = max(inputs.shape[-1], n_features)
        else:
            rbs_idxs = layout
            circuit_dim = max(
                [max(idxs) for moment in layout for idxs in moment])
        make_unitary = _make_orthogonal_fn(rbs_idxs, circuit_dim)
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        if inputs.shape[-1] < circuit_dim:
            zeros = jnp.zeros(
                (*inputs.shape[:-1], circuit_dim - inputs.shape[-1]), )
            inputs = jnp.concatenate([zeros, inputs], axis=-1)
        unitary = make_unitary(params['t'])
        outputs = jnp.dot(inputs, unitary.T)[..., -n_features:]
        if normalize_outputs:
            norm = jnp.linalg.norm(outputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            outputs /= norm
        if with_scale:
            outputs *= params['s']
        if with_bias:
            outputs += params['b']
        key, _ = jax.random.split(key)
        outputs += noise_scale*jax.random.normal(key, outputs.shape)
        return outputs, state

    def init_fn(key, inputs_shape):
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(inputs_shape[-1], n_features)
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(inputs_shape[-1], n_features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles, )
        params['t'] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features, )
            params['s'] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features, )
            params['b'] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features, )
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def compound_2D(
    n_features: int,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create an orthogonal layer from a layout of RBS gates.

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
    def _apply_compound(inputs, unitary):

        length = inputs.shape[-2]
        idxs1 = sum([(length * n_features) * [k, k, length + l, length + l]
                     for k in range(length) for l in range(n_features)], [])
        idxs2 = sum((length * n_features) * [[i, length + j, i, length + j]
                                             for i in range(length)
                                             for j in range(n_features)], [])
        compound = unitary[idxs1, idxs2].reshape(length * n_features,
                                                 length * n_features, 2, 2)
        compound = jnp.linalg.det(compound)
        outputs = inputs.reshape(-1, length * n_features) @ compound.T
        outputs = outputs.reshape(-1, length, n_features)
        return outputs

    def apply_fn(params, state, key, inputs, **kwargs):
        length = inputs.shape[-2]
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(length + n_features,
                                           length + n_features)
            circuit_dim = int(2**np.ceil(np.log2(length + n_features)))
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(length + n_features,
                                         length + n_features)
            circuit_dim = length + n_features
        else:
            rbs_idxs = layout
            circuit_dim = max(
                [max(idxs) for moment in layout for idxs in moment])
        make_unitary = _make_orthogonal_fn(rbs_idxs, circuit_dim)
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        unitary = make_unitary(params['t'])
        outputs = _apply_compound(inputs, unitary)
        if normalize_outputs:
            norm = jnp.linalg.norm(outputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            outputs /= norm
        if with_scale:
            outputs *= params['s']
        if with_bias:
            outputs += params['b']
        return outputs, None

    def init_fn(key, inputs_shape):
        length = inputs_shape[-2]
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(inputs_shape[-1],
                                           length + n_features)
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(inputs_shape[-1], length + n_features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles, )
        params['t'] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (n_features, )
            params['s'] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (n_features, )
            params['b'] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (n_features, )
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def compound_patches():
    import itertools

    def get_compound_from_seq_features(inputs):
        length, features = inputs.shape[-2:]
        compound_idxs = list(
            itertools.combinations(range(length + features), 2))
        subset_idxs = [
            compound_idxs.index((i, length + j)) for i in range(length)
            for j in range(features)
        ]
        compound_inputs = jnp.zeros((*inputs.shape[:-2], len(compound_idxs)))
        compound_inputs = compound_inputs.at[..., subset_idxs].set(
            inputs.reshape(*inputs.shape[:-2], -1))
        return compound_inputs

    def apply_fn(inputs, **kwargs):
        return get_compound_from_seq_features(inputs)

    return ModuleFn(apply_fn)


def compound_transform_patches(
    length,
    features,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
    convert_inputs: bool = False,
) -> ModuleFn:
    """ Create an orthogonal layer from a layout of RBS gates.

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
    import itertools

    def get_compound_from_matrix(matrix, k=2):
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]
        dim = matrix.shape[0]
        subsets = list(itertools.combinations(range(dim), k))
        submatrices = [
            matrix[tuple(zip(*itertools.product(sub1, sub2)))]
            for sub1 in subsets for sub2 in subsets
        ]
        compounds = jnp.stack(submatrices).reshape(len(subsets), len(subsets),
                                                   2, 2)
        compound_matrix = jnp.linalg.det(compounds)
        return compound_matrix

    def apply_fn(params, state, key, inputs, **kwargs):
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(length + features,
                                           length + features)
            circuit_dim = int(2**np.ceil(np.log2(length + features)))
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(length + features, length + features)
            circuit_dim = length + features
        else:
            rbs_idxs = layout
            circuit_dim = max(
                [max(idxs) for moment in layout for idxs in moment])
        make_unitary = _make_orthogonal_fn(rbs_idxs, circuit_dim)
        if normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            inputs /= norm
        unitary = make_unitary(params['t'])
        from scipy.special import binom
        size = int(binom(length + features, 2))
        compound_unitary = get_compound_from_matrix(unitary)[-size:, -size:]
        outputs = jnp.dot(inputs, compound_unitary.T)
        if normalize_outputs:
            norm = jnp.linalg.norm(outputs, axis=-1)[..., None]
            if normalize_stop_gradient:
                norm = lax.stop_gradient(norm)
            outputs /= norm
        if with_scale:
            outputs *= params['s']
        if with_bias:
            outputs += params['b']
        return outputs, None

    def init_fn(key, inputs_shape):
        from scipy.special import binom
        if layout == 'butterfly':
            rbs_idxs = _get_butterfly_idxs(length + features,
                                           length + features)
        elif layout == 'pyramid':
            rbs_idxs = _get_pyramid_idxs(length + features, length + features)
        else:
            rbs_idxs = layout
        n_angles = sum(map(len, rbs_idxs))
        params, state = {}, None
        key, t_key, b_key, s_key = jax.random.split(key, 4)
        t_init_ = t_init or uniform(-np.pi, np.pi)
        t_shape = (n_angles, )
        params['t'] = t_init_(t_key, t_shape)
        if with_scale:
            s_init_ = s_init or ones()
            s_shape = (int(binom(length + features, 2)), )
            params['s'] = s_init_(s_key, s_shape)
        if with_bias:
            b_init_ = b_init or zeros()
            b_shape = (int(binom(length + features, 2)), )
            params['b'] = b_init_(b_key, b_shape)
        shape = inputs_shape[:-1] + (int(binom(length + features, 2)), )
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def _unitary_wrapper(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        unitary = func(*args, **kwargs)
        if unitary.ndim > 2:
            perm = [*range(2, unitary.ndim), 0, 1]
            unitary = unitary.transpose(*perm)
        return unitary.astype(jnp.complex64)

    return wrapper


@_unitary_wrapper
def rot_x(theta):
    cos_t, sin_t = jnp.cos(theta / 2.), jnp.sin(theta / 2.)
    a = d = cos_t
    b = c = -1.j * sin_t
    unitary = jnp.array([
        [a, b],
        [c, d],
    ])
    return unitary


@_unitary_wrapper
def rot_y(theta):
    cos_t, sin_t = jnp.cos(theta / 2.), jnp.sin(theta / 2.)
    a = d = cos_t
    b = -sin_t
    c = sin_t
    unitary = jnp.array([
        [a, b],
        [c, d],
    ])
    return unitary


@_unitary_wrapper
def rot_z(theta):
    d = jnp.exp(1.j * theta / 2.)
    a = d.conj()
    b = c = jnp.zeros_like(a)
    unitary = jnp.array([
        [a, b],
        [c, d],
    ])
    return unitary


@_unitary_wrapper
def controlled_z():
    unitary = jnp.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., -1.],
    ])
    return unitary


def controlled_z_func(kets, wire):
    n_wires = int(np.log2(kets.shape[-1]))
    idx = 2**(wire % n_wires) - 1
    kets = kets.at[..., idx].multiply(-1)
    return kets


def _multi_kron(arrays, order, i, j):
    if i == j:
        return arrays[i]
    else:
        return jnp.kron(_multi_kron(arrays, order, i, order[i, j]),
                        _multi_kron(arrays, order, order[i, j] + 1, j))


def tensordot_unitary(unitaries):
    from jax._src.third_party.numpy.linalg import _multi_dot_matrix_chain_order
    if len(unitaries) == 1:
        unitary = unitaries[0]
    elif len(unitaries) == 2:
        unitary = jnp.kron(unitaries[0], unitaries[1])
    else:
        order = _multi_dot_matrix_chain_order(unitaries)
        unitary = _multi_kron(unitaries, order, 0, len(unitaries) - 1)
    return unitary


def variational_layer(n_layers, data_reupload=False):
    def _block(theta_y, theta_z):
        y = tensordot_unitary(rot_y(theta_y))
        z = tensordot_unitary(rot_z(theta_z))
        unitary = z @ y
        for wire in range(len(theta_y)):
            unitary = jax.vmap(controlled_z_func, (0, None))(unitary, wire)
        return unitary

    def init_fn(key, inputs_shape):
        n_wires = inputs_shape[-1]
        params = {}
        state = None
        y_key, z_key = jax.random.split(key)
        params['y'] = jax.random.uniform(y_key, (n_layers, n_wires),
                                         minval=-np.pi,
                                         maxval=np.pi)
        params['z'] = jax.random.uniform(z_key, (n_layers, n_wires),
                                         minval=-np.pi,
                                         maxval=np.pi)
        return params, state, inputs_shape

    def apply_fn(params, state, key, inputs, **kwargs):
        n_wires = inputs.shape[-1]
        x = jax.vmap(tensordot_unitary)(rot_x(inputs))
        blocks = jax.vmap(_block)(params['y'], params['z'])
        if not data_reupload:
            result = jnp.linalg.multi_dot(list(blocks)[::-1] +
                                          [x[:, 0, :].T]).T
        else:

            def new_block(block):
                block = jnp.matmul(block, x.T)
                return block

            blocks = jax.vmap(new_block)(blocks).transpose(3, 0, 1, 2)
            result = jax.vmap(jnp.linalg.multi_dot)(blocks)[:, :, 0]
        probs = result * result.conj()
        probs = probs.reshape((result.shape[0], ) + n_wires * (2, )).real
        final_probs = []
        for i in range(n_wires):
            _probs = probs.sum(
                [len(inputs.shape) + j - 1 for j in range(n_wires) if j != i])
            final_probs.append(_probs)
        result = jnp.stack(final_probs, axis=-2)
        result = result @ jnp.array([1., -1.])
        return result, state

    return ModuleFn(apply_fn, init=init_fn)


def variational_conv(n_layers, data_reupload=False):
    kernel = variational_layer(n_layers, data_reupload)
    init_fn = kernel.init

    def apply_fn(params, state, key, inputs, **kwargs):
        batch_size = inputs.shape[0]
        n_wires = inputs.shape[-1]
        inputs = inputs.reshape(-1, n_wires)
        outputs, _ = kernel.apply(params, state, key, inputs)
        outputs = outputs.reshape(batch_size, -1, n_wires)
        return outputs, state

    return ModuleFn(apply_fn, init=init_fn)


def ortho_conv(
    n_spatial_dims: int,
    n_channels: int,
    kernel_shape: Union[int, Shape],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create an orthogonal convolutional layer from a layout of RBS gates.

    Args:
        n_spatial_dims: The number of spatial dimensions.
        n_channels: The number of channels in the output.
        kernel_shape: The shape of the kernel.
        stride: The stride of the convolution.
        padding: The padding of the convolution.
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

    if isinstance(padding, str):
        padding = padding.upper()
        assert padding in ['VALID',
                           'SAME'], 'padding must be either "VALID" or "SAME"'

    n_dims = n_spatial_dims + 2
    spatial_dims = tuple(range(1, n_dims - 1))
    n_dims_input = (0, n_dims - 1) + spatial_dims
    n_dims_kernel = (n_dims - 1, n_dims - 2) + tuple(range(n_dims - 2))
    conv_dims = lax.ConvDimensionNumbers(n_dims_input, n_dims_kernel,
                                         n_dims_input)
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape, ) * n_spatial_dims

    if isinstance(stride, int):
        stride = (stride, ) * n_spatial_dims

    ortho_layer = ortho_linear(n_channels,
                               layout=layout,
                               normalize_inputs=normalize_inputs,
                               normalize_outputs=normalize_outputs,
                               normalize_stop_gradient=normalize_stop_gradient,
                               with_scale=with_scale,
                               with_bias=with_bias,
                               t_init=t_init,
                               s_init=s_init,
                               b_init=b_init)

    def apply_fn(params, state, key, inputs, **kwargs):
        conv_inputs = lax.conv_general_dilated_patches(
            inputs, kernel_shape, stride, padding, dimension_numbers=conv_dims)
        conv_outputs, state = ortho_layer.apply(params, state, key,
                                                conv_inputs)
        outputs = conv_outputs.reshape(*inputs.shape[:-1], -1)
        return outputs, state

    def init_fn(key, inputs_shape):
        conv_shape = lax.conv_general_dilated_patches(
            jnp.ones(inputs_shape),
            kernel_shape,
            stride,
            padding,
            dimension_numbers=conv_dims).shape
        return ortho_layer.init(key, conv_shape)

    return ModuleFn(apply_fn, init=init_fn)


def ortho_conv_1D(
    n_channels: int,
    kernel_shape: Union[int, Shape],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create a 1D orthogonal convolutional layer from a layout of RBS gates.

    Args:
        n_channels: The number of channels in the output.
        kernel_shape: The shape of the convolutional kernel.
        stride: The stride of the convolutional kernel.
        padding: The padding of the convolutional kernel.
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
    return ortho_conv(1, n_channels, kernel_shape, stride, padding, layout,
                      normalize_inputs, normalize_outputs,
                      normalize_stop_gradient, with_scale, with_bias, t_init,
                      s_init, b_init)


def ortho_conv_2D(
    n_channels: int,
    kernel_shape: Union[int, Shape],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
    normalize_inputs: bool = False,
    normalize_outputs: bool = True,
    normalize_stop_gradient: bool = True,
    with_scale: bool = True,
    with_bias: bool = True,
    t_init: Optional[InitializerFn] = None,
    s_init: Optional[InitializerFn] = None,
    b_init: Optional[InitializerFn] = None,
) -> ModuleFn:
    """ Create a 2D orthogonal convolutional layer from a layout of RBS gates.

    Args:
        n_channels: The number of channels in the output.
        kernel_shape: The shape of the convolutional kernel.
        stride: The stride of the convolutional kernel.
        padding: The padding of the convolutional kernel.
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
    return ortho_conv(2, n_channels, kernel_shape, stride, padding, layout,
                      normalize_inputs, normalize_outputs,
                      normalize_stop_gradient, with_scale, with_bias, t_init,
                      s_init, b_init)


def ortho_attention_v0(
    n_features: int,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
) -> ModuleFn:
    """ Create an attention layer.
    
    Args:
        n_features: The number of features.
        layout: The layout of the RBS gates.
    """
    to_qk = ortho_linear(n_features,
                         layout=layout,
                         with_scale=False,
                         with_bias=False,
                         normalize_inputs=True,
                         normalize_outputs=False)
    to_v = ortho_linear(n_features,
                        layout=layout,
                        with_scale=True,
                        with_bias=False,
                        normalize_inputs=True,
                        normalize_outputs=False)

    def apply_fn(params, state, key, inputs, **kwargs):
        q_params = get_params_by_scope('query', params)
        q, _ = to_qk.apply(q_params, None, None, inputs)
        k_params = get_params_by_scope('key', params)
        k, _ = to_qk.apply(k_params, None, None, inputs)
        v_params = get_params_by_scope('value', params)
        v, _ = to_v.apply(v_params, None, None, inputs)
        dots = jnp.einsum("...thd,...Thd->...htT", q, k)
        w = jnp.power(dots, 2)
        w /= jnp.sum(w, axis=-1, keepdims=True)
        outputs = jnp.einsum("...htT,...Thd->...thd", w, v)
        return outputs, None

    def init_fn(key, inputs_shape):
        state = None
        params = {}
        q_key, k_key, v_key = jax.random.split(key, 3)
        q_params, _, _ = to_qk.init(q_key, inputs_shape)
        k_params, _, _ = to_qk.init(k_key, inputs_shape)
        v_params, _, shape = to_v.init(v_key, inputs_shape)
        params.update(add_scope_to_params('query', q_params))
        params.update(add_scope_to_params('key', k_params))
        params.update(add_scope_to_params('value', v_params))
        return params, state, shape

    return ModuleFn(apply_fn, init=init_fn)


def ortho_transformer_v0(
    n_features: int,
    n_heads: int,
    depth: int,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
) -> ModuleFn:
    """ Create a quantum transformer layer.
    
    Args:
        n_features: The number of features.
        n_heads: The number of heads.
        depth: The number of layers.
        layout: The layout of the RBS gates.
    """
    gelu = elementwise(jax.nn.gelu)
    mlp_layers = []
    attn_layers = []
    mlp_norm_layers = []
    attn_norm_layers = []
    for _ in range(depth):
        mlp_layers.append(
            sequential(
                ortho_linear(n_features, with_bias=True, layout=layout),
                gelu,
                ortho_linear(n_features, with_bias=True, layout=layout),
            ))
        attn_layers.append(attention(n_features, n_heads))
        mlp_norm_layers.append(layer_norm(with_scale=True, with_bias=True))
        attn_norm_layers.append(layer_norm(with_scale=True, with_bias=True))

    def init_fn(key, inputs_shape):
        params, state = {}, None
        for i in range(depth):
            key, attn_key, attn_norm_key, mlp_key, mlp_norm_key = jax.random.split(
                key, 5)
            attn_params, _, inputs_shape = attn_layers[i].init(
                attn_key,
                inputs_shape,
            )
            attn_norm_params, _, inputs_shape = attn_norm_layers[i].init(
                attn_norm_key, inputs_shape)
            params.update(add_scope_to_params('attn_{}'.format(i),
                                              attn_params))
            params.update(
                add_scope_to_params('attn_norm_{}'.format(i),
                                    attn_norm_params))

            mlp_params, _, inputs_shape = mlp_layers[i].init(
                mlp_key, inputs_shape)
            params.update(add_scope_to_params('mlp_{}'.format(i), mlp_params))
            mlp_norm_params, _, inputs_shape = mlp_norm_layers[i].init(
                mlp_norm_key, inputs_shape)
            params.update(
                add_scope_to_params('mlp_norm_{}'.format(i), mlp_norm_params))
        return params, state, inputs_shape

    def apply_fn(params, state, key, inputs, **kwargs):
        outputs = inputs
        for i in range(depth):
            attn_params = get_params_by_scope('attn_{}'.format(i), params)
            attn_norm_params = get_params_by_scope('attn_norm_{}'.format(i),
                                                   params)
            mlp_params = get_params_by_scope('mlp_{}'.format(i), params)
            mlp_norm_params = get_params_by_scope('mlp_norm_{}'.format(i),
                                                  params)
            outputs = attn_norm_layers[i].apply(attn_norm_params, None, None,
                                                outputs)[0]
            outputs = attn_layers[i].apply(attn_params, None, None,
                                           outputs)[0] + outputs
            outputs = mlp_norm_layers[i].apply(mlp_norm_params, None, None,
                                               outputs)[0]
            outputs = mlp_layers[i].apply(mlp_params, None, None,
                                          outputs)[0] + outputs
        return outputs, None

    return ModuleFn(apply_fn, init=init_fn)


def ortho_visual_transformer_v0(
    n_features: int,
    patch_size: int = 7,
    depth: int = 8,
    layout: Union[str, List[List[Tuple[int, int]]]] = 'butterfly',
) -> ModuleFn:
    """ Create a visual transformer layer.
    
    Args:
        n_features: The number of features.
        patch_size: The size of the patch.
        depth: The number of layers.
        layout: The layout of the RBS gates.
    """
    embed = linear(n_features, True)
    transform = ortho_transformer_v0(n_features,
                                     n_heads=1,
                                     depth=depth,
                                     layout=layout)
    to_patch = extract_patches(patch_size)

    def init_fn(key, inputs_shape):
        n_patches = (inputs_shape[1] * inputs_shape[2]) // (patch_size**2)
        params, state = {}, None
        embed_key, transformer_key, pos_key, cls_key = jax.random.split(key, 4)
        _, _, patches_shape = to_patch.init(None, inputs_shape)
        embed_params, _, embed_shape = embed.init(embed_key, patches_shape)
        transform_params, _, shape = transform.init(transformer_key,
                                                    embed_shape)
        params.update(add_scope_to_params('embed', embed_params))
        params.update(add_scope_to_params('transformer', transform_params))
        params['pos_embed'] = normal()(pos_key, (1, n_patches + 1, n_features))
        params['cls_token'] = normal()(cls_key, (1, 1, n_features))
        return params, state, shape

    def apply_fn(params, state, key, inputs, **kwargs):
        patches, _ = to_patch.apply(None, None, None, inputs)
        embed_params = get_params_by_scope('embed', params)
        patches, _ = embed.apply(embed_params, None, None, patches)
        tokens = jnp.tile(params['cls_token'], [inputs.shape[0], 1, 1])
        patches = jnp.concatenate([tokens, patches], axis=1)
        patches += params['pos_embed']
        transform_params = get_params_by_scope('transformer', params)
        patches = transform.apply(transform_params, None, None, patches)[0]
        return patches[:, 0], None

    return ModuleFn(apply_fn, init=init_fn)
import jax
import optax
from jax import numpy as jnp
import jax
from functools import partial
from tqdm import tqdm, trange

from models import simple_network, recurrent_network, lstm_network, attention_network
from qnn import linear, ortho_linear, ortho_linear_noisy
from train import build_train_fn
from loss_metrics import entropy_loss
from data import gen_paths
from utils import HyperParams
import utils

seed = 42
key = jax.random.PRNGKey(seed)
hps = HyperParams()

# Data
S = gen_paths(hps, seed=0)
[S_train, S_test] = utils.train_test_split([S], test_size=hps.test_size)
_, train_batches = utils.get_batches(
    jnp.array(S_train[0]), batch_size=hps.batch_size)

if hps.layer_type in ['linear', 'linear_svb']:
    layer_func = linear
elif hps.layer_type == 'ortho':
    layer_func = ortho_linear
elif hps.layer_type == 'noisy_ortho':
    layer_func = partial(ortho_linear_noisy, noise_scale=0.01)

if hps.model_type == 'simple':
    net = simple_network(hps=hps, layer_func=layer_func)
elif hps.model_type == 'recurrent':
    net = recurrent_network(hps=hps, layer_func=layer_func)
elif hps.model_type == 'lstm':
    net = lstm_network(hps=hps, layer_func=layer_func)
elif hps.model_type == 'attention':
    net = attention_network(hps=hps, layer_func=layer_func)

opt = utils.make_optimizer(optimizer=hps.optimizer,
                           learning_rate=hps.learning_rate)
key, init_key = jax.random.split(key)
params, state, _ = net.init(init_key, (1, hps.n_steps+1, 1))
opt_state = opt.init(params)
loss_metric = entropy_loss

# Training

train_fn, loss_fn = build_train_fn(
    hps=hps, net=net, opt=opt, loss_metric=loss_metric)
loss = 0.0
with trange(1, hps.num_epochs+1) as t:
    for epoch in t:
        for i, inputs in enumerate(train_batches):
            inputs = inputs[..., None]
            key, train_key = jax.random.split(key)
            params, state, opt_state, loss, (wealths, deltas, outputs) = train_fn(
                params, state, opt_state, train_key, inputs)
        t.set_postfix(loss=loss, model=hps.model_type,
                      layer=hps.layer_type, eps=hps.epsilon)

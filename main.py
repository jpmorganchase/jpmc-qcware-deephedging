import jax
from jax import numpy as jnp
from tqdm import tqdm, trange

import utils
from data import gen_paths
from loss_metrics import entropy_loss
from train import build_train_fn
from utils import HyperParams

seed = 42
key = jax.random.PRNGKey(seed)
hps = HyperParams(S0=100,
                  n_steps=30,
                  n_paths=120000,
                  discrete_path=False,
                  strike_price=100,
                  epsilon=0,
                  sigma=0.2,
                  risk_free=0,
                  dividend=0,
                  model_type='simple',
                  layer_type='linear',
                  n_features=8,
                  n_layers=1,
                  loss_param=1.0,
                  batch_size=256,
                  test_size=0.2,
                  optimizer='adam',
                  learning_rate=1E-3,
                  num_epochs=100
                  )

# Data
S = gen_paths(hps)
[S_train, S_test] = utils.train_test_split([S], test_size=hps.test_size)
_, train_batches = utils.get_batches(
    jnp.array(S_train[0]), batch_size=hps.batch_size)

# Model
layer_func = utils.make_layer(layer_type=hps.layer_type)
net = utils.make_model(hps.model_type)(hps=hps, layer_func=layer_func)
opt = utils.make_optimizer(optimizer=hps.optimizer,
                           learning_rate=hps.learning_rate)
key, init_key = jax.random.split(key)
params, state, _ = net.init(init_key, (1, hps.n_steps, 1))
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

import jax
import optax
from jax import numpy as jnp
import jax

from tqdm import tqdm, trange

from classical_models import shared_simple_network ,simple_network, lstm_network, attention_network
from quantum_models import ortho_simple_network, ortho_lstm_network, ortho_attention_network
from train import build_train_fn
from loss_metrics import entropy_loss
from data import gen_paths
from utils import train_test_split, get_batches, HyperParams

seed = 42
key = jax.random.PRNGKey(seed)
hps = HyperParams()

# Data
S = gen_paths(n_paths=120000, n_steps=hps.n_steps, seed=0)
[S_train, S_test] = train_test_split([S], test_size=0.2)
_, train_batches = get_batches(jnp.array(S_train[0]), batch_size=hps.batch_size)



# Model
# Change model type from the available options in classical_models.py, quantum_models.py
net = shared_simple_network()  #  ['simple_network', 'lstm_network', 'attention_network', 'ortho_simple_network', 'ortho_lstm_network', 'ortho_attention_network']
opt = optax.adam(1E-3)
key, init_key = jax.random.split(key)
params, state, _ = net.init(init_key, (1, 31, 1))
opt_state = opt.init(params)


# Loss
# Change loss function from the available options in loss_metrics.py
loss_metric = entropy_loss

# Training

train_fn, loss_fn = build_train_fn(net, opt, loss_metric)
num_epochs = 150
loss = 0.0    
with trange(1, num_epochs+1) as t:
  for epoch in t:
    for i, inputs in enumerate(train_batches):
      inputs = inputs[...,None]
      key, train_key = jax.random.split(key)
      params, state, opt_state, loss, (wealths, deltas, outputs) = train_fn(
          params, state, opt_state, train_key, inputs)
    t.set_postfix(loss=loss)
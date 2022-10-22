import itertools
import sys
import warnings
from math import factorial
from re import S
from typing import Callable, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

warnings.filterwarnings('ignore')


class Agent(NamedTuple):
    init: Callable
    train_step: Callable
    eval_step: Callable


def binomial(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)


def sample_continuous_jumps(
    key,
    *,
    num_paths=1,
    num_days=8,
    num_jumps=1,
):
    shape = (num_paths, num_days, num_jumps)
    seq_jumps = jax.random.normal(key, shape=shape)
    return seq_jumps


def sample_discrete_jumps(key,
                          *,
                          num_paths=1,
                          num_days=8,
                          num_jumps=1,
                          bernoulli_prob=0.5):
    shape = (num_paths, num_days, num_jumps)
    seq_jumps = jax.random.bernoulli(key, shape=shape, p=bernoulli_prob)
    seq_jumps = (seq_jumps - bernoulli_prob)  # mean 0
    seq_jumps /= np.sqrt(bernoulli_prob * (1 - bernoulli_prob))  # variance 1
    return seq_jumps


def all_discrete_jumps(*, num_days=8, num_jumps=1, bernoulli_prob=0.5):
    all_jumps = list(itertools.product(*[[0, 1]] * (num_days * num_jumps)))
    seq_jumps = jnp.array(all_jumps).reshape(-1, num_days, num_jumps)
    seq_jumps = (seq_jumps - bernoulli_prob)  # mean 0
    seq_jumps /= np.sqrt(bernoulli_prob * (1 - bernoulli_prob))  # variance 1
    return seq_jumps


def compute_black_scholes_deltas(seq_prices,
                                 *,
                                 num_days=8,
                                 num_trading_days=252,
                                 mu=0.,
                                 sigma=0.5,
                                 strike=1.):
    seq_prices = seq_prices[..., None]
    seq_prices = seq_prices[:, :-1]
    strike_price = seq_prices[0, 0] * strike
    T = jnp.arange(1, num_days + 1) / num_trading_days
    T = jnp.repeat(jnp.flip(T[None, :]), seq_prices.shape[0], 0)
    d1 = jnp.divide(
        jnp.log(seq_prices[..., 0] / strike_price) + (mu + 0.5 * sigma**2) * T,
        sigma * jnp.sqrt(T))
    seq_deltas = jax.scipy.stats.norm.cdf(d1, 0.0, 1.0)
    return seq_deltas


def compute_prices(
    seq_jumps,
    *,
    num_trading_days=252,
    mu=0.,
    sigma=0.5,
    initial_price=100.,
):
    num_paths, num_days, num_jumps = seq_jumps.shape
    seq_jumps = seq_jumps.reshape(num_paths, num_days * num_jumps)
    brownian = jnp.cumsum(seq_jumps, axis=1)
    brownian /= np.sqrt(num_jumps * num_trading_days)
    t = jnp.arange(1, 1 + num_days) / num_trading_days
    log_prices = (mu - sigma**2 / 2) * t + sigma * brownian
    seq_prices = jnp.exp(log_prices)
    seq_prices = jnp.concatenate([jnp.ones((num_paths, 1)), seq_prices],
                                 axis=1)
    seq_prices *= initial_price
    return seq_prices


def compute_rewards(
    seq_prices,
    seq_deltas,
    *,
    strike=0.9,
    cost_eps=0.,
):
    seq_actions = [
        seq_deltas[:, [0]],
        seq_deltas[:, 1:] - seq_deltas[:, :-1],
        -seq_deltas[:, [-1]],
    ]
    seq_actions = jnp.concatenate(seq_actions, axis=1)
    payoff = -jnp.maximum(seq_prices[:, -1] - strike * seq_prices[:, 0], 0.)
    costs = -(jnp.abs(seq_actions) * cost_eps + seq_actions) * seq_prices
    seq_rewards = costs.at[:, -1].add(payoff)
    return seq_rewards


def compute_bounds(num_days=8,
                   num_jumps=1,
                   bernoulli_prob=0.5,
                   num_trading_days=252,
                   mu=0.,
                   sigma=0.5,
                   initial_price=100.,
                   strike=0.9,
                   cost_eps=0.):
    jumps_max = jnp.ones((num_days, num_jumps))
    jumps_min = -jnp.ones((num_days, num_jumps))
    seq_jumps = jnp.stack([jumps_min, jumps_max], axis=0)
    prices_min, prices_max = compute_prices(seq_jumps,
                                            num_trading_days=num_trading_days,
                                            mu=mu,
                                            sigma=sigma,
                                            initial_price=initial_price)
    payoffs_min = -jnp.maximum(prices_max - strike * initial_price, 0)
    values_max = (2 * (prices_max - strike * initial_price))[::-1][:-1]
    values_min = (2 * (prices_min - strike * initial_price) +
                  payoffs_min)[::-1][:-1]
    Gt_range = jnp.stack((values_min, values_max), axis=0)
    return Gt_range


def compute_returns(seq_rewards):
    seq_returns = jnp.cumsum(seq_rewards[:, ::-1], axis=1)[:, ::-1]
    return seq_returns


def compute_utility(seq_rewards, *, utility_lambda=1.):
    returns = seq_rewards.sum(axis=1)
    utility = -1 / utility_lambda * jnp.log(
        jnp.mean(jnp.exp(-utility_lambda * returns)))
    return utility


@jax.vmap
def encode(seq_jumps, *, bernoulli_prob=0.5):
    num_days, num_jumps = seq_jumps.shape
    seq_jumps *= np.sqrt(bernoulli_prob * (1 - bernoulli_prob))
    seq_jumps += bernoulli_prob
    encodings = []
    for time_step in range(num_days):
        future_qubits = (num_days - time_step) * num_jumps
        future = 1 / np.sqrt(2**future_qubits) * jnp.ones((2**future_qubits, ))
        if time_step == 0:
            encoding = future
        else:
            aux = seq_jumps[time_step - 1].reshape(-1)
            aux = jnp.concatenate((1 - aux, aux), 0)
            if time_step == 1:
                past = aux
            else:
                past = jnp.kron(past, aux)
            encoding = jnp.kron(past, future)
        encoding = jnp.kron(encoding, jnp.array([0., 1., 0., 0.]))
        encodings.append(encoding)
    encodings = jnp.stack(encodings, 0)
    return encodings


def get_pyramid_idxs(num_qubits):
    num_max = num_qubits
    num_min = num_qubits - 1
    if num_max == num_min:
        num_min -= 1
    end_idxs = np.concatenate(
        [np.arange(1, num_max - 1), num_max - np.arange(1, num_min + 1)])
    start_idxs = np.concatenate([
        np.arange(end_idxs.shape[0] + num_min - num_max) % 2,
        np.arange(num_max - num_min)
    ])
    rbs_idxs = [
        np.arange(start_idxs[i], end_idxs[i] + 1).reshape(-1, 2)
        for i in range(len(start_idxs))
    ]
    return rbs_idxs


def get_butterfly_idxs(num_qubits):
    def _get_butterfly_idxs(n):
        if n == 2:
            return np.array([[[0, 1]]])
        else:
            rbs_idxs = _get_butterfly_idxs(n // 2)
            first = np.concatenate([rbs_idxs, rbs_idxs + n // 2], 1)
            last = np.arange(n).reshape(1, 2, n // 2).transpose(0, 2, 1)
            rbs_idxs = np.concatenate([first, last], 0)
            return rbs_idxs

    rbs_idxs = _get_butterfly_idxs(int(2**np.ceil(np.log2(num_qubits))))
    rbs_idxs = [list(map(list, rbs_idx)) for rbs_idx in rbs_idxs]
    rbs_idxs = [[[i, j] for i, j in rbs_idx
                 if (i in range(num_qubits)) and (j in range(num_qubits))]
                for rbs_idx in rbs_idxs]
    return rbs_idxs


def get_triangle_idxs(num_qubits):
    rbs_idxs = [[(i, i + 1)] for i in range(num_qubits - 1)]
    rbs_idxs += rbs_idxs[::-1]
    return rbs_idxs


def get_iks_idxs(num_qubits):
    rbs_idxs_down = [[(i, i + 1)] for i in range(num_qubits - 1)]
    rbs_idxs_up = [[(i, i + 1)] for i in range(num_qubits - 1)][::-1]
    rbs_idxs = [
        (m + n if m != n else m) for m, n in zip(rbs_idxs_down, rbs_idxs_up)
    ] + rbs_idxs_down[num_qubits - 1:]
    return rbs_idxs


def make_ortho_fn(rbs_idxs, num_qubits):
    rbs_idxs = [list(map(list, rbs_idx)) for rbs_idx in rbs_idxs]
    len_idxs = np.cumsum([0] + list(map(len, rbs_idxs)))

    def get_rbs_unary(theta):
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        unary = jnp.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ])
        unary = unary.transpose(*[*range(2, unary.ndim), 0, 1])
        return unary

    def get_rbs_unary_grad(theta):
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        unary = jnp.array([
            [-sin_theta, -cos_theta],
            [cos_theta, -sin_theta],
        ])
        unary = unary.transpose(*[*range(2, unary.ndim), 0, 1])
        return unary

    @jax.custom_jvp
    def get_parallel_rbs_unary(thetas):
        unitaries = []
        for i, idxs in enumerate(rbs_idxs):
            idxs = sum(idxs, [])
            sub_thetas = thetas[len_idxs[i]:len_idxs[i + 1]]
            rbs_blocks = get_rbs_unary(sub_thetas)
            eye_block = jnp.eye(num_qubits - len(idxs), dtype=thetas.dtype)
            permutation = idxs + [
                i for i in range(num_qubits) if i not in idxs
            ]
            permutation = np.argsort(permutation)
            unary = jax.scipy.linalg.block_diag(*rbs_blocks, eye_block)
            unary = unary[permutation][:, permutation]
            unitaries.append(unary)
        unitaries = jnp.stack(unitaries)
        return unitaries

    @get_parallel_rbs_unary.defjvp
    def get_parallel_rbs_unary_jvp(primals, tangents):
        thetas, = primals
        thetas_dot, = tangents
        unitaries = []
        unitaries_dot = []
        for i, idxs in enumerate(rbs_idxs):
            idxs = sum(idxs, [])
            sub_thetas = thetas[len_idxs[i]:len_idxs[i + 1]]
            sub_thetas_dot = thetas_dot[len_idxs[i]:len_idxs[i + 1]]
            rbs_blocks = get_rbs_unary(sub_thetas)
            rbs_blocks_grad = get_rbs_unary_grad(sub_thetas)
            rbs_blocks_dot = sub_thetas_dot[..., None, None] * rbs_blocks_grad
            eye_block = jnp.eye(num_qubits - len(idxs), dtype=thetas.dtype)
            zero_block = jnp.zeros_like(eye_block)
            permutation = idxs + [
                i for i in range(num_qubits) if i not in idxs
            ]
            permutation = np.argsort(permutation)
            unary = jax.scipy.linalg.block_diag(*rbs_blocks, eye_block)
            unary_dot = jax.scipy.linalg.block_diag(*rbs_blocks_dot,
                                                    zero_block)
            unary = unary[permutation][:, permutation]
            unary_dot = unary_dot[permutation][:, permutation]
            unitaries.append(unary)
            unitaries_dot.append(unary_dot)
        primal_out = jnp.stack(unitaries)
        tangent_out = jnp.stack(unitaries_dot)
        return primal_out, tangent_out

    def orthogonal_fn(thetas):
        unitaries = get_parallel_rbs_unary(thetas)
        unary = jnp.linalg.multi_dot(unitaries[::-1])
        return unary

    return orthogonal_fn


def compute_compound(unary, order=1):
    num_qubits = unary.shape[-1]
    if (order == 0) or (order == num_qubits):
        return jnp.ones((1, 1))
    elif order == 1:
        return unary
    else:
        subsets = list(itertools.combinations(range(num_qubits), order))
        compounds = unary[subsets, ...][..., subsets].transpose(0, 2, 1, 3)
        compound = jnp.linalg.det(compounds)
    return compound


def decompose_state(state):
    num_qubits = int(np.log2(state.shape[-1]))
    batch_dims = state.shape[:-1]
    state = state.reshape(-1, 2**num_qubits)
    idxs = list(itertools.product(*[[0, 1]] * num_qubits))
    subspace_idxs = [[(np.array(idx) * 2**np.arange(num_qubits)[::-1]).sum()
                      for idx in idxs if sum(idx) == weight]
                     for weight in range(num_qubits + 1)]
    subspace_states = [
        state[..., subspace_idxs[weight]] for weight in range(num_qubits + 1)
    ]
    alphas = [
        jnp.linalg.norm(subspace_state, axis=-1)
        for subspace_state in subspace_states
    ]
    betas = [
        subspace_state / (alpha[..., None] + 1E-6)
        for alpha, subspace_state in zip(alphas, subspace_states)
    ]
    alphas = [alpha.reshape(*batch_dims, -1) for alpha in alphas]
    betas = [beta.reshape(*batch_dims, -1) for beta in betas]
    alphas = jnp.stack(alphas, -1)[..., 0, :]
    return alphas, betas


def make_classical_net(
    num_hidden_layers=1,
    num_hidden_features=32,
    distributional=False,
):
    def forward_fn(seq_inputs):
        seq_outputs = []
        for time_step, inputs in enumerate(seq_inputs):
            if distributional:
                num_outputs = len(seq_inputs) - time_step + 1
            else:
                num_outputs = 1
            net = hk.nets.MLP((num_hidden_layers + 1) * [num_hidden_features] +
                              [num_outputs],
                              activation=jax.nn.leaky_relu)
            outputs = net(inputs)
            seq_outputs.append(outputs)
        return seq_outputs

    return forward_fn


def make_classical_actor(
    num_hidden_layers=1,
    num_hidden_features=32,
):
    def actor_fn(seq_jumps, seq_prices):
        num_days = seq_prices.shape[1] - 1
        seq_inputs = [
            seq_prices[:, :time_step] for time_step in range(1, num_days + 1)
        ]
        seq_outputs = make_classical_net(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)(seq_inputs)
        seq_deltas = jax.nn.sigmoid(jnp.stack(seq_outputs, axis=1)[..., 0])
        return seq_deltas

    return actor_fn


def make_quantum_net(layout="butterfly"):
    def forward_fn(alphas, betas):
        num_days = alphas.shape[1]
        num_qubits = num_days + 2
        if layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(num_qubits)
        elif layout == "iks":
            rbs_idxs = get_iks_idxs(num_qubits)
        elif layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(num_qubits)
        elif layout == "triangle":
            rbs_idxs = get_triangle_idxs(num_qubits)
        else:
            raise ValueError("Invalid layout.")
        orthogonal_fn = make_ortho_fn(rbs_idxs, num_qubits)
        num_params = sum(map(len, rbs_idxs))
        thetas = hk.get_parameter(
            "thetas", (num_days, num_params),
            jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=1 / len(rbs_idxs)))
        unary = jax.vmap(orthogonal_fn)(thetas)
        compounds = [
            jax.vmap(compute_compound, in_axes=(0, None))(unary, order)
            for order in range(num_qubits + 1)
        ]
        betas = [
            jnp.einsum("tij,btj->bti", compound, beta)
            for compound, beta in zip(compounds, betas)
        ]
        bounds = [
            jnp.arange(beta.shape[-1]) / (beta.shape[-1] - 1 + 1E-8)
            for beta in betas
        ]
        seq_dist_outputs = jnp.array([
            jnp.einsum("i,bti->bt", bound, beta**2)
            for bound, beta in zip(bounds, betas)
        ]).transpose(1, 2, 0)
        seq_outputs = jnp.einsum("bti,bti->bt", alphas**2, seq_dist_outputs)
        return seq_outputs, seq_dist_outputs

    return forward_fn


def make_quantum_actor(layout="butterfly"):
    def actor_fn(seq_jumps, seq_prices):
        encodings = encode(seq_jumps)
        alphas, betas = decompose_state(encodings)
        seq_outputs, _ = make_quantum_net(layout)(alphas, betas)
        seq_deltas = seq_outputs
        return seq_deltas

    return actor_fn


def make_opt(
    opt_name="adam",
    learning_rate=1e-3,
):
    return getattr(optax, opt_name)(learning_rate=learning_rate)


def vanilla(num_days=14,
            num_jumps=1,
            num_trading_days=252,
            mu=0.,
            sigma=0.2,
            initial_price=100.,
            strike=1.,
            cost_eps=0.,
            train_num_paths=32,
            eval_num_paths=32,
            utility_lambda=0.1,
            actor_net="classical",
            actor_net_kwargs={},
            actor_opt="adam",
            actor_opt_kwargs={"learning_rate": 1e-2}):
    if actor_net == "classical":
        num_hidden_features = actor_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = actor_net_kwargs.get("num_hidden_layers", 2)
        actor_fn = make_classical_actor(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)

        def actor_fn(seq_jumps, seq_prices):
            seq_prices = seq_prices[:, :-1, None].transpose(1, 0, 2)
            core = hk.LSTM(16)
            batch_size = seq_prices.shape[1]
            outs, state = hk.dynamic_unroll(core, seq_prices,
                                            core.initial_state(batch_size))
            outs = hk.BatchApply(hk.Linear(1))(outs)[..., 0].transpose(1, 0)
            outs = jax.nn.sigmoid(outs)
            return outs
    elif actor_net == "quantum":
        layout = actor_net_kwargs.get("layout", "butterfly")
        actor_fn = make_quantum_actor(layout=layout)

    actor = hk.transform(actor_fn)
    actor_opt = make_opt(opt_name=actor_opt, **actor_opt_kwargs)

    def init(key):
        actor_params = actor.init(key,
                                  seq_jumps=jnp.zeros((1, num_days, 1)),
                                  seq_prices=jnp.zeros((1, num_days + 1)))
        actor_opt_state = actor_opt.init(actor_params)
        params = (actor_params, )
        opt_state = (actor_opt_state, )
        return params, opt_state

    CONTINUOUS = False

    @jax.jit
    def train_step(key, params, opt_state):
        def actor_loss_fn(key, actor_params):
            keys = jax.random.split(key, 4)
            if CONTINUOUS:
                seq_jumps = sample_continuous_jumps(keys[0],
                                                    num_paths=train_num_paths,
                                                    num_days=num_days,
                                                    num_jumps=num_jumps)
            else:
                seq_jumps = sample_discrete_jumps(keys[0],
                                                  num_paths=train_num_paths,
                                                  num_days=num_days,
                                                  num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)
            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            utility = compute_utility(seq_rewards,
                                      utility_lambda=utility_lambda)
            loss = -utility
            return loss.mean()

        actor_params, = params
        actor_opt_state, = opt_state
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn, argnums=1)(
            key,
            actor_params,
        )
        actor_updates, actor_opt_state = actor_opt.update(
            actor_grads, actor_opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, actor_updates)
        params = (actor_params, )
        opt_state = (actor_opt_state, )
        metrics = {'actor_loss': actor_loss}

        return params, opt_state, metrics

    @jax.jit
    def eval_step(key, params):
        actor_params, = params
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 2)
        if CONTINUOUS:
            seq_jumps = sample_continuous_jumps(keys[0],
                                                num_paths=eval_num_paths,
                                                num_days=num_days,
                                                num_jumps=num_jumps)
        else:
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=eval_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
        seq_prices = compute_prices(seq_jumps,
                                    num_trading_days=num_trading_days,
                                    mu=mu,
                                    sigma=sigma,
                                    initial_price=initial_price)

        seq_deltas = actor.apply(actor_params, keys[1], seq_jumps, seq_prices)
        seq_bs_deltas = compute_black_scholes_deltas(
            seq_prices,
            num_days=num_days,
            num_trading_days=num_trading_days,
            mu=mu,
            sigma=sigma,
            strike=strike)
        seq_rewards = compute_rewards(seq_prices,
                                      seq_deltas,
                                      strike=strike,
                                      cost_eps=cost_eps)
        seq_bs_rewards = compute_rewards(seq_prices,
                                         seq_bs_deltas,
                                         strike=strike,
                                         cost_eps=cost_eps)
        returns = seq_rewards.sum(axis=1).mean()
        bs_returns = seq_bs_rewards.sum(axis=1).mean()
        utility = compute_utility(seq_rewards, utility_lambda=utility_lambda)
        bs_utility = compute_utility(seq_bs_rewards,
                                     utility_lambda=utility_lambda)
        metrics = {
            'utility': utility,
            'returns': returns,
            'bs_utility': bs_utility,
            'bs_returns': bs_returns
        }
        for k in range(num_days):
            metrics['d_mean_{}'.format(k)] = seq_deltas[:, k].mean()
            metrics['d_min_{}'.format(k)] = seq_deltas[:, k].min()
            metrics['d_max_{}'.format(k)] = seq_deltas[:, k].max()
            metrics['d_std_{}'.format(k)] = seq_deltas[:, k].std()
        return metrics

    return Agent(init, train_step, eval_step)


def actor_critic_original(
    num_days=10,
    num_jumps=1,
    num_trading_days=252,
    mu=0.1,
    sigma=0.3,
    initial_price=100.,
    strike=1.,
    cost_eps=0.,
    train_num_paths=32,
    eval_num_paths=32,
    utility_lambda=0.1,
    actor_net="classical",
    actor_net_kwargs={},
    actor_opt="radam",
    actor_opt_kwargs={"learning_rate": 1e-3},
    critic_net="classical",
    critic_net_kwargs={},
    critic_opt="adam",
    critic_opt_kwargs={"learning_rate": 1e-3},
):
    if actor_net == "classical":
        num_hidden_features = actor_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = actor_net_kwargs.get("num_hidden_layers", 2)
        actor_fn = make_classical_actor(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)

        def actor_fn(seq_jumps, seq_prices):
            seq_prices = seq_prices[:, :-1, None].transpose(1, 0, 2)
            core = hk.LSTM(16)
            batch_size = seq_prices.shape[1]
            outs, state = hk.dynamic_unroll(core, seq_prices,
                                            core.initial_state(batch_size))
            outs = hk.BatchApply(hk.Linear(1))(outs)[..., 0].transpose(1, 0)
            outs = jax.nn.sigmoid(outs)
            return outs
    elif actor_net == "quantum":
        layout = actor_net_kwargs.get("layout", "butterfly")
        actor_fn = make_quantum_actor(layout=layout)

    if critic_net == "classical":
        num_hidden_features = critic_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = critic_net_kwargs.get("num_hidden_layers", 2)

        def make_classical_original_critic(
            num_hidden_layers=1,
            num_hidden_features=32,
        ):
            def critic_fn(seq_jumps, seq_prices):
                num_days = seq_prices.shape[1] - 1
                seq_inputs = [
                    seq_prices[:, :time_step]
                    for time_step in range(1, num_days + 1)
                ]
                seq_outputs = make_classical_net(
                    num_hidden_features=num_hidden_features,
                    num_hidden_layers=num_hidden_layers)(seq_inputs)
                seq_values = jnp.stack(seq_outputs, axis=1)[..., 0]
                return seq_values

            return critic_fn

        critic_fn = make_classical_original_critic(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)

        def critic_fn(seq_jumps, seq_prices):
            seq_prices = seq_prices[:, :-1, None].transpose(1, 0, 2)
            core = hk.LSTM(16)
            batch_size = seq_prices.shape[1]
            outs, state = hk.dynamic_unroll(core, seq_prices,
                                            core.initial_state(batch_size))
            outs = hk.BatchApply(hk.Linear(1))(outs)[..., 0].transpose(1, 0)
            outs = -1 / utility_lambda * jnp.log(jax.nn.softplus(outs))
            return outs
    elif critic_net == "quantum":
        layout = critic_net_kwargs.get("layout", "butterfly")
        train_bounds = critic_net_kwargs.get("train_bounds", False)

        def make_quantum_original_critic(layout="butterfly",
                                         train_bounds=False):
            def critic_fn(seq_jumps, seq_prices):
                encodings = encode(seq_jumps)
                alphas, betas = decompose_state(encodings)
                seq_outputs, _ = make_quantum_net(layout)(alphas, betas)
                if train_bounds:
                    utility_min = hk.get_parameter("utility_min", (num_days, ),
                                                   init=jnp.zeros)
                    utility_max = hk.get_parameter("utility_max", (num_days, ),
                                                   init=jnp.ones)
                else:
                    bounds = compute_bounds(num_days=num_days,
                                            mu=mu,
                                            sigma=sigma,
                                            strike=strike,
                                            cost_eps=cost_eps,
                                            initial_price=initial_price,
                                            num_trading_days=num_trading_days)
                    utility_min, utility_max = -1 / utility_lambda * jnp.log(
                        jnp.exp(-utility_lambda * bounds))
                seq_values = utility_min + (utility_max -
                                            utility_min) * seq_outputs
                return seq_values

            return critic_fn

        critic_fn = make_quantum_original_critic(layout=layout,
                                                 train_bounds=train_bounds)

    actor = hk.transform(actor_fn)
    actor_opt = make_opt(opt_name=actor_opt, **actor_opt_kwargs)

    critic = hk.transform(critic_fn)
    critic_opt = make_opt(opt_name=critic_opt, **critic_opt_kwargs)

    def init(key):
        actor_params = actor.init(key,
                                  seq_jumps=jnp.zeros((1, num_days, 1)),
                                  seq_prices=jnp.zeros((1, num_days + 1)))
        critic_params = critic.init(key,
                                    seq_jumps=jnp.zeros((1, num_days, 1)),
                                    seq_prices=jnp.zeros((1, num_days + 1)))
        target_params = critic_params
        actor_opt_state = actor_opt.init(actor_params)
        critic_opt_state = critic_opt.init(critic_params)
        params = (actor_params, critic_params, target_params)
        opt_state = (actor_opt_state, critic_opt_state)
        return params, opt_state

    @jax.jit
    def train_step(key, params, opt_state):
        def critic_loss_fn(key, actor_params, critic_params, target_params):
            keys = jax.random.split(key, 4)
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=train_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)

            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            seq_values = critic.apply(critic_params, keys[2], seq_jumps,
                                      seq_prices)
            seq_target_values = critic.apply(target_params, keys[3], seq_jumps,
                                             seq_prices)
            seq_next_values = jnp.concatenate(
                [seq_target_values[:, 1:], seq_rewards[:, [-1]]], axis=1)
            seq_targets = seq_rewards[:, :-1] + seq_next_values
            loss = 1 / utility_lambda * jnp.exp(
                -utility_lambda * (seq_targets - seq_values)) - seq_values
            return loss.mean()

        def actor_loss_fn(key, actor_params, critic_params):
            keys = jax.random.split(key, 4)
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=train_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)
            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            seq_values = critic.apply(critic_params, keys[2], seq_jumps,
                                      seq_prices)
            seq_next_values = jnp.concatenate(
                [seq_values[:, 1:], seq_rewards[:, [-1]]], axis=1)
            loss = 1 / utility_lambda * jnp.exp(
                -utility_lambda * (seq_rewards[:, :-1] + seq_next_values))
            return loss.mean()

        keys = jax.random.split(key, 2)

        actor_params, critic_params, target_params = params
        actor_opt_state, critic_opt_state = opt_state

        critic_loss, critic_grads = jax.value_and_grad(
            critic_loss_fn, argnums=2)(keys[0], actor_params, critic_params,
                                       target_params)
        critic_updates, critic_opt_state = critic_opt.update(
            critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)
        target_params = optax.incremental_update(critic_params,
                                                 target_params,
                                                 step_size=0.001)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn, argnums=1)(
            keys[1],
            actor_params,
            critic_params,
        )
        actor_updates, actor_opt_state = actor_opt.update(
            actor_grads, actor_opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        params = (actor_params, critic_params, target_params)
        opt_state = (actor_opt_state, critic_opt_state)

        metrics = {'critic_loss': critic_loss, 'actor_loss': actor_loss}
        return params, opt_state, metrics

    @jax.jit
    def eval_step(key, params):
        actor_params, _, _ = params
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 2)
        seq_jumps = sample_discrete_jumps(keys[0],
                                          num_paths=eval_num_paths,
                                          num_days=num_days,
                                          num_jumps=num_jumps)
        seq_prices = compute_prices(seq_jumps,
                                    num_trading_days=num_trading_days,
                                    mu=mu,
                                    sigma=sigma,
                                    initial_price=initial_price)

        seq_deltas = actor.apply(actor_params, keys[1], seq_jumps, seq_prices)
        seq_bs_deltas = compute_black_scholes_deltas(
            seq_prices,
            num_days=num_days,
            num_trading_days=num_trading_days,
            mu=mu,
            sigma=sigma,
            strike=strike)
        seq_rewards = compute_rewards(seq_prices,
                                      seq_deltas,
                                      strike=strike,
                                      cost_eps=cost_eps)
        seq_bs_rewards = compute_rewards(seq_prices,
                                         seq_bs_deltas,
                                         strike=strike,
                                         cost_eps=cost_eps)

        returns = seq_rewards.sum(axis=1).mean()
        bs_returns = seq_bs_rewards.sum(axis=1).mean()
        utility = compute_utility(seq_rewards, utility_lambda=utility_lambda)
        bs_utility = compute_utility(seq_bs_rewards,
                                     utility_lambda=utility_lambda)
        metrics = {
            'utility': utility,
            'returns': returns,
            'bs_utility': bs_utility,
            'bs_returns': bs_returns,
        }
        return metrics

    return Agent(init, train_step, eval_step)


def actor_critic_expected(
    num_days=10,
    num_jumps=1,
    num_trading_days=252,
    mu=0.1,
    sigma=0.3,
    initial_price=100.,
    strike=1.,
    cost_eps=0.,
    train_num_paths=32,
    eval_num_paths=32,
    utility_lambda=0.1,
    actor_net="classical",
    actor_net_kwargs={},
    actor_opt="radam",
    actor_opt_kwargs={"learning_rate": 1e-3},
    critic_net="classical",
    critic_net_kwargs={},
    critic_opt="adam",
    critic_opt_kwargs={"learning_rate": 1e-3},
):
    if actor_net == "classical":
        num_hidden_features = actor_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = actor_net_kwargs.get("num_hidden_layers", 2)
        actor_fn = make_classical_actor(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)

        def actor_fn(seq_jumps, seq_prices):
            seq_prices = seq_prices[:, :-1, None].transpose(1, 0, 2)
            core = hk.LSTM(16)
            batch_size = seq_prices.shape[1]
            outs, state = hk.dynamic_unroll(core, seq_prices,
                                            core.initial_state(batch_size))
            outs = hk.BatchApply(hk.Linear(1))(outs)[..., 0].transpose(1, 0)
            outs = jax.nn.sigmoid(outs)
            return outs
    elif actor_net == "quantum":
        layout = actor_net_kwargs.get("layout", "butterfly")
        actor_fn = make_quantum_actor(layout=layout)

    if critic_net == "classical":
        num_hidden_features = critic_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = critic_net_kwargs.get("num_hidden_layers", 2)

        def make_classical_expected_critic(
            num_hidden_layers=1,
            num_hidden_features=32,
        ):
            def critic_fn(seq_jumps, seq_prices):
                num_days = seq_prices.shape[1] - 1
                seq_inputs = [
                    seq_prices[:, :time_step]
                    for time_step in range(1, num_days + 1)
                ]
                seq_outputs = make_classical_net(
                    num_hidden_features=num_hidden_features,
                    num_hidden_layers=num_hidden_layers)(seq_inputs)
                seq_outputs = jnp.stack(seq_outputs, axis=1)[..., 0]
                seq_expectations = jax.nn.softplus(seq_outputs)
                return seq_expectations

            return critic_fn

        critic_fn = make_classical_expected_critic(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)
    elif critic_net == "quantum":
        layout = critic_net_kwargs.get("layout", "butterfly")
        train_bounds = critic_net_kwargs.get("train_bounds", False)

        def make_quantum_expected_critic(layout="butterfly",
                                         train_bounds=False):
            def critic_fn(seq_jumps, seq_prices):
                encodings = encode(seq_jumps)
                alphas, betas = decompose_state(encodings)
                seq_outputs, _ = make_quantum_net(layout)(alphas, betas)
                if train_bounds:
                    utility_min = hk.get_parameter("utility_min", (num_days, ),
                                                   init=jnp.zeros)
                    utility_max = hk.get_parameter("utility_max", (num_days, ),
                                                   init=jnp.ones)
                else:
                    bounds = compute_bounds(num_days=num_days,
                                            mu=mu,
                                            sigma=sigma,
                                            strike=strike,
                                            cost_eps=cost_eps,
                                            initial_price=initial_price,
                                            num_trading_days=num_trading_days)
                    utility_min, utility_max = jnp.exp(-utility_lambda *
                                                       bounds[::-1])
                seq_expectations = utility_min + (utility_max -
                                                  utility_min) * seq_outputs
                return seq_expectations

            return critic_fn

        critic_fn = make_quantum_expected_critic(layout=layout,
                                                 train_bounds=train_bounds)

    actor = hk.transform(actor_fn)
    actor_opt = make_opt(opt_name=actor_opt, **actor_opt_kwargs)

    critic = hk.transform(critic_fn)
    critic_opt = make_opt(opt_name=critic_opt, **critic_opt_kwargs)

    def init(key):
        actor_params = actor.init(key,
                                  seq_jumps=jnp.zeros((1, num_days, 1)),
                                  seq_prices=jnp.zeros((1, num_days + 1)))
        critic_params = critic.init(key,
                                    seq_jumps=jnp.zeros((1, num_days, 1)),
                                    seq_prices=jnp.zeros((1, num_days + 1)))
        actor_opt_state = actor_opt.init(actor_params)
        critic_opt_state = critic_opt.init(critic_params)
        params = (actor_params, critic_params)
        opt_state = (actor_opt_state, critic_opt_state)
        return params, opt_state

    @jax.jit
    def train_step(key, params, opt_state):
        def critic_loss_fn(key, actor_params, critic_params):
            keys = jax.random.split(key, 4)
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=train_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)

            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            seq_expectations = critic.apply(critic_params, keys[2], seq_jumps,
                                            seq_prices)
            seq_returns = compute_returns(seq_rewards)[:, :-1]
            seq_returns = jnp.exp(-utility_lambda * seq_returns)
            loss = optax.huber_loss(seq_returns, seq_expectations)
            return loss.mean()

        def actor_loss_fn(key, actor_params, critic_params):
            keys = jax.random.split(key, 4)
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=train_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)
            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            seq_expectations = critic.apply(critic_params, keys[2], seq_jumps,
                                            seq_prices)
            seq_values = -1 / utility_lambda * jnp.log(
                jnp.maximum(seq_expectations, 1E-6))
            seq_next_values = jnp.concatenate(
                [seq_values[:, 1:], seq_rewards[:, [-1]]], axis=1)
            loss = 1 / utility_lambda * jnp.exp(
                -utility_lambda * (seq_rewards[:, :-1] + seq_next_values))
            return loss.mean()

        keys = jax.random.split(key, 2)

        actor_params, critic_params = params
        actor_opt_state, critic_opt_state = opt_state

        critic_loss, critic_grads = jax.value_and_grad(
            critic_loss_fn, argnums=2)(keys[0], actor_params, critic_params)
        critic_updates, critic_opt_state = critic_opt.update(
            critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn, argnums=1)(
            keys[1],
            actor_params,
            critic_params,
        )
        actor_updates, actor_opt_state = actor_opt.update(
            actor_grads, actor_opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        params = (actor_params, critic_params)
        opt_state = (actor_opt_state, critic_opt_state)

        metrics = {'critic_loss': critic_loss, 'actor_loss': actor_loss}
        return params, opt_state, metrics

    @jax.jit
    def eval_step(key, params):
        actor_params, _ = params
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 2)
        seq_jumps = sample_discrete_jumps(keys[0],
                                          num_paths=eval_num_paths,
                                          num_days=num_days,
                                          num_jumps=num_jumps)
        seq_prices = compute_prices(seq_jumps,
                                    num_trading_days=num_trading_days,
                                    mu=mu,
                                    sigma=sigma,
                                    initial_price=initial_price)

        seq_deltas = actor.apply(actor_params, keys[1], seq_jumps, seq_prices)
        seq_bs_deltas = compute_black_scholes_deltas(
            seq_prices,
            num_days=num_days,
            num_trading_days=num_trading_days,
            mu=mu,
            sigma=sigma,
            strike=strike)
        seq_rewards = compute_rewards(seq_prices,
                                      seq_deltas,
                                      strike=strike,
                                      cost_eps=cost_eps)
        seq_bs_rewards = compute_rewards(seq_prices,
                                         seq_bs_deltas,
                                         strike=strike,
                                         cost_eps=cost_eps)
        returns = seq_rewards.sum(axis=1).mean()
        bs_returns = seq_bs_rewards.sum(axis=1).mean()
        utility = compute_utility(seq_rewards, utility_lambda=utility_lambda)
        bs_utility = compute_utility(seq_bs_rewards,
                                     utility_lambda=utility_lambda)
        metrics = {
            'utility': utility,
            'returns': returns,
            'bs_utility': bs_utility,
            'bs_returns': bs_returns,
        }
        return metrics

    return Agent(init, train_step, eval_step)


def actor_critic_distributional(
    num_days=10,
    num_jumps=1,
    num_trading_days=252,
    mu=0.1,
    sigma=0.3,
    initial_price=100.,
    strike=1.,
    cost_eps=0.,
    train_num_paths=32,
    eval_num_paths=32,
    utility_lambda=0.1,
    actor_net="classical",
    actor_net_kwargs={},
    actor_opt="radam",
    actor_opt_kwargs={"learning_rate": 1e-3},
    critic_net="classical",
    critic_net_kwargs={},
    critic_opt="adam",
    critic_opt_kwargs={"learning_rate": 1e-3},
):
    if actor_net == "classical":
        num_hidden_features = actor_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = actor_net_kwargs.get("num_hidden_layers", 2)
        actor_fn = make_classical_actor(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)

        def actor_fn(seq_jumps, seq_prices):
            seq_prices = seq_prices[:, :-1, None].transpose(1, 0, 2)
            core = hk.LSTM(16)
            batch_size = seq_prices.shape[1]
            outs, state = hk.dynamic_unroll(core, seq_prices,
                                            core.initial_state(batch_size))
            outs = hk.BatchApply(hk.Linear(1))(outs)[..., 0].transpose(1, 0)
            outs = jax.nn.sigmoid(outs)
            return outs
    elif actor_net == "quantum":
        layout = actor_net_kwargs.get("layout", "butterfly")
        actor_fn = make_quantum_actor(layout=layout)

    if critic_net == "classical":
        num_hidden_features = critic_net_kwargs.get("num_hidden_features", 16)
        num_hidden_layers = critic_net_kwargs.get("num_hidden_layers", 2)

        def make_classical_distributional_critic(
            num_hidden_layers=1,
            num_hidden_features=32,
        ):
            def critic_fn(seq_jumps, seq_prices):
                num_days = seq_prices.shape[1] - 1
                seq_inputs = [
                    seq_prices[:, :time_step]
                    for time_step in range(1, num_days + 1)
                ]
                seq_outputs = make_classical_net(
                    num_hidden_features=num_hidden_features,
                    num_hidden_layers=num_hidden_layers,
                    distributional=True)(seq_inputs)
                seq_dist_values = seq_outputs
                seq_values = []
                for time_step, dist_values in enumerate(seq_dist_values):
                    dist_probs = jnp.array([
                        binomial(num_days - time_step, weight)
                        for weight in range(num_days - time_step + 1)
                    ])
                    dist_probs /= 2**(num_days - time_step)
                    values = jnp.sum(dist_values * dist_probs, axis=-1)
                    seq_values.append(values)
                seq_values = jnp.stack(seq_values, axis=1)
                return seq_values, seq_dist_values

            return critic_fn

        critic_fn = make_classical_distributional_critic(
            num_hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers)
    elif critic_net == "quantum":
        layout = critic_net_kwargs.get("layout", "butterfly")
        train_bounds = critic_net_kwargs.get("train_bounds", False)

        def make_quantum_distributional_critic(layout="butterfly",
                                               train_bounds=False):
            def critic_fn(seq_jumps, seq_prices):
                encodings = encode(seq_jumps)
                alphas, betas = decompose_state(encodings)
                seq_outputs, seq_dist_outputs = make_quantum_net(layout)(
                    alphas, betas)
                if train_bounds:
                    utility_min = hk.get_parameter("utility_min", (num_days, ),
                                                   init=jnp.zeros)
                    utility_max = hk.get_parameter("utility_max", (num_days, ),
                                                   init=jnp.ones)
                else:
                    bounds = compute_bounds(num_days=num_days,
                                            mu=mu,
                                            sigma=sigma,
                                            strike=strike,
                                            cost_eps=cost_eps,
                                            initial_price=initial_price,
                                            num_trading_days=num_trading_days)
                    utility_min, utility_max = jnp.exp(-utility_lambda *
                                                       bounds[::-1])
                seq_expectations = utility_min + (utility_max -
                                                  utility_min) * seq_outputs
                seq_dist_expectations = jnp.array([
                    u_min + (u_max - u_min) * dist_outputs
                    for dist_outputs, u_min, u_max in zip(
                        seq_dist_outputs.transpose(1, 0, 2), utility_min,
                        utility_max)
                ]).transpose(1, 0, 2)
                return seq_expectations, seq_dist_expectations

            return critic_fn

        critic_fn = make_quantum_distributional_critic(
            layout=layout, train_bounds=train_bounds)

    actor = hk.transform(actor_fn)
    actor_opt = make_opt(opt_name=actor_opt, **actor_opt_kwargs)

    critic = hk.transform(critic_fn)
    critic_opt = make_opt(opt_name=critic_opt, **critic_opt_kwargs)

    def init(key):
        actor_params = actor.init(key,
                                  seq_jumps=jnp.zeros((1, num_days, 1)),
                                  seq_prices=jnp.zeros((1, num_days + 1)))
        critic_params = critic.init(key,
                                    seq_jumps=jnp.zeros((1, num_days, 1)),
                                    seq_prices=jnp.zeros((1, num_days + 1)))
        actor_opt_state = actor_opt.init(actor_params)
        critic_opt_state = critic_opt.init(critic_params)
        params = (actor_params, critic_params)
        opt_state = (actor_opt_state, critic_opt_state)
        return params, opt_state

    @jax.jit
    def train_step(key, params, opt_state):
        def critic_loss_fn(key, actor_params, critic_params):
            keys = jax.random.split(key, 4)
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=train_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)

            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            _, seq_dist_values = critic.apply(critic_params, keys[2],
                                              seq_jumps, seq_prices)
            if critic_net == "quantum":

                @jax.vmap
                def get_values(dist_value, idx):
                    return jax.lax.dynamic_index_in_dim(dist_value,
                                                        idx,
                                                        axis=-1)

                weights = seq_jumps[..., 0].sum(-1) + 1
                seq_values = get_values(seq_dist_values,
                                        jnp.int32(weights))[..., 0]
            else:
                seq_values = []
                for time_step, dist_values in enumerate(seq_dist_values):
                    weights = seq_jumps[:, time_step:, 0].sum(-1)
                    values = jax.vmap(jax.lax.dynamic_index_in_dim)(
                        dist_values, jnp.int32(weights))
                    seq_values.append(values)
                seq_values = jnp.stack(seq_values, axis=1)[..., 0]
            seq_returns = compute_returns(seq_rewards)[:, :-1]
            seq_returns = jnp.exp(-utility_lambda * seq_returns)
            loss = optax.huber_loss(seq_returns, seq_values)
            return loss.mean()

        def actor_loss_fn(key, actor_params, critic_params):
            keys = jax.random.split(key, 4)
            seq_jumps = sample_discrete_jumps(keys[0],
                                              num_paths=train_num_paths,
                                              num_days=num_days,
                                              num_jumps=num_jumps)
            seq_prices = compute_prices(seq_jumps,
                                        num_trading_days=num_trading_days,
                                        mu=mu,
                                        sigma=sigma,
                                        initial_price=initial_price)
            seq_deltas = actor.apply(actor_params, keys[1], seq_jumps,
                                     seq_prices)
            seq_rewards = compute_rewards(seq_prices,
                                          seq_deltas,
                                          strike=strike,
                                          cost_eps=cost_eps)
            seq_expectations, _ = critic.apply(critic_params, keys[2],
                                               seq_jumps, seq_prices)
            seq_values = -1 / utility_lambda * jnp.log(
                jnp.maximum(seq_expectations, 1E-6))
            seq_next_values = jnp.concatenate(
                [seq_values[:, 1:], seq_rewards[:, [-1]]], axis=1)
            loss = 1 / utility_lambda * jnp.exp(
                -utility_lambda * (seq_rewards[:, :-1] + seq_next_values))
            return loss.mean()

        keys = jax.random.split(key, 2)

        actor_params, critic_params = params
        actor_opt_state, critic_opt_state = opt_state

        critic_loss, critic_grads = jax.value_and_grad(
            critic_loss_fn, argnums=2)(keys[0], actor_params, critic_params)
        critic_updates, critic_opt_state = critic_opt.update(
            critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn, argnums=1)(
            keys[1],
            actor_params,
            critic_params,
        )
        actor_updates, actor_opt_state = actor_opt.update(
            actor_grads, actor_opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        params = (actor_params, critic_params)
        opt_state = (actor_opt_state, critic_opt_state)

        metrics = {'critic_loss': critic_loss, 'actor_loss': actor_loss}
        return params, opt_state, metrics

    @jax.jit
    def eval_step(key, params):
        actor_params, _ = params
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 2)
        seq_jumps = sample_discrete_jumps(keys[0],
                                          num_paths=eval_num_paths,
                                          num_days=num_days,
                                          num_jumps=num_jumps)
        seq_prices = compute_prices(seq_jumps,
                                    num_trading_days=num_trading_days,
                                    mu=mu,
                                    sigma=sigma,
                                    initial_price=initial_price)

        seq_deltas = actor.apply(actor_params, keys[1], seq_jumps, seq_prices)
        seq_bs_deltas = compute_black_scholes_deltas(
            seq_prices,
            num_days=num_days,
            num_trading_days=num_trading_days,
            mu=mu,
            sigma=sigma,
            strike=strike)
        seq_rewards = compute_rewards(seq_prices,
                                      seq_deltas,
                                      strike=strike,
                                      cost_eps=cost_eps)
        seq_bs_rewards = compute_rewards(seq_prices,
                                         seq_bs_deltas,
                                         strike=strike,
                                         cost_eps=cost_eps)
        returns = seq_rewards.sum(axis=1).mean()
        bs_returns = seq_bs_rewards.sum(axis=1).mean()
        utility = compute_utility(seq_rewards, utility_lambda=utility_lambda)
        bs_utility = compute_utility(seq_bs_rewards,
                                     utility_lambda=utility_lambda)
        metrics = {
            'utility': utility,
            'returns': returns,
            'bs_utility': bs_utility,
            'bs_returns': bs_returns,
        }
        return metrics

    return Agent(init, train_step, eval_step)


def experiment(agent_class, hparams, seed, train_steps):
    import mlflow
    from tqdm import tqdm
    client = mlflow.tracking.MlflowClient()
    num_days = hparams['num_days']
    mu = hparams['mu']
    sigma = hparams['sigma']
    lamda = hparams['utility_lambda']
    cost = hparams['cost_eps']
    num_trading_days = hparams['num_trading_days']
    run_name = f'T_{num_days}/{num_trading_days}_{mu}/{sigma}_{lamda}/{cost}'
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        for n, v in hparams.items():
            if not type(v) is dict:
                mlflow.log_param(n, v)
            else:
                for _n, _v in v.items():
                    mlflow.log_param(f'{n}.{_n}', _v)
        del hparams["agent"]
        agent = agent_class(**hparams)
        np_random = np.random.RandomState(seed=seed)
        rng_key = hk.PRNGSequence(np_random.randint(0, sys.maxsize + 1))
        params, opt_state = agent.init(next(rng_key))
        for it in tqdm(range(train_steps)):
            eval_metrics = agent.eval_step(next(rng_key), params)
            params, opt_state, train_metrics = agent.train_step(
                next(rng_key), params, opt_state)
            train_metrics, eval_metrics = jax.device_get(
                (train_metrics, eval_metrics))
            train_metrics = jax.tree_map(float, train_metrics)
            eval_metrics = jax.tree_map(float, eval_metrics)
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric('train/' + metric_name, metric_value, it)
            for metric_name, metric_value in eval_metrics.items():
                mlflow.log_metric('eval/' + metric_name, metric_value, it)


if __name__ == '__main__':
    AGENT = "D"  # "V" for Vanilla, "E" for expected, "O" for ORIGINAL, "D" for DISTRIBUTIONAL
    ACTOR = "C"  # "C" for classical, "Q" for Quantum
    CRITIC = "C"  # "C" for classical, "Q" for Quantum
    num_days = 30
    env_kwargs = dict(num_days=num_days,
                      num_jumps=1,
                      num_trading_days=num_days + 1,
                      mu=0.,
                      sigma=.2,
                      initial_price=100.,
                      strike=1.,
                      cost_eps=0.,
                      utility_lambda=0.1)
    train_kwargs = dict(train_num_paths=64, eval_num_paths=256)
    if ACTOR == "C":
        actor_kwargs = dict(actor_net="classical",
                            actor_opt="adam",
                            actor_opt_kwargs={"learning_rate": 1e-3})
    elif ACTOR == "Q":
        actor_kwargs = dict(actor_net="quantum",
                            actor_net_kwargs={
                                "layout": "pyramid",
                            },
                            actor_opt="radam",
                            actor_opt_kwargs={"learning_rate": 1e-3})
    if AGENT != "V" and CRITIC == "C":
        critic_kwargs = dict(critic_net="classical",
                             critic_net_kwargs={
                                 "num_hidden_layers": 1,
                                 "num_hidden_features": 16
                             },
                             critic_opt="adam",
                             critic_opt_kwargs={"learning_rate": 1e-4})
    elif AGENT != "V" and CRITIC == "Q":
        critic_kwargs = dict(critic_net="quantum",
                             critic_net_kwargs={
                                 "layout": "pyramid",
                                 "train_bounds": True,
                             },
                             critic_opt="adam",
                             critic_opt_kwargs={"learning_rate": 1e-3})
    if AGENT == "O":
        hparams = dict(**env_kwargs, **train_kwargs, **actor_kwargs,
                       **critic_kwargs)
        agent_class = actor_critic_original
    elif AGENT == "E":
        hparams = dict(**env_kwargs, **train_kwargs, **actor_kwargs,
                       **critic_kwargs)
        agent_class = actor_critic_expected
    elif AGENT == "D":
        hparams = dict(**env_kwargs, **train_kwargs, **actor_kwargs,
                       **critic_kwargs)
        agent_class = actor_critic_distributional
    hparams["agent"] = AGENT
    experiment(agent_class, hparams, seed=123, train_steps=5000)

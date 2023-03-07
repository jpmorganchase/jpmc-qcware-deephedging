import itertools
import sys
from typing import List, Literal, Optional, Tuple

sys.path.append("..")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import numpy as jnp
from quantum import make_ortho_fn

from .env import (
    compute_black_scholes_deltas,
    compute_bounds,
    compute_prices,
    compute_returns,
    compute_rewards,
    compute_utility,
)


def split_params(params):
    """Splits the parameters of the actor and the critic."""
    actor_params = dict((k, v) for k, v in params["~"].items() if k.startswith("actor"))
    critic_params = dict(
        (k, v) for k, v in params["~"].items() if k.startswith("critic")
    )
    return actor_params, critic_params


def join_params(actor_params, critic_params):
    """Joins the parameters of the actor and the critic."""
    params = dict(actor_params)
    params.update(critic_params)
    return {"~": params}


# Note: Reduce the number of layers for the hardware
def get_brick_idxs(
    num_qubits: int,
    num_layers: int = None,
) -> List[List[Tuple[int, int]]]:
    """Computes the indices of the RBS gates for the Brick architecture and returns a nested
    list where each inner list contains pairs of indices indicating the RBS gates to be applied
    in parallel.
    Args:
        num_qubits: Number of qubits to use in the circuit.
        num_layers: Number of layers to use in the circuit. Default is
            `None`, in which case the number of layers is logarithmic.

    Returns:
        A nested list where each inner list contains pairs of indices
        indicating the RBS gates to be applied in parallel.
    """
    if num_layers is None:
        num_layers = 1 + int(np.log2(num_qubits))
    rbs_idxs = [[(i, i + 1) for i in range(0, num_qubits - 1, 2)]]
    rbs_idxs += [[(i, i + 1) for i in range(1, num_qubits - 1, 2)]]
    return rbs_idxs * num_layers


def compute_compound(
    matrix: jnp.ndarray,
    order: int = 1,
) -> jnp.ndarray:
    """
    Args:
        unary: The orthogonal matrix used for calculating the compound matrix.
        order: The order k of the compound matrix to be computed.

    Returns:
        The compound matrix of order k.

    Raises:
        ValueError: If the order of the compound matrix is greater than the number of qubits in the orthogonal matrix.
    """
    num_qubits = matrix.shape[-1]
    # revert endian notation
    compound_1 = matrix[::-1][:, ::-1]
    if (order == 0) or (order == num_qubits):
        return jnp.ones((1, 1))
    elif order == 1:
        return compound_1
    else:
        # Compute the compound matrix of order k
        subsets = list(itertools.combinations(range(num_qubits), order))
        compounds = compound_1[subsets, ...][..., subsets].transpose(0, 2, 1, 3)
        compound_k = jnp.linalg.det(compounds)
    return compound_k


def decompose_state(
    state: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Args:
        state: The quantum state to be decomposed.

    Returns:
        A tuple containing the weight of every subspace, as a numpy array of shape (batch_dims, n+1),
        where `n` is the number of qubits, and the projection on each subspace, as a list of numpy
        arrays of shape (batch_dims, n choose k).

    """
    num_qubits = int(np.log2(state.shape[-1]))
    batch_dims = state.shape[:-1]
    # Reshape the state to be of shape (product of batch_dims, 2**num_qubits)
    state = state.reshape(-1, 2**num_qubits)
    # Select the indices of the basis states that belong to each subspace
    subspace_idxs = [
        [
            int((2 ** np.array(b)).sum())
            for b in itertools.combinations(range(num_qubits), weight)
        ]
        for weight in range(num_qubits + 1)
    ]
    # Compute the unnormalized projection on each subspace
    subspace_states = [
        state[..., subspace_idxs[weight]] for weight in range(num_qubits + 1)
    ]
    # Compute the norm of each subspace
    subspace_weights = [
        jnp.linalg.norm(subspace_state, axis=-1) for subspace_state in subspace_states
    ]
    # Compute the normalized projection on each subspace
    subspace_projs = [
        subspace_state / (alpha[..., None] + 1e-6)
        for alpha, subspace_state in zip(subspace_weights, subspace_states)
    ]
    # Reshape subspace_weight to be of shape (*batch_dims, n+1)
    subspace_weights = [a.reshape(*batch_dims, -1) for a in subspace_weights]
    # Reshape subspace_projs to be of shape (*batch_dims, n choose k)
    subspace_projs = [b.reshape(*batch_dims, -1) for b in subspace_projs]
    subspace_weights = jnp.stack(subspace_weights, -1)[..., 0, :]
    return subspace_weights, subspace_projs


def apply_compound_t(
    seq_jumps_t: jnp.ndarray,
    scope: str,
    num_days: int,
    obs_min: float,
    obs_max: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies the compound neural network to the sequence of jumps at timestep t.

    Args:
        seq_jumps_t: The sequence of jumps at timestep t.
        scope: The scope of haiku's parameters.
        num_days: The number of days T.
        obs_min: The minimum value of the observation.
        obs_max: The maximum value of the observation.

    Returns:
        The expectation of the obsrvable at timestep t, as an array of shape (batch_size, 1)
        and the expectation per subspace as an array of shape (batch_size, T-t+2+1).
    """
    time_step = seq_jumps_t.shape[-1]
    num_qubits = num_days - time_step + 2
    # Compute the RBS indices for the circuits at timestep t
    rbs_idxs = get_brick_idxs(num_qubits)
    # Compute the sequence of deltas (actions)
    # Initialize the parameters for the actor circuit at timestep t
    circuit_depth = max(time_step, 1) * len(rbs_idxs)
    init = hk.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(circuit_depth))
    num_params = sum(map(len, rbs_idxs))
    if time_step == 0:
        thetas_shape = (1, num_params)
    else:
        thetas_shape = (2 * time_step, num_params)
    thetas = hk.get_parameter(
        f"{scope}_thetas_{time_step}",
        thetas_shape,
        jnp.float32,
        init=init,
    )
    thetas = thetas.reshape(-1, num_params)
    matrices = jax.vmap(make_ortho_fn(rbs_idxs, num_qubits))(thetas)
    # Create the input quantum state for the actor circuit at timestep t
    state = jnp.ones((2 ** (num_days - time_step),)) / np.sqrt(
        2 ** (num_days - time_step)
    )
    # Add two ancilla qubits initialized to |01>
    state = jnp.kron(state, jnp.array([0.0, 1.0, 0.0, 0.0]))
    # Project the state onto subspaces
    alphas, betas = decompose_state(state)
    if time_step == 0:
        # One unary for timestep 0
        seq_matrices = matrices
    else:
        # One unary per element in the sequence of jumps until timestep t
        matrices = matrices.reshape(2, time_step, num_qubits, num_qubits)
        seq_matrices = jnp.einsum("bt,tij->btij", seq_jumps_t, matrices[1])
        seq_matrices += jnp.einsum("bt,tij->btij", 1 - seq_jumps_t, matrices[0])
        if time_step > 1:
            seq_matrices = jax.vmap(jnp.linalg.multi_dot)(seq_matrices[:, ::-1, :, :])
        else:
            seq_matrices = seq_matrices[:, 0]
    # Compoute the compound of the orthogonal matrices
    compounds = [
        jax.vmap(compute_compound, in_axes=(0, None))(seq_matrices, order)
        for order in range(num_qubits + 1)
    ]
    # Update the projections onto the subspaces
    betas = [compound @ beta for compound, beta in zip(compounds, betas)]
    # Compute the mean of the observable in every subspace
    # For the actions, the bounds are [0, 1]
    ranges = [(obs_min, obs_max) for _ in range(len(betas))]
    dist = [
        beta**2 @ jnp.linspace(*delta_range, beta.shape[-1])
        for beta, delta_range in zip(betas, ranges)
    ]
    # Compute the expected delta of the observable over all subspaces
    exp = [alpha**2 * dist for alpha, dist in zip(alphas, dist)]
    exp = jnp.array(exp).sum(0)
    return exp, dist


def make_train_nn(
    model: Literal["vanilla", "actor_critic"],
    num_days: int = 10,
    bounds: Optional[jnp.ndarray] = None,
) -> hk.Transformed:
    """Returns the quantum compounds NNs for the given model.

    Args:
        model: The model to be used for training the QNN.
        num_days: The number of days to be used for training the QNN.
        bounds: The bounds to be used for the QNN.

    Returns:
        The policy/value quantum compounds NNs used for train.
    """
    bernoulli_prob = 0.5

    @hk.transform
    def net(num_paths):
        """Creates the module for the given number of paths.

        Args:
            num_paths: The number of paths to be used for training the QNN.

        Returns:
            The generated sequences of jumps, actions, values and subspace values.
        """
        key = hk.next_rng_key()
        # Avoid computing the same action for all paths by creating a batching tree
        # The probability of seeing a particular path is still to 1/num_paths
        tree_depth = np.log2(num_paths).astype(int)
        # Loop over the time steps to compute M_t,s_t,d_t,v_t
        for time_step in range(num_days):
            # Compute the sequence of jumps
            if time_step == 0:
                # Initialize the sequence of jumps for t=0
                seq_jumps = jnp.array([[]])
            # Compute the sequence of jumps for t=1, ..., tree_depth
            elif time_step < tree_depth + 1:
                # Compute the jumps for timestep t if t < tree_depth
                seq_p = jnp.concatenate(
                    [seq_jumps, jnp.ones((seq_jumps.shape[0], 1))], axis=-1
                )
                seq_m = jnp.concatenate(
                    [seq_jumps, jnp.zeros((seq_jumps.shape[0], 1))], axis=-1
                )
                seq_jumps = jnp.concatenate([seq_p, seq_m], axis=0)
            else:
                # Sample the jumps for timestep t if t > tree_depth
                key, subkey = jax.random.split(key)
                day_jumps = jax.random.bernoulli(
                    subkey, bernoulli_prob, (seq_jumps.shape[0], 1)
                )
                seq_jumps = jnp.concatenate([seq_jumps, day_jumps], axis=-1)
            # Compute the sequence of actions
            deltas_exp, _ = apply_compound_t(
                seq_jumps_t=seq_jumps,
                scope="actor",
                num_days=num_days,
                obs_min=0.0,
                obs_max=1.0,
            )
            # Duplicate the deltas for the next step
            if time_step == 0:
                seq_deltas_exp = [deltas_exp]
            elif time_step < tree_depth + 1:
                seq_deltas_exp = [
                    jnp.concatenate([s, s], axis=0) for s in seq_deltas_exp
                ]
                seq_deltas_exp.append(deltas_exp)
            else:
                seq_deltas_exp.append(deltas_exp)
            if model != "vanilla":
                values_exp, values_dist = apply_compound_t(
                    seq_jumps_t=seq_jumps,
                    scope="critic",
                    num_days=num_days,
                    obs_min=bounds[0][time_step],
                    obs_max=bounds[1][time_step],
                )
                values_dist = jnp.stack(values_dist, axis=-1)
                if time_step == 0:
                    seq_values_exp = [values_exp]
                    seq_values_dist = [values_dist]
                elif time_step < tree_depth + 1:
                    seq_values_exp = [
                        jnp.concatenate([s, s], axis=0) for s in seq_values_exp
                    ]
                    seq_values_dist = [
                        jnp.concatenate([s, s], axis=0) for s in seq_values_dist
                    ]
                    seq_values_dist.append(values_dist)
                    seq_values_exp.append(values_exp)
                else:
                    seq_values_exp.append(values_exp)
                    seq_values_dist.append(values_dist)
        # Add last day jumps
        # Deltas are know and value are know, no need to use quantum circuits for the last day
        key, subkey = jax.random.split(key)
        last_day_jumps = jax.random.bernoulli(
            key=subkey,
            p=bernoulli_prob,
            shape=(seq_jumps.shape[0], 1),
        )
        seq_jumps = jnp.concatenate([seq_jumps, last_day_jumps], axis=-1)
        if model == "vanilla":
            seq_values_exp = None
            seq_values_sub = None
        else:
            # Compute the values of the observable in the subspace of the future jumps that will be used for the distributional critic
            # This is done after looping over all time steps and seq_jumps is already computed
            # Since jax has static computational graph and we need to use dynamic indexing, we need to use jax.lax.dynamic_index_in_dim
            @jax.vmap
            def get_values(dist_value, idx):
                return jax.lax.dynamic_index_in_dim(dist_value, idx, axis=-1)

            seq_values_sub = []
            for time_step in range(num_days):
                values_dist = seq_values_dist[time_step]
                values_sub = get_values(
                    values_dist, jnp.int32(seq_jumps[:, time_step:].sum(-1) + 1)
                )[..., 0]
                seq_values_sub.append(values_sub)
        return (
            seq_jumps,
            seq_deltas_exp,
            seq_values_exp,
            seq_values_sub,
        )

    return net


def make_test_nn(
    num_days: int = 10,
) -> hk.Transformed:
    """Returns the quantum compounds NNs for the given model.

    Args:
        num_days: The number of days to be used for training the QNN.

    Returns:
        The policy quantum compounds NNs used for test.
    """

    @hk.transform
    def net(seq_jumps):
        """Creates the module for the given sequence of paths.

        Args:
            seq_jumps: The sequence of jumps.

        Returns:
            The generated sequences of actions.
        """
        # Loop over the time steps to compute M_t,s_t,d_t,v_t
        for time_step in range(num_days):
            # Compute the sequence of actions
            deltas_exp, _ = apply_compound_t(
                seq_jumps_t=seq_jumps[:, :time_step],
                scope="actor",
                num_days=num_days,
                obs_min=0.0,
                obs_max=1.0,
            )
            # Duplicate the deltas for the next step
            if time_step == 0:
                seq_deltas_exp = [jnp.repeat(deltas_exp, seq_jumps.shape[0], axis=0)]
            else:
                seq_deltas_exp.append(deltas_exp)
        return seq_deltas_exp

    return net


def make_opt(
    opt_name: str = "adam",
    learning_rate: float = 1e-3,
) -> optax.GradientTransformation:
    """
    Creates an optimizer from the `optax` library.
    https://optax.readthedocs.io/en/latest/api.html

    Args:
      opt_name: The name of the optimizer to use.
      learning_rate: The learning rate to use.

    Returns:
      An optax optimizer.
    """
    return getattr(optax, opt_name)(learning_rate=learning_rate)


def make_train(
    num_days=10,
    num_trading_days=30,
    mu=0.0,
    sigma=0.2,
    initial_price=100.0,
    strike=1.0,
    cost_eps=0.0,
    train_num_paths=16,
    utility_lambda=0.1,
    model="vanilla",
    actor_opt="radam",
    actor_lr=1e-3,
    critic_update=None,
    critic_opt="adam",
    critic_lr=1e-2,
):

    if model != "vanilla" and critic_update is None:
        raise ValueError("Critic update must be specified for non-vanilla models.")

    if model == "vanilla":
        bounds = None
    else:
        bounds = compute_bounds(
            num_days,
            num_trading_days,
            mu,
            sigma,
            initial_price,
            strike,
            cost_eps,
        )
        bounds = jnp.exp(-utility_lambda * bounds[:, ::-1])

    # Define the QNN
    net = make_train_nn(model=model, num_days=num_days, bounds=bounds)

    # Define the actor optimizer
    actor_opt = make_opt(opt_name=actor_opt, learning_rate=actor_lr)

    if model == "vanilla":

        def init_step(key):
            """Initialize the actor network and the optimizer."""
            params = net.init(key, num_paths=1)
            opt_state = actor_opt.init(params)
            return params, opt_state

        @jax.jit
        def train_step(key, params, opt_state):
            """Train the actor network for one step."""

            def actor_loss_fn(key, actor_params):
                """Compute the loss of the actor network."""
                key, subkey = jax.random.split(key)
                # Compute the jumps and the deltas
                seq_jumps, seq_deltas_exp, _, _ = net.apply(
                    actor_params,
                    subkey,
                    train_num_paths,
                )
                # Compute the prices
                seq_prices = compute_prices(
                    seq_jumps,
                    num_trading_days=num_trading_days,
                    mu=mu,
                    sigma=sigma,
                    initial_price=initial_price,
                )
                # Compute the rewards
                seq_deltas = jnp.stack(seq_deltas_exp, axis=1)
                seq_rewards = compute_rewards(
                    seq_prices,
                    seq_deltas,
                    strike=strike,
                    cost_eps=cost_eps,
                )
                # Compute the utility
                utility = compute_utility(
                    seq_rewards,
                    utility_lambda=utility_lambda,
                )
                # Compute the loss
                loss = -utility
                return loss.mean()

            actor_params = params
            actor_opt_state = opt_state
            # Compute the loss and the gradients
            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn, argnums=1)(
                key,
                actor_params,
            )
            # Update the parameters
            actor_updates, actor_opt_state = actor_opt.update(
                actor_grads, actor_opt_state, actor_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)
            params = actor_params
            # Update the optimizer state
            opt_state = actor_opt_state
            metrics = {"actor_loss": actor_loss}

            return params, opt_state, metrics

    else:
        # Define the critic optimizer
        critic_opt = make_opt(opt_name=critic_opt, learning_rate=critic_lr)

        def init_step(key):
            """Initialize the actor and critic networks and the optimizers."""
            params = net.init(
                key,
                num_paths=1,
            )
            actor_params, critic_params = split_params(params)
            actor_opt_state = actor_opt.init(actor_params)
            critic_opt_state = critic_opt.init(critic_params)
            params = (actor_params, critic_params)
            opt_state = (actor_opt_state, critic_opt_state)
            return params, opt_state

        @jax.jit
        def train_step(key, params, opt_state):
            """Train the actor and critic networks for one step."""

            def critic_loss_fn(key, actor_params, critic_params):
                """Compute the loss of the critic network."""
                key, subkey = jax.random.split(key)
                net_params = join_params(actor_params, critic_params)
                # Compute the jumps, the deltas and the values
                (
                    seq_jumps,
                    seq_deltas_exp,
                    seq_values_exp,
                    seq_values_sub,
                ) = net.apply(net_params, subkey, train_num_paths)
                seq_deltas = jnp.stack(seq_deltas_exp, axis=1)
                seq_values = jnp.stack(seq_values_exp, axis=1)
                seq_sub = jnp.stack(seq_values_sub, axis=1)
                seq_prices = compute_prices(
                    seq_jumps,
                    num_trading_days=num_trading_days,
                    mu=mu,
                    sigma=sigma,
                    initial_price=initial_price,
                )
                seq_rewards = compute_rewards(
                    seq_prices, seq_deltas, strike=strike, cost_eps=cost_eps
                )
                # Compute future returns
                seq_returns = compute_returns(seq_rewards)[:, :-1]
                seq_returns = jnp.exp(-utility_lambda * seq_returns)
                # Compute the loss
                # The loss is scaled by the maximum value of the utility
                # Use the huber loss to avoid the gradient explosion
                if critic_update == "expected":
                    loss = optax.huber_loss(seq_values / seq_returns, 1.0)
                if critic_update == "distributional":
                    loss = optax.huber_loss(seq_sub / seq_returns, 1.0)
                return loss.mean()

            def actor_loss_fn(key, actor_params, critic_params):
                """Compute the loss of the actor network."""
                key, subkey = jax.random.split(key)
                net_params = join_params(actor_params, critic_params)
                seq_jumps, seq_deltas_exp, seq_values_exp, _ = net.apply(
                    net_params,
                    subkey,
                    train_num_paths,
                )
                seq_prices = compute_prices(
                    seq_jumps,
                    num_trading_days=num_trading_days,
                    mu=mu,
                    sigma=sigma,
                    initial_price=initial_price,
                )
                seq_deltas = jnp.stack(seq_deltas_exp, axis=1)
                seq_rewards = compute_rewards(
                    seq_prices, seq_deltas, strike=strike, cost_eps=cost_eps
                )
                # Compute the values
                seq_values = jnp.stack(seq_values_exp, axis=1)
                seq_values = -1 / utility_lambda * jnp.log(seq_values)
                # Compute the next values
                seq_next_values = jnp.concatenate(
                    [seq_values[:, 1:], seq_rewards[:, [-1]]], axis=1
                )
                # Compute the loss
                loss = (
                    1
                    / utility_lambda
                    * jnp.exp(-utility_lambda * (seq_rewards[:, :-1] + seq_next_values))
                )
                return loss.mean()

            actor_params, critic_params = params
            actor_opt_state, critic_opt_state = opt_state

            # Update the critic network
            key, subkey = jax.random.split(key)
            critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn, argnums=2)(
                subkey,
                actor_params,
                critic_params,
            )
            critic_updates, critic_opt_state = critic_opt.update(
                critic_grads, critic_opt_state, critic_params
            )
            critic_params = optax.apply_updates(critic_params, critic_updates)

            # Update the actor network
            key, subkey = jax.random.split(key)
            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn, argnums=1)(
                subkey,
                actor_params,
                critic_params,
            )
            actor_updates, actor_opt_state = actor_opt.update(
                actor_grads, actor_opt_state, actor_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)
            params = (actor_params, critic_params)
            opt_state = (actor_opt_state, critic_opt_state)

            metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss}
            return params, opt_state, metrics

    return init_step, train_step


def make_test(
    num_days=10,
    num_trading_days=30,
    mu=0.0,
    sigma=0.2,
    initial_price=100.0,
    strike=1.0,
    cost_eps=0.0,
    utility_lambda=0.1,
    model="vanilla",
    **kwargs,
):
    """Make the test function."""

    net = make_test_nn(num_days=num_days)

    @jax.jit
    def test_step(seq_jumps, params):
        # Fix the seed for evaluation
        if model == "vanilla":
            net_params = params
        else:
            net_params = join_params(params[0], params[1])
        # Compute the jumps and the deltas
        seq_deltas_exp = net.apply(net_params, None, seq_jumps)
        # Compute the prices
        seq_prices = compute_prices(
            seq_jumps,
            num_trading_days=num_trading_days,
            mu=mu,
            sigma=sigma,
            initial_price=initial_price,
        )
        # Compute the deltas
        seq_deltas = jnp.stack(seq_deltas_exp, axis=1)
        # Compute the rewards and the utility
        seq_rewards = compute_rewards(
            seq_prices, seq_deltas, strike=strike, cost_eps=cost_eps
        )
        utility = compute_utility(
            seq_rewards,
            utility_lambda=utility_lambda,
        )
        # Compute the deltas of the Black-Scholes model
        # This is suboptimal but it is used to compute a baseline
        seq_bs_deltas = compute_black_scholes_deltas(
            seq_prices,
            num_days=num_days,
            num_trading_days=num_trading_days,
            mu=mu,
            sigma=sigma,
            strike=strike,
        )
        # Compute the rewards and the utility of the Black-Scholes model
        seq_bs_rewards = compute_rewards(
            seq_prices, seq_bs_deltas, strike=strike, cost_eps=cost_eps
        )
        bs_utility = compute_utility(
            seq_bs_rewards,
            utility_lambda=utility_lambda,
        )
        metrics = {"utility": utility, "bs_utility": bs_utility}
        other = {
            "seq_prices": seq_prices,
            "seq_deltas": seq_deltas,
            "seq_rewards": seq_rewards,
            "seq_bs_rewards": seq_bs_rewards,
            "seq_bs_deltas": seq_bs_deltas,
        }
        for i in range(seq_rewards.shape[0]):
            other[f"PnL_{i}"] = seq_rewards[i].sum()
            other[f"PnL_bs_{i}"] = seq_bs_rewards[i].sum()
        return metrics, other

    return test_step

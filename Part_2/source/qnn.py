import itertools
from typing import Callable, List, Tuple

import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp


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


def make_ortho_fn(
    rbs_idxs: List[List[Tuple[int, int]]],
    num_qubits: int,
) -> Callable:
    """
    Args:
        rbs_idxs: List of RBS indices.
        num_qubits: The total number of qubits.

    Returns:
        A pure function that maps a set of parameters to an orthogonal matrix (compound 1).
    """
    rbs_idxs = [list(map(list, rbs_idx)) for rbs_idx in rbs_idxs]
    len_idxs = np.cumsum([0] + list(map(len, rbs_idxs)))

    def get_rbs_unitary(theta):
        """Returns the unitary matrix for the RBS gate."""
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        unitary = jnp.array(
            [
                [cos_theta, sin_theta],
                [-sin_theta, cos_theta],
            ]
        )
        unitary = unitary.transpose(*[*range(2, unitary.ndim), 0, 1])
        return unitary

    def orthogonal_fn(thetas):
        """Returns the orthogonal matrix for the given parameters."""
        unitaries = []
        # Compute the unitary for each layer
        for i, idxs in enumerate(rbs_idxs):
            idxs = sum(idxs, [])
            sub_thetas = thetas[len_idxs[i] : len_idxs[i + 1]]
            rbs_blocks = get_rbs_unitary(sub_thetas)
            eye_block = jnp.eye(num_qubits - len(idxs), dtype=thetas.dtype)
            permutation = idxs + [i for i in range(num_qubits) if i not in idxs]
            permutation = np.argsort(permutation)
            unitary = jsp.linalg.block_diag(*rbs_blocks, eye_block)
            unitary = unitary[permutation][:, permutation]
            unitaries.append(unitary)
        unitaries = jnp.stack(unitaries)
        if len(unitaries) > 1:
            unitary = jnp.linalg.multi_dot(unitaries[::-1])
        else:
            unitary = unitaries[0]
        return unitary[::-1][:, ::-1]

    return orthogonal_fn


def compute_compound(
    unary: jnp.ndarray,
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
    num_qubits = unary.shape[-1]
    if (order == 0) or (order == num_qubits):
        return jnp.ones((1, 1))
    elif order == 1:
        return unary
    else:
        # Compute the compound matrix of order k
        subsets = list(itertools.combinations(range(num_qubits), order))
        compounds = unary[subsets, ...][..., subsets].transpose(0, 2, 1, 3)
        compound = jnp.linalg.det(compounds)
    return compound


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
    alphas = [
        jnp.linalg.norm(subspace_state, axis=-1) for subspace_state in subspace_states
    ]
    # Compute the normalized projection on each subspace
    betas = [
        subspace_state / (alpha[..., None] + 1e-6)
        for alpha, subspace_state in zip(alphas, subspace_states)
    ]
    # Reshape the alphas to be of shape (*batch_dims, n+1)
    alphas = [alpha.reshape(*batch_dims, -1) for alpha in alphas]
    # Reshape the betas to be of shape (*batch_dims, n choose k)
    betas = [beta.reshape(*batch_dims, -1) for beta in betas]
    alphas = jnp.stack(alphas, -1)[..., 0, :]
    return alphas, betas

import quasar
import numpy as np
import collections
from math import ceil, floor, pi, sqrt, log, acos, asin, sin, atan, cos


class KP_Tree:
    """Class used to produce list of angles necessary
    for loading x corresponding to the Kerenidis-Prakash
    binary tree data structure in an online manner.
    Args:
        x (np.array): classical input vector
    """
    def __init__(self, x):
        self.x = x
        L = int(ceil(log(len(x), 2)))

        self.x = np.pad(self.x, (0, 2**L - len(self.x)), mode='constant')

        self.N = len(self.x)
        self.tree = np.zeros((2 * self.N, 2))

        for i in range(self.N):
            self.tree[i][0] = 0
        for i in range(2 * self.N):
            self.tree[i][1] = 1

        for i in range(len(x)):
            self.update(x[i], i)

    def get_tree(self):
        return self.tree

    def get_angles(self):
        return self.tree[:(self.N - 1), 1]

    def P(self, i):
        return int(np.floor(((i - 1) / 2)))

    def L(self, i):
        return int(2 * i + 1)

    def R(self, i):
        return int(2 * i + 2)

    def update(self, xi, i):
        index = self.N - 1 + i
        self.tree[index][0] = xi
        while index != 0:

            index = self.P(index)

            self.tree[index][0] = np.sqrt((self.tree[self.L(index)][0])**2 +
                                          (self.tree[self.R(index)][0])**2)

            if self.tree[index][0] != 0:
                theta = acos(self.tree[self.L(index)][0] / self.tree[index][0])
            else:
                theta = 0

            if self.tree[self.R(index)][0] < 0:
                self.tree[index][1] = 2 * np.pi - theta
            else:
                self.tree[index][1] = theta



def parallel_loader(angles, initial=True):
    """Outputs a parametrized quantum circuit
    which can load the classical vector corresponding
    to the 'angles' vector in log(len(angles)) depth.
    Args:
        angles (ndarray): list of angles used as
        parameters for the PS gates of the loader
        initial: if True then the loader is at the
        beginning of the circuit, so we perform
        an initial X gate on the top qubit.
    Returns:
        loader (quasar.Circuit)
    """

    # depth of circuit
    k = int(ceil(log(len(angles) + 1, 2)))
    # number of qubits
    N = 2**k

    #start building the circuit
    loader = quasar.Circuit()

    # initial gate to make the top qubit 1
    if initial == True:
        loader.X(int(0))

    # adding the PS gates in lexicographical order
    # increasing the indices (i:level, j:node)
    t = 0
    for i in range(1, k + 1):
        for j in range(1, 2**(i - 1) + 1):
            qubitA = int((N / 2**(i - 1)) * (j - 1))
            qubitB = int((N / 2**(i - 1)) * (j - 1) + N / (2**i))
            loader.add_gate(quasar.Gate.RBS(theta=angles[t]), (qubitA, qubitB))
            t += 1

    return loader


def get_angles(x, mode='parallel'):
    return KP_Tree(x).get_angles()

def loader(x, mode='parallel', initial=True, controlled=False):
    """Outputs a quantum circuit to load a
    normalized classical vector or matrix.
    Args:
        x: np.array of dims d.
            Classical data to be encoded in quantum states.
        mode: str (optional)
            'parallel': log(d) depth, d qubits
            'parallel-clifford': 2*log(d) depth, d qubits
            'optimized', ~sqrt(d) depth, ~sqrt(d) qubits
            'diagonal': d depth, d qubits,
            'diagonal-clifford': 2*d depth, d qubits
            'semi-diagonal': d/2 depth, d qubits
            'semi-diagonal-clifford': d depth, d qubits
        initial: bool (optional)
            True;  loader is at the beginning of circuit,
            perform an initial X gate on first qubit.
        controlled: bool (optional)
            if True, then the X gate is controlled
            by an ancilla qubit
    Returns:
        circ: quasar.Circuit
            circuit loading classical vector into quantum state.
    """
    #Input Validation

    if mode not in ['parallel']:
        raise ValueError('Mode must be "parallel"')

    if isinstance(x, np.ndarray) == False:
        raise ValueError("x is not an array.")
    errors = []
    if len(x.shape) != 1: raise ValueError("Ensure that x is a 1-d array")
    if abs(np.linalg.norm(x) - 1) > 1e-06:
        raise ValueError("x must have unit norm")

    # Start by finding the angles to feed the loader
    theta = get_angles(x, mode)

    #Check which type of loader to apply
    if (mode == 'parallel'):
        loader = parallel_loader(theta, initial)
    else:
        raise ValueError('loader must be "parallel"')
    return loader

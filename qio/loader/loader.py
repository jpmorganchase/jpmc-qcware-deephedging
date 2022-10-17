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


def KP_tree_offline(x):
    """Produces list of angles necessary for loading x
    corresponding to the Kerenidis-Prakash
    binary tree data structure in an offline manner.
    Args:
        x (np.array): classical input vector;
    Returns:
        angles (np.array): angles used as parameters for the PS gates.
        The order of the angles follows the lexicographical order
        of the tree as (i:level,j:node from top to bottom).
    """

    angles = []

    L = int(ceil(log(len(x), 2)))
    # L is the number of layers the parallel loader will have.

    x = np.pad(x, (0, 2**L - len(x)), mode='constant')

    while len(angles) < len(x) - 1:
        for l in range(0, L - 1):
            # The l-th layer has 2^l angles in [0,pi/2]
            for i in range(0, 2**l):
                slot_length = 2**(L - 1 - l)
                numerator = sum([
                    abs(xi)**2 for xi in x[(2 * i) * slot_length:(2 * i + 1) *
                                           slot_length]
                ])
                denominator = sum([
                    abs(xi)**2 for xi in x[(2 * i) * slot_length:(2 * i + 2) *
                                           slot_length]
                ])
                numerator = sqrt(numerator)
                denominator = sqrt(denominator)
                if denominator == 0:
                    angle = acos(1)
                else:
                    angle = acos(numerator / denominator)
                angles.append(angle)

        for l in (L - 1, ):
            # The l-th layer has 2^l angles in [0,pi/2]
            for i in range(0, 2**l):
                slot_length = 2**(L - 1 - l)
                leaves = x[(2 * i) * slot_length:(2 * i + 2) * slot_length]
                left_leaf = leaves[0]
                right_leaf = leaves[1]
                denominator = sum([abs(xi)**2 for xi in leaves])
                denominator = sqrt(denominator)

                if denominator == 0:
                    angle = acos(1)
                else:
                    angle = acos(left_leaf / denominator)
                if right_leaf < 0:
                    angle = 2 * pi - angle
                angles.append(angle)

    return angles


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


def opt_loader_basic(angles, initial=True):
    """Outputs a parametrized quantum circuit
   which can load the classical vector corresponding
   to the 'angles' vector by optimizing
   the number of qubits and the circuit depth.
   (Basic version)
   Args:
       angles (ndarray): list of angles used as
       parameters for the PS gates of the loader
   Returns:
       loader (quasar.Circuit)
   """

    # depth of parallel loader
    k = int(ceil(log(len(angles) + 1, 2)))

    # number of qubits for parallel loader
    N = 2**k

    # Split N into N=N1*N2, where
    # N1: parallel part, N2: sequential part

    N1 = 2**(floor(k / 2))
    N2 = 2**ceil(log(N / N1, 2))

    #building the optimized loader
    # The main idea is we do a parallel loader for an initial part of the vector
    # and then conditioned on the unary outputs of this parallel loader
    # we invoke conditionned parallel loaders for the remaining parts of the vector

    # initial gate to make the top qubit of the sequential part 1
    if initial == True:
        init = quasar.Circuit().X(N1)
    else:
        init = quasar.Circuit().CX(0, N1)

    # parallel part: parallel loader of size N1
    par_load = parallel_loader(angles[0:N1 - 1], initial=initial)

    loader = quasar.Circuit().join_in_time([init, par_load])

    # sequential part: interleaving sequence of N1 trees of size N2 each

    # Here we don't add controlled-PS composite gates, but we add controls
    # only on the CF (controls are the output qubits of the parallel loader).
    # Instead of a series of controlled CX gates we just add one CX gate at the end.
    # We add the gates by increasing the indices
    # (i:level, num: index of control qubit, j:node)

    t = 0
    for i in range(1, ceil(log(N / N1, 2)) + 1):
        for num in range(1, N1 + 1):
            for j in range(1, 2**(i - 1) + 1):

                t = N1 - 1 + (2**(i - 1) -
                              1) * N1 + (num - 1) * (2**(i - 1)) + (j - 1)
                control = num - 1
                qubitA = N1 + int((N2 / 2**(i - 1)) * (j - 1))
                qubitB = N1 + int((N2 / 2**(i - 1)) * (j - 1) + N2 / (2**i))
                loader.add_controlled_gate(quasar.Gate.CF(theta=angles[t]),
                                           (control, qubitA, qubitB))
                if (num == N1): loader.CX(qB, qA)

    return loader


def opt_loader(angles, initial=True):
    """Outputs a parametrized quantum circuit
   which can load the classical vector corresponding
   to the 'angles' vector by optimizing
   the number of qubits and the circuit depth.
   Args:
       angles (ndarray): list of angles used as
       parameters for the PS gates of the loader
   Returns:
       loader (quasar.Circuit)
   """

    # depth of parallel loader
    k = int(ceil(log(len(angles) + 1, 2)))

    # number of qubits for parallel loader
    N = 2**k

    # Split N into N=N1*N2, where
    # N1: parallel part, N2: sequential part

    N1 = 2**(floor(k / 2))
    N2 = 2**ceil(log(N / N1, 2))
    #print(N1,N2)

    #building the optimized loader
    # The main idea is we do a parallel loader for an initial part of the vector
    # and then conditioned on the unary outputs of this parallel loader
    # we invoke conditionned parallel loaders for the remaining parts of the vector

    # initial gate to make the top qubit of the sequential part 1
    if initial == True:
        init = quasar.Circuit().X(N1)
    else:
        init = quasar.Circuit().CX(0, N1)

    # parallel part: parallel loader of size N1
    par_load = parallel_loader(angles[0:N1 - 1], initial=initial)

    loader = quasar.Circuit().join_in_time([init, par_load])

    # sequential part: interleaving sequence of N1 trees of size N2 each

    # Here we will not add controlled-PS composite gates, but we add controls
    # only on the CF (controls are the output qubits of the parallel loader).
    # Instead of a series of controlled CX gates we just add one CX gate at the end.

    # We start by going through indices (i:level, num: index of control qubit, j:node)
    # to find the right angles and qubits for all gates without adding them to the circuit
    # gates: the list of gates

    gates = []

    t = 0
    for i in range(1, ceil(log(N / N1, 2)) + 1):
        for j in range(1, 2**(i - 1) + 1):
            for num in range(1, N1 + 1):

                t = N1 - 1 + (2**(i - 1) -
                              1) * N1 + (num - 1) * (2**(i - 1)) + (j - 1)
                control = num - 1
                qA = N1 + int((N2 / 2**(i - 1)) * (j - 1))
                qB = N1 + int((N2 / 2**(i - 1)) * (j - 1) + N2 / (2**i))
                gates.append([i, num, j, t, control, qA, qB])
                #loader.add_controlled_gate(PScirc(theta=angles[N1-1+t]),(control,qA,qB))
                if (num == N1):
                    gates.append([i, num, j, -1, qB, qA])  #loader.CX(qB,qA)

    # We now rearrange the order of the gates in order to get optimal depth.
    # Basic idea: We group gates together which are applied to disjoint qubits
    # so we can apply them all on the same time step.
    # gates2: the rearranged list of gates

    gates2 = []

    for i in range(1, ceil(log(N / N1, 2)) + 1):
        if N1 >= 2**(i - 1):
            for diff in range(0, N1 + 1):
                for gate in gates:
                    if gate[0] == i and gate[1] - gate[2] == diff and gate[
                            3] != -1:
                        gates2.append(gate)
            for diff in range(1, 2**(i - 1) + 1):
                for gate in gates:
                    if gate[0] == i and gate[2] - gate[1] == diff and gate[
                            3] != -1:
                        gates2.append(gate)
            for gate in gates:
                if gate[0] == i and gate[3] == -1:
                    gates2.append(gate)
        else:
            for diff in range(0, 2**(i - 1) + 1):
                for gate in gates:
                    if gate[0] == i and gate[2] - gate[1] == diff and gate[
                            3] != -1:
                        gates2.append(gate)
            for diff in range(1, N1 + 1):
                for gate in gates:
                    if gate[0] == i and gate[1] - gate[2] == diff and gate[
                            3] != -1:
                        gates2.append(gate)
            for gate in gates:
                if gate[0] == i and gate[3] == -1:
                    gates2.append(gate)

    # We are now ready to add the gates from the list 'gates2' to the loader

    # loading the gates from gates2 to the circuit
    for gate in gates2:
        if gate[3] == -1:
            loader.CX(gate[4], gate[5])
        else:
            loader.add_controlled_gate(quasar.Gate.CF(theta=angles[gate[3]]),
                                       (gate[4], gate[5], gate[6]))

    return loader



def diagonal_angles(x):
    """Outputs the angles that are useful
    for the diagonal loader circuit
    Args:
        x (numpy.array): input vector
    Returns:
        angles (numpy.array)
    """
    n = len(x)
    prod = 1
    angles = np.zeros(n-1)
    angles[0] = acos(x[0])
    for i in range(1, n-1):
        prod *= sin(angles[i-1])
        if prod==0:
            break
        else:
            angles[i] = acos(np.clip(x[i]/prod, -1, 1))

    angles[-1] *= np.sign(x[-1])

    return angles


            
            

def diagonal_loader(angles, initial = True):
    """Outputs the diagonal loader circuit
    Args:
        angles (numpy.array): the angles used
        as parameters of the RBS gates for the
        diagonal loader
        initial: if True then we perform an initial
        X gate on the first qubit
    Returns:
        circuit (quasar.Circuit)
    """
    n = len(angles) + 1

    circuit = quasar.Circuit()
    if initial:
        circuit.X(0)

    for i in range(n-1):
        circuit.add_gate(quasar.Gate.RBS(angles[i]), (i, i+1))

    return circuit


def semi_diagonal_angles(x):
    """Outputs the angles that are useful
    for the semi-diagonal loader circuit
    Args:
        x (numpy.array): input vector
    Returns:
        angles (numpy.array)
    """

    x = x/np.linalg.norm(x)
    n = len(x)
    angles = np.zeros(n-1)

    if n==2:
        angles[0] = acos(x[0])
        angles[-1] *= np.sign(x[-1])
        return angles

    if x[1]!=0:
        angles[n + (n%2) - 3] = atan(x[0]/x[1])
    elif x[0]!=0:
        angles[n + (n%2) - 3] = np.pi/2
    else:
        angles[n + (n%2) - 3] = 0

    k=0

    for i in range(n + (n%2) - 4, 1, -2):
        k+=1
        if x[k+1]!=0:
            angles[i-1] = atan(x[k]/(x[k+1]*cos(angles[i+1])))
        else:
            angles[i-1] = np.pi/2

    if x[(n+1)//2 - 1]==0:
        if angles[1]==0:
            angles[0] = np.pi/2
        else:
            angles[0] = acos(np.sum(x[:(n+1)//2 - 1]))
    else:
        if x[(n+1)//2 - 1]/cos(angles[1]) > 1:
            angles[0] = 0
        elif x[(n+1)//2 - 1]/cos(angles[1]) < -1:
            angles[0] = np.pi
        else:
            angles[0] = acos(x[(n+1)//2 - 1]/cos(angles[1]))

    if np.abs(angles[0] - 0) < 1e-5:
        return angles

    prod = 1
    for i in range(3, n-(n%2)+1, 2):
        prod *= sin(angles[i-3])
        if np.abs(x[(n+i)//2 - 1]/prod) > 1: #np.sum(np.abs(prod-x[(n+i)//2 - 1])) < 1e-5:
            angles[i-1] = 0
        elif np.abs(x[(n+i)//2 - 1]/prod) < -1: #np.sum(np.abs(prod-x[(n+i)//2 - 1])) < 1e-5:
            angles[i-1] = np.pi
        else:
            angles[i-1] = acos(x[(n+i)//2 - 1]/prod)

    if n%2==0:
        angles[-1] *= np.sign(x[-1])
    else:
        angles[-2] *= np.sign(x[-1])

    return angles


def semi_diagonal_loader(angles, initial=True):
    """Outputs the semi-diagonal loader circuit
    Args:
        angles (numpy.array): the angles used
        as parameters of the RBS gates for the
        diagonal loader.
        initial: if True then we perform an initial
        X gate on the first qubit.
    Returns:
        circuit (quasar.Circuit)
    """

    circuit = quasar.Circuit()
    n = len(angles) + 1
    h = (n+1)//2 - 1
    l = (n+1)//2
    if initial:
        circuit.X(h)
    circuit.add_gate(quasar.Gate.RBS(angles[0]), (h,l))
    for i in range((n-2)//2):
        circuit.add_gate(quasar.Gate.RBS(angles[2*(i+1)-1]), (h, h-1))
        circuit.add_gate(quasar.Gate.RBS(angles[2*(i+1)]), (l, l+1))
        h-=1
        l+=1
    if n%2==1:
        circuit.add_gate(quasar.Gate.RBS(angles[-1]), (1, 0))
    return circuit


def diagonal_clifford_loader(angles, controlled=False):
    """Outputs the diagonal Clifford loader circuit
    Args:
        angles (numpy.array): the angles used
        as parameters of the RBS gates for the
        diagonal clifford loader.
        controlled: if True then the X gate is controlled
        by an ancilla qubit.
    Returns:
        circuit (quasar.Circuit)
    """

    N = len(angles) + 1
    circuit = diagonal_loader(angles, False).adjoint()

    if controlled:
        adjoint_circuit = diagonal_loader(angles, False)
        anc = quasar.Circuit().I(0)
        full_circuit = quasar.Circuit.join_in_qubits([anc, circuit])
        full_circuit.CX(0,1)
        full_circuit = full_circuit.add_gates(adjoint_circuit, tuple(range(1, N+1)))
        full_circuit.remove_gate(qubits=(0, ), times=(0, ))
    else:
        adjoint_circuit = diagonal_loader(angles, True)
        full_circuit = quasar.Circuit.join_in_time([circuit, adjoint_circuit])

    return full_circuit


def semi_diagonal_clifford_loader(angles, controlled=False):
    """Outputs the semi-diagonal Clifford loader circuit
    Args:
        angles (numpy.array): the angles used
        as parameters of the RBS gates for the
        diagonal clifford loader.
        controlled: if True then the X gate is controlled
        by an ancilla qubit.
    Returns:
        circuit (quasar.Circuit)
    """

    N = len(angles) + 1
    circuit = semi_diagonal_loader(angles, False).adjoint()
    if controlled:
        adjoint_circuit = semi_diagonal_loader(angles, False)
        anc = quasar.Circuit().I(0)
        full_circuit = quasar.Circuit.join_in_qubits([anc, circuit])
        full_circuit.CX(0,(N+1)//2)
        full_circuit = full_circuit.add_gates(adjoint_circuit, tuple(range(1, N+1)))
        full_circuit.remove_gate(qubits=(0, ), times=(0, ))
    else:
        adjoint_circuit = semi_diagonal_loader(angles)
        full_circuit = quasar.Circuit.join_in_time([circuit, adjoint_circuit])
    return full_circuit


# Helper functions for the Clifford loader
def half_circ(angles):
    """Outputs a recursively built quantum circuit
    that is part of the Clifford loader (first version)
    Args:
        angles (numpy.array): the angles used as parameters
        of the clifford loader for the RBS gates
    Returns:
        C (quasar.Circuit)
    """
    k = int(ceil(log(len(angles) + 1, 2)))
    N = 2**k

    circuit = quasar.Circuit()

    if k==1:
        circuit.add_gate(quasar.Gate.RBS(angles[0]), (0, 1))
        return circuit
    else:
        circuit.add_gates(half_circ_prime(angles[(N-1)//2:-1]), tuple(range(N//2)))
        circuit.add_gates(half_circ(angles[:(N-1)//2]), tuple(range(N//2, N)))
        circuit.CZ(1, 0)
        circuit.add_gate(quasar.Gate.RBS(angles[-1]), (0, N//2))
        circuit.CZ(1, 0)
        return circuit

def half_circ_prime(angles):
    """Outputs a recursively built quantum circuit
    that is part of the Clifford loader (second version)
    Args:
        angles (numpy.array): the angles used as parameters
        of the clifford loader for the RBS gates
    Returns:
        C' (quasar.Circuit)
    """

    k = int(ceil(log(len(angles) + 1, 2)))
    N = 2**k
    circuit = quasar.Circuit()

    if k==1:
        circuit.add_gate(quasar.Gate.RBS(angles[0]), (0, 1))
        return circuit
    else:
        circuit.add_gates(half_circ_prime(angles[(N-1)//2:-1]), tuple(range(N//2)))
        circuit.add_gates(half_circ_prime(angles[:(N-1)//2]), tuple(range(N//2, N)))
        circuit.CZ(1, 0)
        circuit.add_gate(quasar.Gate.RBS(angles[-1]), (0, N//2))
        circuit.CZ(1, 0)
        circuit.CX((N//2)+1, N//2)
        circuit.CX(N//2, 1)
        return circuit

def tree_ind(depth,t=1):
    """Helper function that outputs indexes
    of the angles afer a traversal of the KP tree
    Args:
        depth (int): depth of the tree
        t (int): index of the theta at the root
    Returns:
        List of new indexes for the angles (numpy.array)
    """
    if depth==1:
        return np.array([t])
    else:
        return np.append(t, np.append(tree_ind(depth-1,t*2), tree_ind(depth-1, t*2+1)))

def parallel_clifford_loader(angles, controlled=False):
    """Outputs a paramaterized quantum circuit
    which can load a classical vector into a quantum
    state given any input state. This corresponds to a
    unitary operator described as the Clifford loader.
    Args:
        angles (numpy.array): list of angles used as
        parameters for the loader's RBS gates
    Returns:
        loader (quasar.Circuit)
    """

    # depth of the loader
    k = int(ceil(log(len(angles) + 1, 2)))

    # number of qubits
    N = 2**k

    # rearranging the angles
    angles = angles[tree_ind(k)-1]
    angles = -angles[::-1]

    # constructing the circuit
    circuit = half_circ(angles)
    copy_circuit = circuit.copy()
    if controlled:
        anc = quasar.Circuit().I(0)
        full_circuit = quasar.Circuit.join_in_qubits([anc, circuit])
        full_circuit.CX(0,1)
        full_circuit = full_circuit.add_gates(copy_circuit.adjoint(), tuple(range(1,N+1)))
        full_circuit.remove_gate(qubits=(0, ), times=(0, ))
    else:
        circuit.X(0)
        full_circuit = circuit.add_gates(copy_circuit.adjoint(), tuple(range(N)))

    return full_circuit

def get_angles(x, mode='parallel'):
    if mode in ['parallel', 'parallel-clifford', 'optimized']:
        return KP_Tree(x).get_angles()
    elif mode in ['diagonal', 'diagonal-clifford']:
        return diagonal_angles(x)
    elif mode in ['semi-diagonal', 'semi-diagonal-clifford']:
        return semi_diagonal_angles(x)

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

    if mode not in ['parallel', 'parallel-clifford', 'optimized', 'diagonal',\
                    'diagonal-clifford', 'semi-diagonal', 'semi-diagonal-clifford']:
        raise ValueError('Mode must be "parallel" or "parallel-clifford" or "optimized" or "diagonal"\
                    or "diagonal-clifford" or "semi-diagonal" or "semi-diagonal-clifford"')

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
    elif (mode=='parallel-clifford'):
            loader = parallel_clifford_loader(theta, controlled)
    elif (mode == 'optimized'):
        loader = opt_loader(theta, initial)
    elif (mode == 'diagonal'):
        loader = diagonal_loader(theta, initial)
    elif (mode=='diagonal-clifford'):
        loader = diagonal_clifford_loader(theta, controlled)
    elif (mode=='semi-diagonal'):
        loader = semi_diagonal_loader(theta, initial)
    else:
        loader = semi_diagonal_clifford_loader(theta, controlled)
    return loader

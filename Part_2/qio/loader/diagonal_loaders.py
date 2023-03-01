import quasar

import numpy as np
from math import log, ceil, acos, sin, cos, atan

def diagonal_angles(x):
    """Outputs the angles that are useful
    for the diagonal loader circuit
    Args:
        x (numpy.array): input vector
    Returns:
        angles (numpy.array)
    """
    x = x[:]/np.linalg.norm(x)
    n = len(x)
    prod = 1
    angles = np.zeros(n-1)
    for i in range(n-1):
        if i==0:
            angles[i] = acos(x[i])
        else:
            prod *= sin(angles[i-1])
            angles[i] = acos(x[i]/prod)
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

    n = len(x)
    x = x[:]/np.linalg.norm(x)
    angles = np.zeros(n-1)

    if n==2:
        angles[0] = acos(x[0])
        angles[-1] *= np.sign(x[-1])
        return angles
    
    angles[n + (n%2) - 3] = atan(x[0]/x[1])
    k=0
    for i in range(n + (n%2) - 4, 1, -2):
        k+=1
        angles[i-1] = atan(x[k]/(x[k+1]*cos(angles[i+1])))

    if n==2:
        angles[0] = acos(x[0])
    else:
        angles[0] = acos(x[(n+1)//2 - 1]/cos(angles[1]))

    prod = 1
    for i in range(3, n-(n%2)+1, 2):
        prod *= sin(angles[i-3])
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
        adjoint_circuit = simple_loader(angles, False)
        anc = quasar.Circuit().I(0)
        full_circuit = quasar.Circuit.join_in_qubits([anc, circuit])
        full_circuit.CX(0,1)
        full_circuit = full_circuit.add_gates(adjoint_circuit, tuple(range(1, N+1)))
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
    else:
        adjoint_circuit = semi_diagonal_loader(angles)
        full_circuit = quasar.Circuit.join_in_time([circuit, adjoint_circuit])
    return full_circuit


def dloader(x, mode='semi-diagonal', type='standard', initial=True, controlled=False):
    """Outputs a quantum circuit to load a
    normalized classical vector or matrix.
    Args:
        x: np.array of dims d.
            Classical data to be encoded in quantum states.
        mode, type: str (optional), str(optional)
            'diagonal', 'standard': d depth, d qubits,
            'diagonal', 'clifford': 2*d depth, d qubits
            'semi-diagonal', 'standard': d/2 depth, d qubits
            'semi-diagonal', 'clifford': d depth, d qubits
        initial: bool (optional)
            True;  loader is at the beginning of circuit,
            perform an initial X gate on first qubit.
        controlled: bool (optional)
            if True, the X gate of the clifford loader is
            controlled by an ancilla qubit.
    Returns:
        circ: quasar.Circuit
            circuit loading classical vector into quantum state.
    """
    #Input Validation

    if (mode != "diagonal" \
    and mode != "semi-diagonal") \
    or (type != "standard" \
    and type != "clifford"):
        raise ValueError('Mode must be "diagonal" or "semi-diagonal" and \
        type must be "standard" or "clifford"')
    if isinstance(x, np.ndarray) == False:
        raise ValueError("x is not an array.")
    errors = []
    if len(x.shape) != 1: raise ValueError("Ensure that x is a 1-d array")
    if abs(np.linalg.norm(x) - 1) > 1e-06:
        raise ValueError("x must have unit norm")

    # get the right angles to use and choose the type of loader to apply

    if mode=="diagonal":
        theta = diagonal_angles(x)
        if type=="standard":
            loader = diagonal_loader(theta, initial)
        else:
            loader = diagonal_clifford_loader(theta, controlled)
    else:
        theta = semi_diagonal_angles(x)
        if type=="standard":
            loader = semi_diagonal_loader(theta, initial)
        else:
            loader = semi_diagonal_clifford_loader(theta, controlled)

    return loader

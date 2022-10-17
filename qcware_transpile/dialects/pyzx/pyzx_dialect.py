import pyzx  # type: ignore
from qcware_transpile.gates import GateDef, Dialect
from qcware_transpile.circuits import Circuit
from qcware_transpile.instructions import Instruction
from pyrsistent import pset, pmap
from pyrsistent.typing import PSet, PMap
from typing import Tuple, Any, Set, Sequence, Generator, Dict
from inspect import isclass, signature
from functools import lru_cache
from icontract import require

__dialect_name__ = "pyzx"

# these gates don't translate yet as we don't have the concept of
# variable-qubit gates or no-qubit instructions
_do_not_include_instructions = {"ParityPhase", "InitAncilla", "PostSelect"}


@lru_cache(1)
def pyzx_gatethings() -> PSet[Any]:
    """
    The set of all things in qiskit which represent a gate.  We
    don't include the ParityPhase gate because it handles variable
    numbers of qubits.
    """
    possible_things = [
        x for x in pyzx.circuit.gates.__dict__.values()
        if isclass(x) and issubclass(x, pyzx.circuit.Gate) and x !=
        pyzx.circuit.Gate and x.__name__ not in _do_not_include_instructions
    ]
    return pset(possible_things)


def parameter_names_from_gatething(thing: pyzx.circuit.Gate) -> PSet[str]:
    sig = signature(thing.__init__)
    result = set(sig.parameters.keys()).difference(
        {'self', 'target', 'control', 'ctrl1', 'ctrl2', 'label'})
    return pset(result)


def number_of_qubits_from_gatething(thing: pyzx.circuit.Gate) -> int:
    # unfortunately it's not obvious from just the class how many qubits
    # the gate can operate on, and pretty much the only path forward
    # seems to be to instantiate the gate and see.  That means coming
    # up with fake parameter arguments.
    param_names = list(signature(thing.__init__).parameters.keys())
    # in pyzx rather than having a vector of target bits, you have bit names,
    # such as "ctrl1", "ctrl2", "target"
    result = 0
    if "target" in param_names:
        result = result + 1
    if "control" in param_names:
        result = result + 1
    else:
        if "ctrl1" in param_names:
            result = result + 1
        if "ctrl2" in param_names:
            result = result + 1
    return result


def gatedef_from_gatething(thing) -> GateDef:
    return GateDef(
        name=thing.__name__,
        parameter_names=parameter_names_from_gatething(thing),  # type: ignore
        qubit_ids=number_of_qubits_from_gatething(thing))


# some gates are problematic -- in particular Qiskit's "gates" which
# really just generate other gates, for example those which take a
# number of bits as an argument.  for now we are disabling those
Problematic_gatenames: PSet[str] = pset({})


def attempt_gatedefs() -> Tuple[PSet[GateDef], PSet[str]]:
    """
    Iterate through all the gate definitions and build a set of successful
    gate definitions and a list of things which could not be converted
    """
    all_things = pyzx_gatethings()
    success = set()
    failure = set()
    for thing in all_things:
        try:
            gd = gatedef_from_gatething(thing)
            if gd.name in Problematic_gatenames:
                failure.add(thing.__name__)
            else:
                success.add(gd)
        except Exception:
            failure.add(thing.__name__)
    return (pset(success), pset(failure))


def gate_defs() -> PSet[GateDef]:
    s, f = attempt_gatedefs()
    return s


def gate_names() -> PSet[str]:
    return pset([g.name for g in gate_defs()])


@lru_cache(1)
def dialect() -> Dialect:
    """
    The pyzx dialect
    """
    return Dialect(name=__dialect_name__,
                   gate_defs=gate_defs())  # type: ignore


@lru_cache(1)
def valid_gatenames() -> Set[str]:
    return {g.name for g in dialect().gate_defs}


def possible_parameter_names():
    return set().union(*[x.parameter_names for x in dialect().gate_defs])


def parameter_bindings_from_gate(gate: pyzx.circuit.Gate) -> PMap[str, Any]:
    # this is a little trickier than others.  Rather than storing parameters
    # in a dictionary, pyzx stores them as individual Gate members, ie
    # self.target, self.phi, etc.
    result = {
        str(k): getattr(gate, k)
        for k in possible_parameter_names()
        if k in signature(gate.__init__).parameters.keys()
    }
    return pmap(result)


def qubit_bindings(g: pyzx.circuit.Gate):
    """
    Gets an ordered set of qubit bindings for the given gate.
    Pyzx, rather sensibly, does gates with names bound to qubits,
    whereas we usually store an ordered list of qubits.  For pyzx,
    then,
    1 qubit -> target
    2 qubit -> control, target
    3 qubit -> ctrl1, ctrl2, target
    """
    if "ctrl1" in dir(g):
        result = [g.ctrl1, g.ctrl2, g.target]
    elif "control" in dir(g):
        result = [g.control, g.target]
    else:
        result = [g.target]
    return result


def native_instructions(
        pc: pyzx.circuit.Circuit) -> Generator[pyzx.circuit.Gate, None, None]:
    for g in pc.gates:
        yield g


@require(lambda g: g.__class__.__name__ in valid_gatenames())
def ir_instruction_from_native(g: pyzx.circuit.Gate) -> Instruction:
    return Instruction(
        gate_def=gatedef_from_gatething(g.__class__),
        parameter_bindings=parameter_bindings_from_gate(g),  # type: ignore
        bit_bindings=qubit_bindings(g))


def native_to_ir(pc: pyzx.circuit.Circuit) -> Circuit:
    """
    Return a transpile-style Circuit object from a qiskit Circuit object
    """
    instructions = list(
        (ir_instruction_from_native(x) for x in native_instructions(pc)))
    qubits = list(range(pc.qubits))
    return Circuit(
        dialect_name=__dialect_name__,
        instructions=instructions,  # type: ignore
        qubits=qubits)  # type: ignore


def pyzx_qubit_bindings(qubits: Sequence[int]):
    if len(qubits) == 3:
        result = {'ctrl1': qubits[0], 'ctrl2': qubits[1], 'target': qubits[2]}
    elif len(qubits) == 2:
        result = {'control': qubits[0], 'target': qubits[1]}
    elif len(qubits) == 1:
        result = {'target': qubits[0]}
    else:
        raise ValueError(f"invalid number of qubits: {qubits}")
    return result


def pyzx_gate_from_instruction(i: Instruction):
    """
    Create a pyzx Gate object from an instruction.  Unlike
    some other toolkits, a "gate" in pyzx is a fully-
    instantiated instruction (with qubit assignments)
    """
    gclass = getattr(pyzx.circuit.gates, i.gate_def.name)
    parms = i.parameter_bindings
    parms = parms.update(pyzx_qubit_bindings(i.bit_bindings))
    gate = gclass(**parms)
    return gate


def ir_to_native(c: Circuit) -> pyzx.Circuit:
    """
    Make a pyzx circuit from a qcware_transpile Circuit
    """
    # qiskit wants the number of qubits first.
    num_qubits = max(c.qubits) - min(c.qubits) + 1
    result = pyzx.Circuit(num_qubits)
    for instruction in c.instructions:
        g = pyzx_gate_from_instruction(instruction)
        result.add_gate(g)
    return result


def native_circuits_are_equivalent(c1: pyzx.Circuit, c2: pyzx.Circuit) -> bool:
    """
    Whether or not two circuits are equivalent.  This is used to check
    for equivalence of the sequence of instructions, for testing; while
    pyzx has a more complicated test for equivalence, we here check only
    that the series of gates are the same
    """
    return c1.gates == c2.gates  # c1.verify_equality(c2) fails with FSim gates


def audit(c: pyzx.Circuit) -> Dict:
    """
    Retrieve a dictionary with various members indicating
    any aspects of the circuit which would make it not
    convertible to the IR
    """
    invalid_gate_names = set()
    for g in native_instructions(c):
        if g.__class__.__name__ not in valid_gatenames():
            invalid_gate_names.add(g.__class__.__name__)
    result = {}
    if len(invalid_gate_names) > 0:
        result['invalid_gate_names'] = invalid_gate_names
    return result

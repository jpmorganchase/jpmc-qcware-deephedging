import qiskit  # type: ignore
from qcware_transpile.gates import GateDef, Dialect
from qcware_transpile.circuits import Circuit
from qcware_transpile.instructions import Instruction
from qcware_transpile.helpers import map_seq_to_seq_unique
from pyrsistent import pset, pmap, pvector
from pyrsistent.typing import PSet, PMap
from typing import Tuple, Any, Set, Generator, List, Dict, Callable
from inspect import isclass, signature
from functools import lru_cache
from icontract import require

__dialect_name__ = "qiskit"


@lru_cache(1)
def qiskit_gatethings() -> PSet[Any]:
    """
    The set of all things in qiskit which represent a gate
    """
    possible_things = qiskit.circuit.library.__dict__.values()
    return pset({
        x
        for x in possible_things
        if isclass(x) and issubclass(x, qiskit.circuit.Instruction)
    })


@lru_cache(1)
def name_to_class() -> PMap[str, type]:
    return pmap({gatething_name(c): c for c in attempt_gatedefs()[2]})


@lru_cache(1)
def class_to_name() -> PMap[type, str]:
    """
    Returns a map of classes to qiskit names
    """
    return pmap({c: gatething_name(c) for c in attempt_gatedefs()[2]})


def parameter_names_from_gatething(thing: qiskit.circuit.Gate) -> PSet[str]:
    sig = signature(thing.__init__)
    result = set(sig.parameters.keys()).difference(
        {'self', 'label', 'ctrl_state'})
    return pset(result)


def number_of_qubits_from_gatething(thing: qiskit.circuit.Gate) -> int:
    # unfortunately it's not obvious from just the class how many qubits
    # the gate can operate on, and pretty much the only path forward
    # seems to be to instantiate the gate and see.  That means coming
    # up with fake parameter arguments.
    param_names = parameter_names_from_gatething(thing)
    # Obviously not every parameter can be a float, but let's start
    # there.  setting everything to zero also takes care of the
    # ctrl_state parameter which doesn't matter so much for basic gate
    # instantiation, but could affect specialized applications
    # also note that the standard for qiskit is that the first X bits of
    # a controlled gate are the control bits
    # ("In the new gate the first ``num_ctrl_qubits``
    # of the gate are the controls."), see
    # https://github.com/Qiskit/qiskit-terra/qiskit/circuit/controlledgate.py
    params = {k: 0 for k in param_names}
    g = thing(**params)
    return g.num_qubits


def gatething_name(thing: type) -> str:
    """Attempts to get the qiskit "name" from a qiskit gate class.  This
    is a mild irritation-- qiskit does a few things like lists of basis
    gates and transpilation targets through the "name" of a gate
    rather than the gate class.  That's fine, but the problem is you
    have to create an instance of the gate to get its "name"
    """
    parameter_names = parameter_names_from_gatething(thing)
    # we're so far assuming it's ok to call gate instantiation with
    # all parameters being the float 0
    parameters = {name: 0 for name in parameter_names}
    gate = thing(**parameters)
    return gate.name


@lru_cache(128)
def gatedef_from_gatething(thing) -> GateDef:
    return GateDef(
        name=gatething_name(thing),
        parameter_names=parameter_names_from_gatething(thing),  # type: ignore
        qubit_ids=number_of_qubits_from_gatething(thing))


# some gates are problematic -- in particular Qiskit's "gates" which
# really just generate other gates, for example those which take a
# number of bits as an argument.  for now we are disabling those.
# we also disable some gates which resolve to identical "names"
# such as C3XGate and C4XGate, or C3SXGate and C3XGate
Problematic_gatenames = pset({"ms", "mcx", "barrier", "c3sx"})


@lru_cache(1)
def attempt_gatedefs() -> Tuple[PSet[GateDef], PSet[str], PSet[type]]:
    """
    Iterate through all the gate definitions and build a set of successful
    gate definitions and a list of things which could not be converted
    """
    all_things = qiskit_gatethings()
    success = set()
    successful_things = set()
    failure = set()
    for thing in all_things:
        try:
            gd = gatedef_from_gatething(thing)
            if gd.name in Problematic_gatenames:
                failure.add(thing.__name__)
            else:
                success.add(gd)
                successful_things.add(thing)
        except Exception:
            failure.add(thing.__name__)
    return (pset(success), pset(failure), pset(successful_things))


def gate_defs() -> PSet[GateDef]:
    return attempt_gatedefs()[0]


def gate_names() -> PSet[str]:
    return pset([g.name for g in gate_defs()])


@lru_cache(1)
def dialect() -> Dialect:
    """
    The qiskit dialect
    """
    return Dialect(name=__dialect_name__,
                   gate_defs=gate_defs())  # type: ignore


@lru_cache(1)
def valid_gatenames() -> Set[str]:
    return {g.name for g in dialect().gate_defs}


@lru_cache(128)
def gate_init_parameter_names(initfunc: Callable) -> List[str]:
    return [k for k, v in signature(initfunc).parameters.items()]


def parameter_bindings_from_gate(gate: qiskit.circuit.Gate) -> PMap[str, Any]:
    # sometimes transpile makes ParameterExpressions and here we try to
    # coerce those into floats... which may not always be the right idea
    values = [
        x
        if not isinstance(x, qiskit.circuit.ParameterExpression) else float(x)
        for x in gate.params
    ]
    names = gate_init_parameter_names(gate.__init__)
    return map_seq_to_seq_unique(names[0:len(values)], values)


def native_instructions(
    qc: qiskit.QuantumCircuit
) -> Generator[Tuple[qiskit.circuit.Gate, List[qiskit.circuit.Qubit],
                     List[qiskit.circuit.Clbit]], None, None]:
    """
    Iterates over the circuit.  Does *NOT* reverse the circuit beforehand,
    because that elides the reversed qregs for mapping.  We just skip
    barriers, because they are problematic (they are multi-qubit, which
    we don't handle well yet, and they are unsupported by any other kit).
    """
    gates_to_skip = {'barrier'}
    for (instruction, qubits, clbits) in qc.data:
        if instruction.name not in gates_to_skip:
            yield instruction, qubits, clbits


@require(lambda qubit, qubits: qubit in qubits)
def raw_qubit_index(qubit: qiskit.circuit.Qubit,
                    qubits: List[qiskit.circuit.Qubit]) -> int:
    """
    Given a qubit object and a list of quantum registers, 
    calculates the "Raw qubit index"
    """
    return qubits.index(qubit)


@require(lambda clbit, clbits: clbit in clbits)
def raw_clbit_index(clbit: qiskit.circuit.Clbit,
                    clbits: List[qiskit.circuit.Clbit]) -> int:
    """
    Given a qubit object and a list of quantum registers, 
    calculates the "Raw qubit index"
    """
    return clbits.index(clbit)


def normalized_instructions(c: qiskit.QuantumCircuit):
    """Custom generator that loops through a circuit giving the instruction
    and "normalized" qubits and clbits (ie from 0->max rather than by register)
    """
    gates_to_skip = {'barrier'}
    for instruction, qubits, clbits in c.data:
        if instruction.name not in gates_to_skip:
            yield instruction, [c.qubits.index(x) for x in qubits
                                ], [c.clbits.index(x) for x in clbits]


@require(lambda gate: gate.name in valid_gatenames())
def ir_instruction_from_native(
        gate: qiskit.circuit.Gate, qubits: List[int], clbits: List[int]) -> Instruction:
    return Instruction(
        gate_def=gatedef_from_gatething(gate.__class__),
        parameter_bindings=parameter_bindings_from_gate(gate),  # type: ignore
        bit_bindings=qubits,
        # this below must be a pvector to handle some hashing
        metadata={
            'clbits':
            pvector(clbits)
        } if len(clbits) > 0 else {})  # type: ignore


def native_to_ir(qc: qiskit.QuantumCircuit) -> Circuit:
    """
    Return a transpile-style Circuit object from a qiskit Circuit object
    """
    rqc = qc.reverse_bits()
    instructions = list(
        ir_instruction_from_native(x[0], x[1], x[2])
        for x in normalized_instructions(rqc))
    qubits = list(range(qc.num_qubits))
    clbits = list(range(qc.num_clbits))
    return Circuit(
        dialect_name=__dialect_name__,
        instructions=instructions,  # type: ignore
        qubits=qubits,
        metadata={'clbits': pvector(clbits)})  # type: ignore


def qiskit_gate_from_instruction(i: Instruction):
    """
    Create a qiskit Gate object from an instruction
    """
    gclass = name_to_class()[i.gate_def.name]
    gate = gclass(**i.parameter_bindings)
    return gate


def ir_to_native(c: Circuit) -> qiskit.QuantumCircuit:
    """
    Make a qiskit circuit from a qcware_transpile Circuit
    """
    # qiskit wants the number of qubits first.
    num_qubits = max(c.qubits) - min(c.qubits) + 1
    _clbits = c.metadata.get('clbits',[]) if c.metadata is not None else {}
    # this is slightly different because we only list classical bits used and assume
    # they start from 0 without checking for unused edge bits.
    num_clbits = len(_clbits)
    result = qiskit.QuantumCircuit(
        num_qubits) if num_clbits == 0 else qiskit.QuantumCircuit(
            num_qubits, num_clbits)
    for instruction in c.instructions:
        g = qiskit_gate_from_instruction(instruction)
        if 'clbits' in instruction.metadata:
            result.append(g, instruction.bit_bindings,
                          instruction.metadata['clbits'])
        else:
            result.append(g, instruction.bit_bindings)
    result = result.reverse_bits()
    return result


def native_circuits_are_equivalent(c1: qiskit.QuantumCircuit,
                                   c2: qiskit.QuantumCircuit) -> bool:
    """
    Whether or not two circuits are equivalent.  Not having a test_equivalence
    method here, we brute-force it by evaluating statevectors, but this does not
    function correctly with measurement gates.
    """
    instruction_pairs = zip(normalized_instructions(c1),
                            normalized_instructions(c2))
    return all((x[0] == x[1] for x in instruction_pairs))


def audit(c: qiskit.QuantumCircuit) -> Dict:
    """
    Retrieve a dictionary with various members indicating
    any aspects of the circuit which would make it not
    convertible to the IR
    """
    # check for classical instructions
    # unhandled_classical_instructions = set()
    # rqc = c.reverse_bits()
    # for (instruction, qubits, cbits) in rqc.data:
    #     if (len(cbits) != 0):
    #         unhandled_classical_instructions.add(
    #             instruction.__class__.__name__)

    invalid_gate_names = set()
    for g, qubits, clbits in normalized_instructions(c):
        if g.name not in valid_gatenames():
            invalid_gate_names.add(gatething_name(g))
    result = {}
    if len(invalid_gate_names) > 0:
        result['invalid_gate_names'] = invalid_gate_names

    # if len(unhandled_classical_instructions) != 0:
    #     result['unhandled_classical_instructions'] = \
    #         unhandled_classical_instructions
    return result

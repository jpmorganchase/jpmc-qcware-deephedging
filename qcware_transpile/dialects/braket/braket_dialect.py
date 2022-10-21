import braket.circuits
from qcware_transpile.gates import GateDef, Dialect
from qcware_transpile.circuits import Circuit
from qcware_transpile.instructions import Instruction
from pyrsistent import pmap, pset
from pyrsistent.typing import PMap, PSet
from typing import Tuple, Any, Set, Generator, List, Dict
from inspect import isclass, signature
from functools import lru_cache
from icontract import require

__dialect_name__ = "braket"

# don't include the AngledGate and Gate base classes in
# the set of all things in braket which represent a gate.
_do_not_include_instructions = pset({'AngledGate', 'Gate'})


@lru_cache(1)
def braket_gatethings() -> PSet[Any]:
    """
    The set of all things in braket which represent a gate.
    """
    possible_things = dir(braket.circuits.gates)
    return pset([
        getattr(braket.circuits.gates, x) for x in possible_things
        if isclass(getattr(braket.circuits.gates, x))
        and issubclass(getattr(braket.circuits.gates, x), braket.circuits.Gate)
        and x not in _do_not_include_instructions
    ])


def parameter_names_from_gatething(thing: braket.circuits.Gate) -> PSet[str]:
    sig = signature(thing.__init__)
    result = set(sig.parameters.keys()).difference({'self', 'display_name'})
    return pset(result)


def number_of_qubits_from_gatething(thing: braket.circuits.Gate) -> int:
    # unfortunately it's not obvious from just the class how many qubits
    # the gate can operate on, and pretty much the only path forward
    # seems to be to instantiate the gate and see.  That means coming
    # up with fake parameter arguments. Some gate initialization methods
    # accept an angle parameter, which we set to zero.
    param_names = parameter_names_from_gatething(thing)
    params = {k: 0 for k in param_names}
    g = thing(**params)
    return g.qubit_count


@lru_cache(128)
def gatedef_from_gatething(thing: braket.circuits.Gate) -> GateDef:
    return GateDef(name=thing.__name__,
                   parameter_names=parameter_names_from_gatething(thing),
                   qubit_ids=number_of_qubits_from_gatething(thing))



# the Unitary gate is problematic since it allows for
# the construction of a gate with an arbitrary matrix.
# for now we will disable the Unitary gate.
Problematic_gatenames = pset({'Unitary'})


@lru_cache(1)
def attempt_gatedefs() -> Tuple[PSet[GateDef], PSet[str], PSet[type]]:
    """
    Iterate through all the gate definitions and build a set of successful
    gate definitions and a list of things which could not be converted
    """
    all_things = braket_gatethings()
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


@lru_cache(1)
def dialect() -> Dialect:
    """
    The braket dialect
    """
    return Dialect(name=__dialect_name__, gate_defs=gate_defs())



@lru_cache(1)
def valid_gatenames() -> Set[str]:
    return {g.name for g in dialect().gate_defs}


def parameter_bindings_from_gate(gate: braket.circuits.Gate) -> PMap[str, Any]:
    param_names = parameter_names_from_gatething(gate)
    result = {}
    for param_name in param_names:
        result[param_name] = getattr(gate, param_name)
    return pmap(result)


def occupy_empty_qubits(qc: braket.circuits.Circuit) -> braket.circuits.Circuit:
    # insert identity gate on empty qubits because
    # circuit execution on braket backends requires
    # contiguous qubit indices 
    result = braket.circuits.Circuit()
    for instruction in qc.instructions:
        result.add_instruction(instruction)
    if max(qc.qubits) >= len(qc.qubits):
        qubits = pset(int(q) for q in qc.qubits)
        empty_qubits = pset(range(max(qc.qubits)+1)).difference(qubits)
        for q in empty_qubits:
            result.add_instruction(braket.circuits.Instruction(braket.circuits.Gate.I(), q))
    return result


def native_instructions(
    qc: braket.circuits.Circuit
) -> Generator[Tuple[braket.circuits.instruction.InstructionOperator,
                     List[int]], None, None]:
    for instruction in qc.instructions:
        if isinstance(instruction.operator, braket.circuits.Gate):
            qubits = [int(qubit) for qubit in instruction.target]
            yield instruction.operator, qubits


@require(lambda gate: gate.name in valid_gatenames())
def ir_instruction_from_native(gate: braket.circuits.instruction.InstructionOperator, qubits: List[int]) -> Instruction:
    return Instruction(
        gate_def=gatedef_from_gatething(gate.__class__),
        parameter_bindings=parameter_bindings_from_gate(gate),
        bit_bindings=qubits)


def native_to_ir(qc: braket.circuits.Circuit) -> Circuit:
    instructions = list(
        ir_instruction_from_native(x[0], x[1])
        for x in native_instructions(qc))
    qubits = list(range(qc.qubit_count))
    return Circuit(dialect_name=__dialect_name__,
                   instructions=instructions,
                   qubits=qubits)



def braket_gate_from_instruction(i: Instruction):
    gclass = getattr(braket.circuits.Gate, i.gate_def.name)
    gate = gclass(**i.parameter_bindings)
    return gate


def ir_to_native(c: Circuit) -> braket.circuits.Circuit:
    result = braket.circuits.Circuit()
    for instruction in c.instructions:
        g = braket_gate_from_instruction(instruction)
        i = braket.circuits.Instruction(g, instruction.bit_bindings)
        result.add_instruction(i)
    return result


def native_circuits_are_equivalent(c1: braket.circuits.Circuit, c2: braket.circuits.Circuit) -> bool:
    return c1.__eq__(c2) 
    

def audit(c: braket.circuits.Circuit) -> Dict:
    invalid_gate_names = set()
    for g, qubits in native_instructions(c):
        if g.name not in valid_gatenames():
            invalid_gate_names.add(g.name)
    result = {}
    if len(invalid_gate_names) > 0:
        result['invalid_gate_names'] = invalid_gate_names
    return result

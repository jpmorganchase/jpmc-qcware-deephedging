import attr
import itertools
from typing import Set, Tuple, Sequence, Mapping, Any, Optional
from pyrsistent import pvector, pmap, pset
from pyrsistent.typing import PMap, PSet, PVector
from icontract import require  # type: ignore
from .helpers import reverse_map
from .gates import Dialect
from .instructions import (Instruction, instruction_bit_bindings_map,
                           instruction_parameters_are_fully_bound,
                           instruction_is_valid_executable,
                           instruction_is_valid_replacement,
                           instruction_pattern_matches_target)


@attr.s(frozen=True)
class Circuit(object):
    dialect_name = attr.ib(type=str)
    instructions = attr.ib(type=PVector[Instruction], converter=pvector)
    qubits = attr.ib(type=PSet[Any], converter=pset)
    # 'metadata' contains extra information not necessarily
    # understood by translations, such as "classical bit" arguments,
    # etc. which are not normally "part" of a quantum circuit.
    metadata = attr.ib(type=PMap[str, Any],
                       default=pmap(),
                       converter=pmap)

    @classmethod
    def from_tuples(cls, dialect: Dialect,
                    instructions: Sequence[Tuple[str, Mapping,
                                                 Sequence[int]]]):
        """
        A simplified version of from_instructions, where instead of
        specifying the actual instruction objects, you specify
        just the gate names, dict of parameters, and sequence of bit bindings
        """
        real_instructions = [
            Instruction(gate_def=dialect.gate_named(x[0]),
                        parameter_bindings=x[1],
                        bit_bindings=x[2]) for x in instructions
        ]
        return Circuit.from_instructions(dialect.name, real_instructions)

    @classmethod
    def from_instructions(cls,
                          dialect_name: str,
                          instructions: Sequence[Instruction],
                          qubits: Optional[Set[Any]] = None):
        if qubits is None:
            new_qubits: PSet = pset(
                set().union(*[set(i.bit_bindings)
                              for i in instructions]))  # type: ignore
        else:
            new_qubits = pset(qubits)
        return cls(
            dialect_name=dialect_name,
            instructions=instructions,  # type: ignore
            qubits=new_qubits)  # type: ignore

    def __str__(self):
        return "\n".join([self.dialect_name] + [f"Qubits: {self.qubits}"] +
                         [str(i) for i in self.instructions])


def circuit_conforms_to_dialect(c: Circuit, d: Dialect) -> bool:
    gatedefs_in_circuit = {i.gate_def for i in c.instructions}
    return c.dialect_name == d.name and gatedefs_in_circuit.issubset(
        d.gate_defs)


def circuit_bit_bindings(circuit: Circuit) -> PMap[Tuple[int, int], Set[int]]:
    """
    Given a sequence of instructions, return the complete
    map of bit bindings.  So "H(1), CX(0,1)" would return
    the bit bindings { (1,0):1, (2,0):0, (2,1):1 }
    """
    result: dict = {}
    for i, instruction in enumerate(circuit.instructions):
        for k, v in instruction_bit_bindings_map(instruction).items():
            result[(i, k)] = v
    return pmap(result)


"""
A BitBindingSignature is a type declaration here, but it's
"the set of sets of bit IDs in the circuit instructions which are bound to the
same bit"

So {{(0,0), (1,0)}, {(0,1)}} means that for a two-instruction circuit,
both instruction 0, bit 0 and instruction 1 bit 1 are bound to the same input
bit, and instruction 0, bit 1 is bound to a different bit.

The BitBindingSignature is used to compare two circuits and see if they have
the same basic graph structure
"""
BitBindingSignature = PSet[PSet[Tuple[int, int]]]


def circuit_bit_binding_signature(c: Circuit) -> BitBindingSignature:
    """
    Create the BitBindingSignature of a circuit
    """
    forward_bindings = circuit_bit_bindings(c)
    reverse_bindings = reverse_map(forward_bindings)
    return pset(reverse_bindings.values())


def circuit_parameters_are_fully_bound(c: Circuit) -> bool:
    """
    Whether or not every instruction in the circuit has its parameters
    fully bound
    """
    return all(
        [instruction_parameters_are_fully_bound(i) for i in c.instructions])


def circuit_is_valid_executable(c: Circuit) -> bool:
    """
    Whether not each instruction in the circuit is executable
    """
    return all([instruction_is_valid_executable(i) for i in c.instructions])


def circuit_is_valid_replacement(c: Circuit) -> bool:
    """
    Whether or not each instruction in the circuit is a valid replacement
    """
    return all([instruction_is_valid_replacement(i) for i in c.instructions])


@require(lambda target: circuit_parameters_are_fully_bound(target))
def circuit_pattern_matches_target(pattern: Circuit, target: Circuit) -> bool:
    return ((len(pattern.instructions) == len(target.instructions)) and all([
        instruction_pattern_matches_target(pattern.instructions[i],
                                           target.instructions[i])
        for i in range(len(pattern.instructions))
    ]) and (circuit_bit_binding_signature(pattern)
            == circuit_bit_binding_signature(target)))


def circuit_parameter_map(c: Circuit) -> Mapping[Tuple[int, str], Any]:
    result = {}
    for index, i in enumerate(c.instructions):
        for k, v in i.parameter_bindings.items():
            result[(index, k)] = v
    return pmap(result)


def circuit_parameter_names(c: Circuit) -> PSet[Tuple[int, str]]:
    result = set([])
    for index, i in enumerate(c.instructions):
        for name in i.gate_def.parameter_names:
            result.add((index, name))
    return pset(result)


def circuit_bit_targets(c: Circuit) -> PVector[int]:
    """
    an ordered list of bits addressed by bit bindings, so for
    H(0), CX(0,1) you should get [0,0,1]

    Circuits with the same bit binding signature should have a 
    1:1 correspondence with their circuit_bit_targets
    """
    return pvector(
        itertools.chain.from_iterable([i.bit_bindings for i in c.instructions
                                       ]))  # type: ignore

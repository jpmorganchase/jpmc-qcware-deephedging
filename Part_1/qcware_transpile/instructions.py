import attr
from typing import Mapping, Any, Set, Tuple
from pyrsistent import pmap
from pyrsistent.typing import PVector, PMap, PSet
from .gates import GateDef, _qubit_ids
from .helpers import map_seq_to_seq
from inspect import signature
from fractions import Fraction
import numpy


@attr.s(frozen=True)
class Instruction(object):
    """
    An instruction: a gate definition paired with parameter bindings
    and bit bindings
    """
    gate_def = attr.ib(type=GateDef)
    bit_bindings = attr.ib(type=PVector[int], converter=_qubit_ids)
    parameter_bindings = attr.ib(type=PMap[str, Any],
                                 default=pmap(),
                                 converter=pmap)
    # 'metadata' contains extra information not necessarily
    # understood by translations, such as "classical bit" arguments,
    # etc. which are not normally "part" of a quantum circuit.
    metadata = attr.ib(type=PMap[str, Any],
                       default=pmap(),
                       converter=pmap)

    @parameter_bindings.validator
    def check_parameter_bindings(self, attribute, value):
        if not set(value.keys()).issubset(self.gate_def.parameter_names):
            raise ValueError(
                "parameter bindings must bind parameters in the gate def")

    @bit_bindings.validator
    def check_bit_bindings(self, attribute, value):
        if len(value) != len(self.gate_def.qubit_ids):
            raise ValueError(
                "number of bit bindings must be equal to #bits in the gate def"
            )

    def __str__(self):
        parameter_bindings_str = ",".join(
            [f"{k}={v}" for k, v in self.parameter_bindings.items()])
        bit_bindings_str = ",".join([f"{x}" for x in self.bit_bindings])
        metadata = "" if len(self.metadata) == 0 else ", (" + ",".join(["{k}={v}" for k,v in self.metadata.items()]) + ")"
        return f"{self.gate_def.name}({parameter_bindings_str}), ({bit_bindings_str}){metadata})"


def instruction_parameters_are_fully_bound(i: Instruction) -> bool:
    """
    Whether or not all parameters in the instruction are fully
    bound to values, not functions.  This is important in deciding if a 
    circuit is fully bound (a target for a match pattern)
    or only a pattern (only a subset of 
    parameters is bound), or a match target (some parameters are bound
    to callables)
    """
    return all([(k in i.parameter_bindings)
                for k in i.gate_def.parameter_names])


def _is_valid_executable_parameter_value(v: Any) -> bool:
    """
    whether the instruction parameter is valid for execution (is numeric)
    """
    return numpy.isscalar(v)


def _is_valid_replacement_parameter_value(v: Any) -> bool:
    """
    Determines if a given value is a valid replacement
    parameter, ie if it is a tuple, callable of one argument,
    or numeric parameter
    """
    return ((isinstance(v, tuple) and isinstance(v[0], int)
             and isinstance(v[1], str))
            or (callable(v) and len(signature(v).parameters) == 1)
            or (numpy.isscalar(v)))


def audit_instruction_for_executable(i: Instruction) -> PMap:
    """
    Here to give more information about why an instruction isn't executable
    """
    result = {}
    if not instruction_parameters_are_fully_bound(i):
        result['instruction_parameters_not_fully_bound'] = True
    invalid_bindings = {
        k: (v, type(v))
        for k, v in i.parameter_bindings.items()
        if not _is_valid_executable_parameter_value(v)
    }
    if len(invalid_bindings) > 0:
        result['invalid parameter bindings'] = invalid_bindings # type: ignore
    return pmap(result)


def instruction_is_valid_executable(i: Instruction) -> bool:
    """
    An instruction is a valid executable instruction if it is
    fully bound and if all parameter values are concrete
    numbers
    """
    return (instruction_parameters_are_fully_bound(i) and all([
        _is_valid_executable_parameter_value(v)
        for v in i.parameter_bindings.values()
    ]))


def instruction_is_valid_replacement(i: Instruction) -> bool:
    """
    An instruction is a valid replacement instruction if it is
    fully bound and if all parameter "values" are either a
    value, tuple, or callable of one argument
    """
    return (instruction_parameters_are_fully_bound(i) and all([
        _is_valid_replacement_parameter_value(v)
        for v in i.parameter_bindings.values()
    ]))


def instruction_parameter_bindings_match(pattern: Instruction,
                                         target: Instruction) -> bool:
    """
    the parameters of an instruction "pattern" matches a target if all
    bound parameters in the pattern have keys present in the target
    and the bound values are the same
    """
    return set(pattern.parameter_bindings.keys()).issubset(
        set(target.parameter_bindings.keys())) \
        and all([pattern.parameter_bindings[x]
               == target.parameter_bindings[x]
               for x in pattern.parameter_bindings.keys()])


def instruction_pattern_matches_target(pattern: Instruction,
                                       target: Instruction) -> bool:
    """
    An instruction pattern matches a target if it has the same gate name,
    and matching parameter bindings
    """
    return ((pattern.gate_def.name == target.gate_def.name)
            and instruction_parameter_bindings_match(pattern, target))


def instruction_bit_bindings_map(
        instruction: Instruction) -> PMap[int, PSet[int]]:
    """
    Returns a "binding map" of bit ids to bit assignments;
    in other words, an instruction binding the gate CX with
    bit ids 0 and 1 to circuit bits 7 and 8 would return
    the map {0:7, 1:8}
    """
    qubit_ids = instruction.gate_def.qubit_ids
    bit_assignments = instruction.bit_bindings
    return map_seq_to_seq(qubit_ids, bit_assignments)


def _remapped_parameter(parameter_map: Mapping[Tuple[int, str], Any],
                        parameter_value: Any):
    """
    Remaps a parameter in a match target.  There are three
    options for a match target value:

    * If the value is a callable, call it, passing the parameter
      map as the argument
    * If the value is a tuple, replace it with the appropriate
      value from the parameter map
    * Otherwise, just keep the original value
    """
    if callable(parameter_value):
        result = parameter_value(parameter_map)
    elif isinstance(parameter_value, tuple):
        # ignoring type here as we *should* have a
        # value of type (int, str)->number
        result = parameter_map[parameter_value]  # type: ignore
    else:
        result = parameter_value
    return result


def remapped_instruction(qubit_map: Mapping[int, int],
                         parameter_map: Mapping[Tuple[int, str], Any],
                         target: Instruction) -> Instruction:
    """
    This remaps an instruction given a new qubit mapping (from
    target qubits in one circuit to target qubits in another) and
    a new parameter map (from the parameters in the original circuit,
    ie the keys in the parameter map are tuples of (index, parameter_name)

    """
    new_parameters = {
        # ignoring type as the following seems to confuse it; v can
        # either be a key into the parameter map or a tuple
        k: _remapped_parameter(parameter_map, v)
        for k, v in target.parameter_bindings.items()
    }
    new_bit_bindings = [qubit_map[b] for b in target.bit_bindings]
    return Instruction(
        gate_def=target.gate_def,
        parameter_bindings=new_parameters,  # type: ignore
        bit_bindings=new_bit_bindings)

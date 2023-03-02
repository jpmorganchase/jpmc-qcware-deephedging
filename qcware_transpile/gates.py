from pyrsistent import pvector, pset, pmap
from pyrsistent.typing import PSet, PMap
from icontract import require  # type: ignore
from typing import Union, Sequence, Set
import attr


def _qubit_ids(qubits: Union[int, Sequence[int]]):
    return pvector(range(qubits)) if isinstance(qubits,
                                                int) else pvector(qubits)


@attr.s(frozen=True)
class GateDef(object):
    """
    A gate definition, consisting only of a name, a set of names
    for parameters, and an ordered collection of integer qubit IDs.

    GateDefs can also support variable numbers of qubit args if 
    varargs == true.  This is not currently exploited
    """
    name = attr.ib(type=str)
    parameter_names = attr.ib(type=Set[str], converter=pset)
    qubit_ids = attr.ib(type=Sequence[int], converter=_qubit_ids)
    has_varargs = attr.ib(type=bool, default=False)

    def __str__(self):
        return "".join([self.name, "("] +
                       [",".join([s
                                  for s in self.parameter_names])] + ["), ("] +
                       [",".join([str(i) for i in self.qubit_ids])] + [")"])


@attr.s(frozen=True)
class Dialect(object):
    """
    A "dialect" -- essentially just a named set of gate definitions
    """
    name = attr.ib(type=str)
    gate_defs = attr.ib(type=PSet[GateDef], converter=pset)
    gate_map = attr.ib(type=PMap[str, GateDef], init=False)

    def __attrs_post_init__(self):
        # build an index by gate name to speed things up a bit
        gate_map: PMap[str,
                       GateDef] = pmap({g.name: g
                                        for g in self.gate_defs})
        object.__setattr__(self, "gate_map", gate_map)

    def __str__(self):
        return "\n  ".join([self.name] + [str(g) for g in self.gate_defs])

    def has_gate_named(self, name: str) -> bool:
        return name in self.gate_map

    @require(lambda self, name: self.has_gate_named(name))
    def gate_named(self, name: str) -> GateDef:
        return self.gate_map[name]

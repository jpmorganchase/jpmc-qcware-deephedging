"""
Unfortunately, Qsharp, being in a separate language and library,
doesn't lend itself to introspection, so our usual approach
to finding the possible gates and excluding problematic ones
will have to be the simpler but more labor-intensive route
of hard-coding them.

QSharp python docs: https://docs.microsoft.com/en-us/python/qsharp-core/qsharp
"""
from qcware_transpile.gates import GateDef, Dialect
from qcware_transpile.circuits import Circuit
from typing import Tuple
from pyrsistent import pset
from pyrsistent.typing import PSet

__dialect_name__ = "qsharp"


def simple_gate(t: Tuple[str, set, int]):
    return GateDef(name=t[0], parameter_names=t[1], qubit_ids=t[2])


def gate_defs() -> PSet[GateDef]:
    # a list of basic gates.  Currently these come from the Microsoft.Quantum.Intrinsics
    # library
    simple_gates = (("CCNOT", {}, 3), ("CNOT", {}, 2), ("H", {}, 1),
                    ("I", {}, 1), ("Rx", {"theta"}, 1), ("Ry", {"theta"}, 1),
                    ("Rz", {"theta"}, 1), ("S", {}, 1), ("SWAP", {}, 2),
                    ("T", {}, 1), ("X", {}, 1), ("Y", {}, 1), ("Z", {}, 1))
    return pset({simple_gate(t) for t in simple_gates})


def dialect() -> Dialect:
    return Dialect(name=__dialect_name__, gate_defs=gate_defs())


def ir_to_native(c: Circuit) -> str:
    header = """
open Microsoft.Quantum.Intrinsic;

operation Circuit(): Result {
"""

    footer = """
  return 1;
}
"""
    return header + footer

bellpair = Circuit.from_tuples(dialect(), (("H", {}, [0]), ("CNOT", {}, [0, 1])))

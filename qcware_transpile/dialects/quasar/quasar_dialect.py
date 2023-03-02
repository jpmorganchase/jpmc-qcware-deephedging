from quasar.circuit import Gate  # type: ignore
from quasar.circuit import Circuit as QuasarCircuit  # type: ignore
from pyrsistent import pset
from pyrsistent.typing import PSet
from typing import Callable, Any, Tuple, Generator, Dict
# not using relative imports here for the moment to simplify emacs
# send-to-buffer issues; it should arguably be set back to relative
# imports at some point
from qcware_transpile.gates import GateDef, Dialect
from qcware_transpile.circuits import Circuit
from qcware_transpile.instructions import Instruction
from qcware_transpile.helpers import map_seq_to_seq_unique
from inspect import signature
from icontract import require  # type: ignore
import functools

__dialect_name__ = "quasar"


def is_builtin(thing) -> bool:
    """
    Whether or not something is a Python builtin; used to filter
    out things in the Gate namespace which represent a Gate or the
    function used to create a gate.
    """
    return type(thing).__name__ in dir(__builtins__) or type(
        thing).__name__ == "builtin_function_or_method"


def represents_gate(f) -> bool:
    """
    As expected: whether or not a thing represents a gate
    (either by being a Gate or by being a callable with
    fully-populated defaults which returns a Gate object)
    """
    if isinstance(f, Gate):
        return True
    if (not callable(f)) or is_builtin(f):
        return False
    try:
        sig = signature(f)
    # this is genuinely strange.  When this buffer is sent to a
    # python interpreter, quasar_gatenames_full, etc does just fine;
    # when it's imported, it blows up without the following because
    # it raises "ValueError for item <class 'type'>
    except ValueError:
        return False
    has_all_defaults = all([
        param.default is not param.empty for param in sig.parameters.values()
    ])
    if not has_all_defaults:
        return False
    f_result = f()
    if not isinstance(f_result, Gate):
        return False
    return True


def gatedef_from_gate(name: str, g: Gate) -> GateDef:
    """
    Create a gatedef from a gate object.  We use a given
    name instead of g.name because gate "names" in quasar
    don't always correspond to the Python names (for example,
    Gate.ST has name = 'S^+', but for our dialect we want
    the mapping to Python to be as simple as possible)
    """
    name = name
    parameter_names = {k for k in g.parameters.keys()}
    num_qubits = g.nqubit
    return GateDef(name=name,
                   parameter_names=parameter_names,
                   qubit_ids=num_qubits)


def gatedef_from_gatefun(name: str, g: Callable) -> GateDef:
    """
    Creates a gatedef from a function that returns a gate;
    see above for reasons we have an explicit name
    """
    return gatedef_from_gate(name, g())


@require(lambda thing: represents_gate(thing))
def gatedef_from_gatething(name: str, thing: Any) -> GateDef:
    if isinstance(thing, Gate):
        return gatedef_from_gate(name, thing)
    elif callable(thing):
        return gatedef_from_gatefun(name, thing)
    else:
        assert False


@require(lambda thing: represents_gate(thing))
def gate_name_property(thing: Any) -> str:
    """
    Return the .name property of a Gate (or Gate
    returned by a callable function)
    """
    if isinstance(thing, Gate):
        return thing.name
    elif callable(thing):
        g = thing()
        assert (isinstance(g, Gate))
        return g.name
    else:
        assert False
        return None


@functools.lru_cache(1)
def quasar_names_full() -> PSet[str]:
    return pset(dir(Gate))


@functools.lru_cache(1)
def quasar_gatenames_full() -> PSet[str]:
    """
    The names of verything in the Gate namespace
    """
    result = pset(
        {name
         for name in dir(Gate) if represents_gate(getattr(Gate, name))})
    return result


@functools.lru_cache(1)
def quasar_gatethings_full() -> PSet:
    """
    All the things in the Gate namespace which represent a gate
    """
    result = pset({
        getattr(Gate, name)
        for name in quasar_gatenames_full()
        if represents_gate(getattr(Gate, name))
    })
    return result


@functools.lru_cache(1)
def dialect() -> Dialect:
    """
    Programmatically create the Quasar dialect
    """
    gatedefs = {
        gatedef_from_gatething(name, getattr(Gate, name))
        for name in quasar_gatenames_full()
    }
    return Dialect(name="quasar", gate_defs=gatedefs)  # type: ignore


@functools.lru_cache(1)
def name_property_to_namespace_translation_table():
    """
    Okay... names in the Gate namespace for gates ("ST") don't always translate
    to the gate's .name property (in the ST case, "S^+").  So this is a translation
    table from the gate's name property to the name in the Gate namespace
    """
    namespace_names = list(quasar_gatenames_full())
    name_property = [
        gate_name_property(getattr(Gate, name)) for name in namespace_names
    ]
    return map_seq_to_seq_unique(name_property, namespace_names)


def native_instructions(
        qc: QuasarCircuit) -> Generator[Tuple[Gate, Tuple[int]], None, None]:
    """
    Iterates through the circuit yielding tuples of gate and 
    qubits.
    """
    for key, gate in qc.gates.items():
        yield gate, key[1]


def ir_instruction_from_native(gate: Gate, qubits: Tuple[int]) -> Instruction:
    """
    Direct conversion of one gate/qubits tuple into an IR "instruction"
    """
    ntt = name_property_to_namespace_translation_table()
    return Instruction(gate_def=gatedef_from_gatething(ntt[gate.name], gate),
                       bit_bindings=qubits,
                       parameter_bindings=gate.parameters)


def native_to_ir(qc: QuasarCircuit) -> Circuit:
    """
    Return a transpile-style Circuit object from a quasar Circuit object
    """
    instructions = list(
        ir_instruction_from_native(x[0], x[1])
        for x in native_instructions(qc))
    return Circuit.from_instructions(dialect_name=__dialect_name__,
                                     instructions=instructions)  # type: ignore


def quasar_gate_from_instruction(i: Instruction) -> Gate:
    """
    Create a quasar Gate object from an instruction
    """
    g = getattr(Gate, i.gate_def.name)
    # if g is a Gate, it shouldn't have parameters; if it's a callable,
    # it should.
    if len(i.parameter_bindings.keys()) > 0:
        g = g(**i.parameter_bindings)
    return g


def ir_to_native(c: Circuit, fill_unused_edge_qubits=False) -> QuasarCircuit:
    result = QuasarCircuit()
    for instruction in c.instructions:
        g = quasar_gate_from_instruction(instruction)
        result.add_gate(g, tuple(instruction.bit_bindings))
    # by default, quasar silently ignores unused edge qubits (qubits
    # "above" or "below" instructions that have no instructions of their
    # own).  This can cause disparity with toolkits that do not
    # share this behaviour.  It unfortunately extends the current idea
    # that qubit indices are integers
    if fill_unused_edge_qubits == True:
        # find the unused edge qubits.  We can use quasar's min_qubit and max_qubit
        # combined with the list of qubits in the Circuit object.
        # The Circuit's qubits is a set
        qubits_to_fill = { qb for qb in c.qubits if (qb < result.min_qubit) or (qb > result.max_qubit) }
        # on each unused edge qubits, add an identity gate
        for qb in qubits_to_fill:
            result.I(qb)
    return result 


def native_circuits_are_equivalent(c1: QuasarCircuit,
                                   c2: QuasarCircuit) -> bool:
    """
    Our own definition of whether two quasar circuits are equivalent;
    used to test
    """
    return QuasarCircuit.test_equivalence(c1, c2)


def audit(c: QuasarCircuit) -> Dict:
    """
    Right now, anything expressible in Quasar *should*
    be expressible in the IR, so this is a dummy result.
    """
    return {}

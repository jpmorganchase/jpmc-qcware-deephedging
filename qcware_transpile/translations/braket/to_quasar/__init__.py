from qcware_transpile.matching import (TranslationSet, trivial_rules, 
                                       untranslatable_instructions, untranslated_gates, 
                                       simple_translate)
from qcware_transpile.dialects import (quasar as quasar_dialect, braket as
                                       braket_dialect)
from qcware_transpile import TranslationException
from pyrsistent import pset
from icontract.errors import ViolationError
import braket.circuits
import quasar    
from toolz.functoolz import thread_first
from typing import Dict


def half_angle(theta):
    return theta / 2.0


def translation_set():
    """
    Creates a translation set from braket to quasar
    """
    trivial_gates = {('I', 'I'), ('H', 'H'), ('X', 'X'), 
                     ('Y', 'Y'), ('Z', 'Z'), ('S', 'S'),
                     ('Si', 'ST'), ('T', 'T'), ('Ti', 'TT'),
                     ('CNot', 'CX'), ('CY', 'CY'), ('CZ', 'CZ'),
                     ('CCNot', 'CCX'), ('PhaseShift', 'u1'),
                     ('Swap', 'SWAP'), ('CSwap', 'CSWAP'),
                     ('Rx', 'Rx', half_angle), 
                     ('Ry', 'Ry', half_angle), 
                     ('Rz', 'Rz', half_angle)}

    quasar_d = quasar_dialect.dialect()
    braket_d = braket_dialect.dialect()

    rules = pset().union(trivial_rules(braket_d, quasar_d, trivial_gates))
    return TranslationSet(from_dialect = braket_d, 
                          to_dialect = quasar_d, 
                          rules=rules)


target_gatenames = sorted(
    [x.name for x in translation_set().to_dialect.gate_defs])
untranslated = sorted([x.name for x in untranslated_gates(translation_set())])

def is_centered(c: braket.circuits.Circuit) -> bool:
    # determine if circuit has leading or following "empty" qubits
    if len(c.qubits):
        min_qubit = c.qubits[0]
        max_qubit = c.qubits[-1]
        return (min_qubit == 0 and max_qubit == len(c.qubits)-1)
    return False

def audit(c: braket.circuits.Circuit) -> Dict:
    ir_audit = braket_dialect.audit(c)
    if len(ir_audit.keys()) > 0:
        return ir_audit

    irc = braket_dialect.native_to_ir(c)
    untranslatable = untranslatable_instructions(irc, translation_set())

    result = {}
    if len(untranslatable) > 0:
        result['untranslatable_instructions'] = untranslatable
    # currently there is no way to express leading or following "empty" qubits in quasar
    if not is_centered(c):
        result['circuit_not_centered'] = True
    return result

def native_is_translatable(c: braket.circuits.Circuit):
    return len(audit(c)) == 0

def translate(c: braket.circuits.Circuit) -> quasar.Circuit:
    """
    Native-to-native translation
    """
    if not native_is_translatable(c):
        raise TranslationException(audit(c))
    try: 
        return thread_first(c, braket_dialect.native_to_ir, 
                            lambda x: simple_translate(translation_set(), x), 
                            quasar_dialect.ir_to_native)
    except ViolationError:
        raise TranslationException(audit(c))
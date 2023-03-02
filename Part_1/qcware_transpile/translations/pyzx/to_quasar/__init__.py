from qcware_transpile.matching import (TranslationRule, TranslationSet,
                                       trivial_rule, trivial_rules,
                                       untranslated_gates, simple_translate,
                                       circuit_is_simply_translatable_by,
                                       untranslatable_instructions)
from qcware_transpile.dialects import quasar as quasar_dialect, pyzx as pyzx_dialect
import pyzx
from qcware_transpile.circuits import Circuit
from qcware_transpile import TranslationException
from pyrsistent import pset
import quasar
from typing import Dict
from toolz.functoolz import thread_first
from fractions import Fraction
from numpy import pi
from icontract.errors import ViolationError


def from_phase_angle(phase: Fraction) -> float:
    """
    Pyzx seems to evaluate phase angles as a fraction of pi/2, so we must
    translate our quasar angles from that
    """
    return float(phase * pi)


def translation_set():
    """
    Creates a translation set from quasar to qiskit
    """
    trivial_gates = {
        ('HAD', 'H'),
        ('NOT', 'X'),
        ('CZ', 'CZ'),  # ('T', 'T', lambda pm: False),
        ('ZPhase', 'Rz', from_phase_angle),
        ('Z', 'Z'),
        ('CNOT', 'CX'),
        ('XPhase', 'Rx', from_phase_angle),
        ('SWAP', 'SWAP')
    }

    # the T and S gates can be adjoint in pyzx, although in quasar those
    # adjoints are the TT and ST gates respectively
    quasar_d = quasar_dialect.dialect()
    pyzx_d = pyzx_dialect.dialect()
    st_gates = {
        TranslationRule(pattern=Circuit.from_tuples(pyzx_d, [('T', {
            'adjoint': False
        }, [0])]),
                        replacement=Circuit.from_tuples(
                            quasar_d, [('T', {}, [0])])),
        TranslationRule(pattern=Circuit.from_tuples(pyzx_d, [('T', {
            'adjoint': True
        }, [0])]),
                        replacement=Circuit.from_tuples(
                            quasar_d, [('TT', {}, [0])])),
        TranslationRule(pattern=Circuit.from_tuples(pyzx_d, [('S', {
            'adjoint': False
        }, [0])]),
                        replacement=Circuit.from_tuples(
                            quasar_d, [('S', {}, [0])])),
        TranslationRule(pattern=Circuit.from_tuples(pyzx_d, [('S', {
            'adjoint': True
        }, [0])]),
                        replacement=Circuit.from_tuples(
                            quasar_d, [('ST', {}, [0])])),
    }
    rules = pset().union(trivial_rules(pyzx_d, quasar_d,
                                       trivial_gates)).union(st_gates)
    return TranslationSet(from_dialect=pyzx_d,
                          to_dialect=quasar_d,
                          rules=rules)


target_gatenames = sorted(
    [x.name for x in translation_set().to_dialect.gate_defs])
untranslated = sorted([x.name for x in untranslated_gates(translation_set())])


def audit(c: pyzx.Circuit) -> Dict:
    ir_audit = pyzx_dialect.audit(c)
    if len(ir_audit.keys()) > 0:
        return ir_audit

    irc = pyzx_dialect.native_to_ir(c)
    untranslatable = untranslatable_instructions(irc, translation_set())

    result = {}
    if len(untranslatable) > 0:
        result['untranslatable_instructions'] = untranslatable
    return result


def native_is_translatable(c: pyzx.Circuit):
    """
    A native quasar circuit is translatable to qiskit if it
    is "centered" (ie no leading qubits) and is composed of translatable
    gates
    """
    return len(audit(c)) == 0


def translate(c: pyzx.Circuit) -> quasar.Circuit:
    """
    Native-to-native translation
    """
    if not native_is_translatable(c):
        raise TranslationException(audit(c))
    try:
        return thread_first(c, pyzx_dialect.native_to_ir,
                            lambda x: simple_translate(translation_set(), x),
                            quasar_dialect.ir_to_native)
    except ViolationError:
        raise TranslationException(audit(c))

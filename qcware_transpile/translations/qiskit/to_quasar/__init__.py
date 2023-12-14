# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co and QC Ware
from qcware_transpile.matching import (TranslationRule, TranslationSet,
                                       trivial_rule, trivial_rules,
                                       translated_gates, untranslated_gates,
                                       simple_translate,
                                       circuit_is_simply_translatable_by,
                                       untranslatable_instructions)
from qcware_transpile.dialects import quasar as quasar_dialect, qiskit as qiskit_dialect
import qiskit
from qcware_transpile import TranslationException
from pyrsistent import pset
from icontract.errors import ViolationError
import quasar
from toolz.functoolz import thread_first


def half_angle(theta):
    return theta / 2.0


def translation_set():
    """
    Creates a translation set from quasar to qiskit
    """
    trivial_gates = {('id', 'I'), ('h', 'H'), ('x', 'X'), ('y', 'Y'),
                     ('z', 'Z'), ('s', 'S'), ('t', 'T'), ('cx', 'CX'),
                     ('cy', 'CY'), ('cz', 'CZ'), ('ccx', 'CCX'),
                     ('swap', 'SWAP'), ('cswap', 'CSWAP'),
                     ('rx', 'Rx', half_angle), ('ry', 'Ry', half_angle),
                     ('rz', 'Rz', half_angle), ('measure', 'I')}  #('u1', 'u1')

    quasar_d = quasar_dialect.dialect()
    qiskit_d = qiskit_dialect.dialect()
    # u1/u2/u3 rules were included but are disabled as IBM is deprecating
    # them at some level.  We currently remove measure rules, but cannot
    # translate reset or barrier rules.
    # we "remove" measure rules by converting them to I gates, which is a hack
    # to allow qiskit circuits with edge bits that are measured.
    rules = pset().union(trivial_rules(
        qiskit_d, quasar_d, trivial_gates))
    return TranslationSet.from_trivial_rules(from_dialect=qiskit_d,
                                        to_dialect=quasar_d,
                                        rules=rules)


target_gatenames = sorted(
    [x.name for x in translation_set().to_dialect.gate_defs])
untranslated = sorted([x.name for x in untranslated_gates(translation_set())])


def audit(c: qiskit.QuantumCircuit):
    """
    Tries to return a list of issues with the circuit which would
    make it untranslatable
    """
    ir_audit = qiskit_dialect.audit(c)
    if len(ir_audit) > 0:
        return ir_audit

    qdc = qiskit_dialect.native_to_ir(c)
    untranslatable = untranslatable_instructions(qdc, translation_set())

    # circuit_qubits = sorted(list(qdc.qubits))
    # used_qubits = sorted(
    #     list(set().union(*[set(i.bit_bindings) for i in qdc.instructions])))

    result = {}

    # if len(used_qubits) == 0:
    #     result["no_used_qubits"] = True
    # else:
    #     unused_edge_qubits = {
    #         x
    #         for x in circuit_qubits
    #         if (x < used_qubits[0]) or (x > used_qubits[-1])
    #     }
    #     if len(unused_edge_qubits) > 0:
    #         result['unused_edge_qubits'] = unused_edge_qubits

    if len(untranslatable) > 0:
        result['untranslatable_instructions'] = untranslatable
    return result


def basis_gates():
    """The "basis gates" for the quasar backend, according to qiskit
    """
    return list({x.name for x in translated_gates(translation_set())})


def native_is_translatable(c: qiskit.QuantumCircuit, should_transpile=True):
    """
    A native circuit is translatable to quasar if it has no leading or
    following "empty" qubits as currently there is no way to express this in quasar
    """
    c2 = qiskit.compiler.transpile(
        c, basis_gates=basis_gates()) if should_transpile else c.copy()
    return len(audit(c2)) == 0


def translate(c: qiskit.QuantumCircuit,
              should_transpile=True) -> quasar.Circuit:
    """
    Native-to-native translation.  If should_transpile is True,
    attempt to use qiskit to transpile the circuit to the set of 
    gates supported by the translation table.  There exists a chance
    this results in a strange audit with gates which are not in
    the original circuit, but by and large it works
    """
    if should_transpile:
        c2 = qiskit.compiler.transpile(c, basis_gates=basis_gates())
    else:
        c2 = c.copy()
    # if not native_is_translatable(c2):
    #     raise TranslationException(audit(c2))
    try:
        return thread_first(c2, qiskit_dialect.native_to_ir,
                            lambda x: simple_translate(translation_set(), x),
                            lambda x: quasar_dialect.ir_to_native(x, fill_unused_edge_qubits=True))
    except ViolationError:
        raise TranslationException(audit(c2))

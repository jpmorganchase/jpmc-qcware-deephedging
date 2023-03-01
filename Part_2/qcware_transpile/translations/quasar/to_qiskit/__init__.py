from qcware_transpile.matching import (TranslationRule, TranslationSet,
                                       trivial_rule, trivial_rules,
                                       untranslated_gates, simple_translate,
                                       circuit_is_simply_translatable_by,
                                       untranslatable_instructions)
from qcware_transpile.dialects import (quasar as quasar_dialect, qiskit as
                                       qiskit_dialect)
from qcware_transpile.circuits import Circuit
from qcware_transpile import TranslationException
from icontract.errors import ViolationError
import quasar
from pyrsistent import pset
import qiskit
from toolz.functoolz import thread_first


def double_angle(theta):
    return 2 * theta


def translation_set():
    """
    Creates a translation set from quasar to qiskit
    """
    trivial_gates = {('I', 'id'), ('H', 'h'), ('X', 'x'),
                     ('Y', 'y'), ('Z', 'z'), ('S', 's'),
                     ('T', 't'), ('CX', 'cx'), ('CY', 'cy'),
                     ('CZ', 'cz'), ('CCX', 'ccx'), ('u1', 'u1'),
                     ('SWAP', 'swap'), ('CSWAP', 'cswap'),
                     ('Rx', 'rx', double_angle),
                     ('Ry', 'ry', double_angle),
                     ('Rz', 'rz', double_angle)}

    quasar_d = quasar_dialect.dialect()
    qiskit_d = qiskit_dialect.dialect()
    other_rules = {
        TranslationRule(
            pattern=Circuit.from_tuples(quasar_d, [('RBS', {}, [0, 1])]),
            replacement=Circuit.from_tuples(
                qiskit_d,
                [('h', {}, [0]),
                 ('h', {}, [1]),
                 ('cz', {}, [0,1]),
                 # Note!  We don't double_angle there because the angle
                 # to give to ry is actually theta/2
                 ('ry', {'theta': lambda pm: pm[(0, 'theta')]}, [0]),
                 ('ry', {'theta': lambda pm: -pm[(0, 'theta')]}, [1]),
                 ('cz', {}, [0,1]),
                 ('h', {}, [0]),
                 ('h', {}, [1])]))
    }
    # the U2/U3 rules are disabled for now as they seem to be problematic

    rules = pset().union(trivial_rules(quasar_d, qiskit_d,
                                       trivial_gates)).union(other_rules)
    return TranslationSet(from_dialect=quasar_d,
                          to_dialect=qiskit_d,
                          rules=rules)


target_gatenames = sorted(
    [x.name for x in translation_set().to_dialect.gate_defs])
untranslated = sorted([x.name for x in untranslated_gates(translation_set())])


def audit(c: quasar.Circuit):
    ir_audit = quasar_dialect.audit(c)
    if len(ir_audit.keys()) > 0:
        return ir_audit

    irc = quasar_dialect.native_to_ir(c)
    untranslatable = untranslatable_instructions(irc, translation_set())

    result = {}
    if len(untranslatable) > 0:
        result['untranslatable_instructions'] = untranslatable
    if not quasar.Circuit.test_equivalence(c, c.center()):
        # type notation below since TypedDict was fresh in 3.8
        result['circuit_not_centered'] = True  # type: ignore
    return result


def native_is_translatable(c: quasar.Circuit):
    """
    A native quasar circuit is translatable to qiskit if it
    is "centered" (ie no leading qubits)
    """
    return len(audit(c)) == 0


def translate(c: quasar.Circuit) -> qiskit.QuantumCircuit:
    """
    Native-to-native translation
    """
    if not native_is_translatable(c):
        raise TranslationException(audit(c))
    try:
        return thread_first(c, quasar_dialect.native_to_ir,
                            lambda x: simple_translate(translation_set(), x),
                            qiskit_dialect.ir_to_native)
    except ViolationError:
        raise TranslationException(audit(c))

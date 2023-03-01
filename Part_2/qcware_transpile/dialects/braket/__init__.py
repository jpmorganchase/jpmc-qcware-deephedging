"""
AWS Braket is Amazon's quantum-on-the-cloud service.
"""
from .braket_dialect import (dialect, occupy_empty_qubits, 
                             native_to_ir, ir_to_native,
                             native_circuits_are_equivalent, audit)

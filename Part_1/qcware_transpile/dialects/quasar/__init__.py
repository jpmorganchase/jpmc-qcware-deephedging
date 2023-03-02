"""
Quasar is the toolkit used by QC Ware internally.

It supports fast and accurate simulation and evaluation with
QC Ware's Forge platform
"""
from .quasar_dialect import (dialect, native_to_ir, ir_to_native,
                             native_circuits_are_equivalent, audit)

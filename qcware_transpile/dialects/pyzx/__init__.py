# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co and QC Ware
"""
Pyzx is a toolkit for manipulating circuits using the ZX
calculus.
"""

from .pyzx_dialect import (dialect, native_to_ir, ir_to_native,
                             native_circuits_are_equivalent, audit)

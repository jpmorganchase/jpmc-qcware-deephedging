# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co and QC Ware
"""
Modules and functions for the transpilation of circuits
"""
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

# __version__ = importlib_metadata.version(__name__)

from .exceptions import TranslationException

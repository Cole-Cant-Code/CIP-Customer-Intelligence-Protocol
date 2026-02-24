"""Scaffold subsystem — cognitive reasoning frameworks from YAML."""

from cip_protocol.scaffold.engine import ScaffoldEngine, ScaffoldNotFoundError
from cip_protocol.scaffold.loader import load_scaffold_directory, load_scaffold_file
from cip_protocol.scaffold.models import AssembledPrompt, Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.scaffold.validator import (
    validate_scaffold_directory,
    validate_scaffold_file,
    validate_scaffolds,
)

__all__ = [
    "AssembledPrompt",
    "Scaffold",
    "ScaffoldEngine",
    "ScaffoldNotFoundError",
    "ScaffoldRegistry",
    "load_scaffold_directory",
    "load_scaffold_file",
    "validate_scaffold_directory",
    "validate_scaffold_file",
    "validate_scaffolds",
]

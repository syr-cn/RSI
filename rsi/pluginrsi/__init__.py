"""PluginRSI: Recursive Improvement of Agent Harnesses with Reusable Plugins (arXiv:2609.32423)."""

from .api import make_proposer, run
from .config import Config
from .library import PluginEntry, PluginLibrary
from .loop import PluginRSILoop
from .mock import MockPluginProposer
from .mutation import (
    balanced_batch,
    extract_plugin_patch,
    matches_globs,
    paired_gain,
    resolve_surfaces,
    validate_plugin_surface,
)
from .proposer import LLMPluginProposer, MutationProposal, PluginProposer, RecomposeProposal

__all__ = [
    "Config",
    "LLMPluginProposer",
    "MockPluginProposer",
    "MutationProposal",
    "PluginEntry",
    "PluginLibrary",
    "PluginProposer",
    "PluginRSILoop",
    "RecomposeProposal",
    "balanced_batch",
    "extract_plugin_patch",
    "make_proposer",
    "matches_globs",
    "paired_gain",
    "resolve_surfaces",
    "run",
    "validate_plugin_surface",
]

"""Stage 1 (plugin mutation) and Stage 2 (harness recomposition) proposer interfaces and LLM backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.artifact import Artifact, parse_file_blocks
from ..core.domain import Domain, Trial
from ..core.evaluate import EvalResult
from ..core.llm import LLM, Usage, extract_json
from .library import PluginLibrary
from .mutation import matches_globs

MUTATION_SYSTEM = (
    "You are the Stage 1 Plugin Mutation proposer in PluginRSI. "
    "Invent a reusable plugin-level improvement for a concrete failure in the supplied balanced minibatch traces, "
    "while checking successful traces to avoid regressions. "
    "Modify ONLY files matching the assigned plugin surface globs; keep workflow files and all sibling plugin "
    "surfaces byte-for-byte unchanged. Do not hardcode task IDs or answers."
)

RECOMPOSE_SYSTEM = (
    "You are the Stage 2 Harness Recomposition proposer in PluginRSI. "
    "Build a stronger complete harness by selecting compatible, positive-gain plugins from the Plugin Library "
    "and optionally refining workflow files so the selected plugins work together cleanly. "
    "Do not hardcode task IDs or answers."
)


@dataclass
class MutationProposal:
    surface: str
    artifact: Artifact
    hypothesis: str = ""
    changes: str = ""
    usage: Usage = field(default_factory=Usage)


@dataclass
class RecomposeProposal:
    artifact: Artifact
    selected_refs: list[str] = field(default_factory=list)
    hypothesis: str = ""
    changes: str = ""
    usage: Usage = field(default_factory=Usage)


class PluginProposer:
    """Base interface for Stage 1 single-plugin mutators and Stage 2 library recomposers."""

    def propose_mutation(
        self,
        *,
        domain: Domain,
        incumbent: Artifact,
        surface: str,
        allowed_globs: Sequence[str],
        workflow_globs: Sequence[str],
        batch_trials: Sequence[Trial],
        library: PluginLibrary,
        history: Sequence[dict],
        iteration: int,
        branch: int,
        seed: int,
    ) -> MutationProposal | None:
        raise NotImplementedError

    def propose_recomposition(
        self,
        *,
        domain: Domain,
        incumbent: Artifact,
        incumbent_eval: EvalResult,
        workflow_globs: Sequence[str],
        plugin_surfaces: dict[str, list[str]],
        library: PluginLibrary,
        history: Sequence[dict],
        iteration: int,
        seed: int,
    ) -> RecomposeProposal | None:
        raise NotImplementedError


def _format_trials(trials: Sequence[Trial], max_chars: int = 3000) -> str:
    blocks: list[str] = []
    budget = max_chars
    for tr in trials:
        snippet = (
            f"### task={tr.task_id} family={tr.family} score={tr.score:.3f}\n"
            f"feedback: {tr.feedback[:300]}\n"
            f"trace:\n{(tr.trace or '')[:800]}\n"
        )
        if len(snippet) > budget and blocks:
            break
        blocks.append(snippet[: max(budget, 200)])
        budget -= len(blocks[-1])
        if budget <= 0:
            break
    return "\n".join(blocks) if blocks else "(no traces)"


def _parse_files_from_reply(reply: str) -> tuple[dict[str, str | None], dict]:
    """Accept either a JSON object with ``files`` or ``=== FILE: path ===`` blocks."""
    meta = extract_json(reply)
    if isinstance(meta, dict) and isinstance(meta.get("files"), dict):
        files: dict[str, str | None] = {str(k): (None if v is None else str(v)) for k, v in meta["files"].items()}
        return files, meta
    blocks = parse_file_blocks(reply)
    if blocks:
        files = {k: (None if v.strip() == "<<DELETE>>" else v) for k, v in blocks.items()}
        return files, meta if isinstance(meta, dict) else {}
    return {}, meta if isinstance(meta, dict) else {}


class LLMPluginProposer(PluginProposer):
    """Prompts an :class:`rsi.core.LLM` to perform Stage 1 plugin mutation and Stage 2 recomposition."""

    def __init__(self, llm: LLM, *, max_tokens: int = 4096, trace_chars: int = 3000) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.trace_chars = trace_chars

    def propose_mutation(
        self,
        *,
        domain: Domain,
        incumbent: Artifact,
        surface: str,
        allowed_globs: Sequence[str],
        workflow_globs: Sequence[str],
        batch_trials: Sequence[Trial],
        library: PluginLibrary,
        history: Sequence[dict],
        iteration: int,
        branch: int,
        seed: int,
    ) -> MutationProposal | None:
        surface_files = {p: txt for p, txt in incumbent.items() if matches_globs(p, allowed_globs)}
        workflow_files = {p: txt for p, txt in incumbent.items() if matches_globs(p, workflow_globs)}
        prior = [
            {"ref": e.ref, "paired_gain": round(e.paired_gain, 4), "changes": e.changes}
            for e in library.entries(surface)[:5]
        ]
        prompt = (
            f"# Domain\n{domain.describe()}\n\n"
            f"# Stage 1 Assignment\n"
            f"- Target plugin surface: `{surface}`\n"
            f"- Editable file globs (ONLY touch paths matching these): `{list(allowed_globs)}`\n"
            f"- Frozen workflow globs (DO NOT touch): `{list(workflow_globs)}`\n\n"
            f"# Read-only Workflow Context\n{json.dumps(workflow_files, indent=2)}\n\n"
            f"# Current `{surface}` Plugin Files\n{json.dumps(surface_files, indent=2)}\n\n"
            f"# Prior Admitted `{surface}` Plugins\n{json.dumps(prior, indent=2)}\n\n"
            f"# Balanced Minibatch Traces (failed + solved)\n"
            f"{_format_trials(batch_trials, self.trace_chars)}\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{"hypothesis": "<failure cause and expected fix>", '
            '"changes": "<concise summary>", '
            '"files": {"<path matching editable globs>": "<full file text or null to delete>"}}'
        )
        resp = self.llm.complete(
            prompt,
            system=MUTATION_SYSTEM,
            max_tokens=self.max_tokens,
            seed=seed + iteration * 97 + branch * 13,
            role="propose",
        )
        if not resp.ok:
            return None
        updates, meta = _parse_files_from_reply(resp.text)
        filtered = {p: v for p, v in updates.items() if matches_globs(p, allowed_globs)}
        if not filtered:
            return None
        child = incumbent.with_files(filtered)
        return MutationProposal(
            surface=surface,
            artifact=child,
            hypothesis=str(meta.get("hypothesis", "")),
            changes=str(meta.get("changes", f"mutate {surface}")),
            usage=resp.usage,
        )

    def propose_recomposition(
        self,
        *,
        domain: Domain,
        incumbent: Artifact,
        incumbent_eval: EvalResult,
        workflow_globs: Sequence[str],
        plugin_surfaces: dict[str, list[str]],
        library: PluginLibrary,
        history: Sequence[dict],
        iteration: int,
        seed: int,
    ) -> RecomposeProposal | None:
        if len(library) == 0:
            return None
        sample_trials = incumbent_eval.worst(3) + incumbent_eval.best(2)
        catalog = [
            {
                "ref": e.ref,
                "surface": e.surface,
                "paired_gain": round(e.paired_gain, 4),
                "hypothesis": e.hypothesis,
                "changes": e.changes,
                "files": e.files,
            }
            for e in library.entries()[:12]
        ]
        prompt = (
            f"# Domain\n{domain.describe()}\n\n"
            f"# Incumbent Harness (search score = {incumbent_eval.score:.4f})\n"
            f"{json.dumps(incumbent.files, indent=2)}\n\n"
            f"# Shared Plugin Library (verified positive local gains)\n"
            f"{json.dumps(catalog, indent=2)}\n\n"
            f"# Recent Recomposition History\n{json.dumps(list(history[-5:]), indent=2)}\n\n"
            f"# Representative Incumbent Traces\n{_format_trials(sample_trials, self.trace_chars)}\n\n"
            f"Select compatible `selected_refs` from the Plugin Library and optionally update workflow files "
            f"matching `{list(workflow_globs)}` so the workflow integrates them cleanly.\n"
            "Return ONLY a JSON object of the form:\n"
            '{"selected_refs": ["<surface/vNNN>", ...], '
            '"hypothesis": "<why this combination improves full search score>", '
            '"changes": "<concise summary>", '
            '"workflow_files": {"<workflow path>": "<full file text>"}}'
        )
        resp = self.llm.complete(
            prompt,
            system=RECOMPOSE_SYSTEM,
            max_tokens=self.max_tokens,
            seed=seed + iteration * 193 + 7,
            role="propose",
        )
        if not resp.ok:
            return None
        meta = extract_json(resp.text)
        if not isinstance(meta, dict):
            return None
        raw_refs = meta.get("selected_refs") or []
        refs = [str(r) for r in raw_refs if isinstance(r, str) and r in library]
        composed = library.apply(incumbent, refs) if refs else incumbent
        wf_updates = meta.get("workflow_files") or meta.get("files") or {}
        if isinstance(wf_updates, dict) and wf_updates:
            allowed_wf = {
                str(p): (None if v is None else str(v))
                for p, v in wf_updates.items()
                if not workflow_globs or matches_globs(str(p), workflow_globs)
            }
            if allowed_wf:
                composed = composed.with_files(allowed_wf)
        if composed == incumbent:
            return None
        return RecomposeProposal(
            artifact=composed,
            selected_refs=refs,
            hypothesis=str(meta.get("hypothesis", "")),
            changes=str(meta.get("changes", f"recompose {refs}")),
            usage=resp.usage,
        )

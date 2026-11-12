"""Balanced minibatch sampling, surface isolation, and paired plugin admission checks."""

from __future__ import annotations

import fnmatch
import random
from collections.abc import Iterable, Mapping, Sequence

from ..core.artifact import Artifact
from ..core.domain import Domain
from .config import Config

FALLBACK_WORKFLOW_GLOBS = ("harness.py", "harness.md", "control/*", "config/*")
FALLBACK_PLUGIN_SURFACES: dict[str, list[str]] = {
    "prompt": ["prompts/*"],
    "tool": ["tools/*"],
    "skill": ["skills/*"],
    "memory": ["memory/*"],
    "subagent": ["agents/*"],
}


def matches_globs(path: str, globs: Iterable[str]) -> bool:
    """True when ``path`` matches any glob (with ``dir/*`` matching nested paths under ``dir/`` as well)."""
    for pat in globs:
        if fnmatch.fnmatchcase(path, pat):
            return True
        if pat.endswith("/*") and path.startswith(pat[:-1]):
            return True
    return False


def resolve_surfaces(domain: Domain, artifact: Artifact, cfg: Config) -> tuple[dict[str, list[str]], list[str]]:
    """Partition the domain's editable file tree into plugin surfaces ``P`` and workflow ``W``."""
    comps = getattr(domain, "components", None) or {}
    wf_names = set(cfg.workflow_components)
    if comps:
        workflow_globs = [g for name, gs in comps.items() if name in wf_names for g in gs]
        allowed_names = (
            list(cfg.plugin_components)
            if cfg.plugin_components is not None
            else [name for name in comps if name not in wf_names]
        )
        plugin_surfaces = {name: list(comps[name]) for name in allowed_names if name in comps}
        if plugin_surfaces:
            return plugin_surfaces, workflow_globs

    matched = {
        name: globs
        for name, globs in FALLBACK_PLUGIN_SURFACES.items()
        if any(matches_globs(p, globs) for p in artifact)
    }
    if matched:
        wf = [g for g in FALLBACK_WORKFLOW_GLOBS if any(matches_globs(p, (g,)) for p in artifact)]
        return matched, wf
    return {p: [p] for p in sorted(artifact)}, []


def balanced_batch(
    task_ids: Sequence[str],
    scores: Mapping[str, float],
    batch_size: int,
    failure_quota: int,
    rng: random.Random,
    *,
    pass_score: float = 1.0,
) -> list[str]:
    """Sample a balanced minibatch of failed (``score < pass_score``) and solved tasks (Algorithm 1, Stage 1)."""
    ordered = list(task_ids)
    if not ordered or batch_size <= 0:
        return []
    size = min(batch_size, len(ordered))
    failed = [tid for tid in ordered if scores.get(tid, 0.0) < pass_score]
    failed_set = set(failed)
    solved = [tid for tid in ordered if tid not in failed_set]
    rng.shuffle(failed)
    rng.shuffle(solved)

    n_fail = min(failure_quota, size, len(failed))
    n_pass = min(size - n_fail, len(solved))
    chosen = failed[:n_fail] + solved[:n_pass]
    if len(chosen) < size:
        chosen_set = set(chosen)
        rest = [tid for tid in failed[n_fail:] + solved[n_pass:] if tid not in chosen_set]
        chosen.extend(rest[: size - len(chosen)])
    return chosen


def paired_gain(
    parent_scores: Mapping[str, float],
    child_scores: Mapping[str, float],
    task_ids: Sequence[str],
) -> tuple[float, int]:
    """Mean paired score delta ``child - parent`` across shared tasks in ``task_ids``."""
    shared = [tid for tid in task_ids if tid in parent_scores and tid in child_scores]
    if not shared:
        return 0.0, 0
    delta = sum(float(child_scores[tid]) - float(parent_scores[tid]) for tid in shared) / len(shared)
    return delta, len(shared)


def validate_plugin_surface(
    parent: Artifact,
    child: Artifact,
    allowed_globs: Sequence[str],
    workflow_globs: Sequence[str],
) -> tuple[bool, str]:
    """Enforce Stage 1 isolation: change at least one file on ``allowed_globs``, touching neither workflow nor siblings."""
    changed = parent.changed_files(child)
    if not changed:
        return False, "proposal made no changes to the assigned plugin surface"
    outside = [p for p in changed if not matches_globs(p, allowed_globs)]
    if outside:
        return False, f"Stage 1 mutation touched paths outside assigned surface {list(allowed_globs)}: {outside}"
    if workflow_globs:
        frozen_touched = [p for p in changed if matches_globs(p, workflow_globs) and p not in set(allowed_globs)]
        if frozen_touched:
            return False, f"Stage 1 mutation modified frozen workflow files: {frozen_touched}"
    return True, "OK"


def extract_plugin_patch(parent: Artifact, child: Artifact, allowed_globs: Sequence[str]) -> dict[str, str | None]:
    """Extract the file-level patch (``None`` = deleted) on ``allowed_globs`` for storage in the plugin library."""
    patch: dict[str, str | None] = {}
    for path in parent.changed_files(child):
        if matches_globs(path, allowed_globs):
            patch[path] = child.files.get(path)
    return patch

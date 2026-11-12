"""Deterministic offline PluginProposer for HarnessWorld and AgentQA-layout domains."""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..core.artifact import Artifact
from ..core.domain import Domain, Trial
from ..core.evaluate import EvalResult
from ..core.llm import Usage
from .library import PluginLibrary
from .proposer import MutationProposal, PluginProposer, RecomposeProposal

AGENTQA_PLUGIN_PATCHES: dict[str, list[tuple[str, dict[str, str]]]] = {
    "prompt": [
        (
            "require explicit ANSWER: <value> final line and step-by-step trace",
            {
                "prompts/task.md": (
                    "{question}\n\nThink step by step before answering.\n"
                    "End your reply with a final line of the form 'ANSWER: <value>'.\n"
                )
            },
        ),
        (
            "add arithmetic double-check system directive",
            {
                "prompts/system.md": (
                    "You are a careful problem-solving assistant.\n"
                    "Double-check your arithmetic before giving the final answer.\n"
                )
            },
        ),
    ],
    "skill": [
        (
            "add structured decomposition procedure card",
            {
                "skills/decompose.md": (
                    "# Decomposition Skill\n"
                    "1. Extract every stated quantity and constraint.\n"
                    "2. Compute intermediate values explicitly.\n"
                    "3. Emit `ANSWER: <value>` on the final line.\n"
                )
            },
        )
    ],
    "subagent": [
        (
            "add bounded answer-verification checker guide",
            {
                "agents/checker.md": (
                    "Check the draft calculation against the question constraints and fix any formatting mismatch."
                )
            },
        )
    ],
    "memory": [
        (
            "add scratchpad format conventions",
            {"memory/conventions.md": "Preserve exact units, signs, and integer formatting requested by the prompt."},
        )
    ],
    "tool": [
        (
            "add answer-line normaliser helper",
            {"tools/normalize.md": "Strip trailing punctuation when formatting the final ANSWER line."},
        )
    ],
}

AGENTQA_RECOMPOSE_HARNESS = '''"""Evolved harness recomposed from PluginRSI library."""


def solve(question, llm, tools, files):
    system = files.get("prompts/system.md", "")
    skill = files.get("skills/decompose.md", "")
    if skill:
        system = (system.rstrip() + "\\n\\n" + skill).strip()
    prompt = files["prompts/task.md"].replace("{question}", question)
    finals = []
    for _ in range(3):
        r = llm(prompt, system=system)
        ls = [line for line in r.strip().splitlines() if line.strip()]
        finals.append(ls[-1] if ls else "")
    reply = max(finals, key=finals.count)
    if "agents/checker.md" in files:
        guide = files["agents/checker.md"]
        reply = llm(
            guide + "\\nQuestion: " + question + "\\nDraft answer: " + reply[-300:]
            + "\\nVerify the draft and give a final line ANSWER: <value>.",
            system=system,
        )
    lines = [line for line in reply.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""
'''


class MockPluginProposer(PluginProposer):
    """Offline proposer exercising isolated Stage 1 plugin mutations and Stage 2 library recomposition."""

    def __init__(self, *, seed: int = 0, include_leaks: bool = False) -> None:
        self.base_seed = seed
        self.include_leaks = include_leaks
        self._tried_ids: set[tuple[str, str]] = set()

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
        rng = random.Random(f"mock-mut-{self.base_seed}-{seed}-{iteration}-{branch}-{surface}")
        world = getattr(domain, "world", None)
        if world is not None and hasattr(world, "catalog"):
            return self._propose_harnessworld_mutation(incumbent, world, surface, rng)
        if "harness.py" in incumbent and "prompts/task.md" in incumbent:
            return self._propose_agentqa_mutation(incumbent, surface, iteration, branch)
        return None

    def _propose_harnessworld_mutation(
        self, incumbent: Artifact, world, surface: str, rng: random.Random
    ) -> MutationProposal | None:
        active_paths = set(incumbent.files)
        candidates = [
            m
            for m in world.catalog.values()
            if m.component == surface
            and m.path not in active_paths
            and (self.include_leaks or m.kind not in ("leak", "obfuscated_leak"))
        ]
        if not candidates:
            return None
        fresh = [m for m in candidates if (incumbent.id, m.id) not in self._tried_ids]
        pool = fresh or candidates
        mech = rng.choice(pool)
        self._tried_ids.add((incumbent.id, mech.id))
        child = incumbent.with_files({mech.path: mech.file_text()})
        return MutationProposal(
            surface=surface,
            artifact=child,
            hypothesis=f"Test {surface} plugin {mech.id} ({mech.title}) on balanced minibatch",
            changes=f"add {mech.path} ({mech.id})",
            usage=Usage(),
        )

    def _propose_agentqa_mutation(
        self, incumbent: Artifact, surface: str, iteration: int, branch: int
    ) -> MutationProposal | None:
        options = AGENTQA_PLUGIN_PATCHES.get(surface, [])
        for desc, patch in options:
            child = incumbent.with_files(patch)
            if child != incumbent:
                return MutationProposal(
                    surface=surface,
                    artifact=child,
                    hypothesis=f"Mutate {surface} surface: {desc}",
                    changes=desc,
                    usage=Usage(),
                )
        idx = (iteration + branch) % max(len(AGENTQA_PLUGIN_PATCHES["prompt"]), 1)
        desc, patch = AGENTQA_PLUGIN_PATCHES["prompt"][idx]
        if surface == "prompt":
            child = incumbent.with_files(patch)
            if child != incumbent:
                return MutationProposal(surface=surface, artifact=child, hypothesis=desc, changes=desc, usage=Usage())
        return None

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
        tried_combos = {tuple(h.get("selected_refs", ())) for h in history}
        ranked = [e for e in library.entries() if e.paired_gain > 0]
        if not ranked:
            return None

        # Prefer combining top verified plugin from each distinct surface; fall back to unmerged subsets
        best_by_surface = list(library.best_per_surface().values())
        refs = [e.ref for e in sorted(best_by_surface, key=lambda e: -e.paired_gain) if e.paired_gain > 0]
        if tuple(refs) in tried_combos:
            refs = [e.ref for e in ranked[: min(len(ranked), iteration + 2)]]

        composed = library.apply(incumbent, refs)
        world = getattr(domain, "world", None)
        wf_note = ""
        if world is not None and hasattr(world, "catalog"):
            cf = [
                m
                for m in world.catalog.values()
                if m.component == "control_flow" and m.path not in composed and m.kind in ("generic", "structural")
            ]
            if cf:
                chosen_wf = cf[iteration % len(cf)]
                composed = composed.with_files({chosen_wf.path: chosen_wf.file_text()})
                wf_note = f" + workflow {chosen_wf.id}"
        elif "harness.py" in composed and "prompts/task.md" in composed:
            composed = composed.with_files({"harness.py": AGENTQA_RECOMPOSE_HARNESS})
            wf_note = " + sc3/skill workflow wiring"

        if composed == incumbent:
            for entry in ranked:
                single = library.apply(incumbent, [entry.ref])
                if single != incumbent:
                    composed = single
                    refs = [entry.ref]
                    wf_note = ""
                    break
        if composed == incumbent:
            return None
        return RecomposeProposal(
            artifact=composed,
            selected_refs=refs,
            hypothesis=f"Recompose {len(refs)} verified library plugins{wf_note}",
            changes=f"integrate {refs}{wf_note}",
            usage=Usage(),
        )

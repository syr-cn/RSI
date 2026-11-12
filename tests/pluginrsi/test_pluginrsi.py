"""Offline deterministic unit and integration tests for PluginRSI (arXiv:2609.32423)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import rsi
from rsi.cli import main as cli_main
from rsi.core import LLM, Artifact, LLMResponse, SealedSplitError, Usage
from rsi.domains.agentqa import AgentQADomain, SimModel, make_suite
from rsi.domains.harnessworld import STRONG, WEAK, make_domain
from rsi.pluginrsi import (
    Config,
    LLMPluginProposer,
    MutationProposal,
    PluginLibrary,
    PluginProposer,
    balanced_batch,
    paired_gain,
    resolve_surfaces,
    validate_plugin_surface,
)


def test_api_registration_and_recommend(tmp_path: Path, capsys):
    assert "pluginrsi" in rsi.METHODS
    assert rsi.method_module("plugin-rsi").__name__ == "rsi.pluginrsi"
    assert rsi.method_module("plugin_rsi").__name__ == "rsi.pluginrsi"
    cfg = rsi.make_config("pluginrsi", {"iterations": 2, "batch_size": 6})
    assert isinstance(cfg, Config) and cfg.iterations == 2 and cfg.batch_size == 6
    recs = rsi.recommend(what_changes="harness", modular_plugins=True)
    assert recs[0][0] == "pluginrsi"
    assert cli_main(["recommend", "--modular-plugins"]) == 0
    assert "pluginrsi" in capsys.readouterr().out
    cli_out = tmp_path / "cli_prsi"
    assert (
        cli_main(
            [
                "improve",
                "--method",
                "pluginrsi",
                "--problem",
                "agentqa",
                "--llm",
                "sim",
                "--out",
                str(cli_out),
                "--set",
                "iterations=1",
                "branches_per_iter=2",
            ]
        )
        == 0
    )
    assert (cli_out / "summary.json").is_file() and (cli_out / "transfer.json").is_file()


def test_config_validation():
    with pytest.raises(ValueError, match="iterations"):
        Config(iterations=-1)
    with pytest.raises(ValueError, match="batch_size"):
        Config(batch_size=0)
    with pytest.raises(ValueError, match="trials"):
        Config(trials=0)


def test_balanced_batch_respects_failure_quota_and_backfills():
    ids = [f"t{i}" for i in range(10)]
    scores = {f"t{i}": (0.2 if i < 3 else 1.0) for i in range(10)}
    rng = random.Random(42)
    batch = balanced_batch(ids, scores, batch_size=6, failure_quota=2, rng=rng, pass_score=1.0)
    assert len(batch) == 6
    assert len(set(batch)) == 6
    assert sum(1 for t in batch if scores[t] < 1.0) == 2

    # Only 3 failures exist; asking for failure_quota=5 takes all 3 failures and backfills 3 solved tasks
    batch_backfill = balanced_batch(ids, scores, batch_size=6, failure_quota=5, rng=random.Random(7), pass_score=1.0)
    assert len(batch_backfill) == 6
    assert sum(1 for t in batch_backfill if scores[t] < 1.0) == 3


def test_validate_plugin_surface_isolates_stage1_edits():
    parent = Artifact(
        {
            "harness.md": "# workflow\n",
            "control/loop.py": "x = 1\n",
            "skills/s1/SKILL.md": "old skill\n",
            "tools/t1.py": "y = 2\n",
        }
    )
    ok_child = parent.with_files({"skills/s1/SKILL.md": "improved skill\n"})
    ok, _ = validate_plugin_surface(parent, ok_child, ["skills/*"], ["control/*", "harness.md"])
    assert ok

    ok_unchanged, why_unchanged = validate_plugin_surface(parent, parent, ["skills/*"], ["control/*", "harness.md"])
    assert not ok_unchanged and "no changes" in why_unchanged

    bad_sibling = parent.with_files({"skills/s1/SKILL.md": "improved\n", "tools/t1.py": "y = 3\n"})
    ok_sib, why_sib = validate_plugin_surface(parent, bad_sibling, ["skills/*"], ["control/*", "harness.md"])
    assert not ok_sib and "outside assigned surface" in why_sib

    bad_wf = parent.with_files({"harness.md": "# edited workflow\n"})
    ok_wf, _ = validate_plugin_surface(parent, bad_wf, ["skills/*"], ["control/*", "harness.md"])
    assert not ok_wf


def test_paired_gain_and_plugin_library_persistence(tmp_path: Path):
    parent_scores = {"a": 0.0, "b": 0.5, "c": 1.0}
    child_scores = {"a": 1.0, "b": 0.5, "c": 1.0}
    gain, n = paired_gain(parent_scores, child_scores, ["a", "b", "c", "missing"])
    assert n == 3 and pytest.approx(gain) == 1.0 / 3.0

    lib_dir = tmp_path / "lib"
    lib = PluginLibrary(lib_dir)
    e1 = lib.admit(
        surface="skill",
        files={"skills/verify/SKILL.md": "check units\n"},
        paired_gain=0.25,
        parent_score=0.5,
        child_score=0.75,
        batch_ids=["a", "b"],
        hypothesis="unit check",
        changes="add verify skill",
        iteration=1,
    )
    assert e1.ref == "skill/v001"
    assert (lib_dir / "library.json").is_file()

    reloaded = PluginLibrary(lib_dir)
    assert len(reloaded) == 1 and "skill/v001" in reloaded
    base = Artifact({"harness.md": "wf\n"})
    applied = reloaded.apply(base, ["skill/v001"])
    assert applied["skills/verify/SKILL.md"] == "check units\n"


def test_stage1_rejects_cross_surface_proposals_and_keeps_sealed_splits_closed(tmp_path: Path):
    dom = make_domain(seed=0, n_evolve=16, n_holdout=16, n_ood_per_family=4)

    class RogueProposer(PluginProposer):
        def propose_mutation(self, *, incumbent, **kw):
            # Illegally edits workflow file harness.md during Stage 1
            return MutationProposal(
                surface=kw["surface"],
                artifact=incumbent.with_files({"harness.md": incumbent["harness.md"] + "\n# rogue\n"}),
                changes="illegal workflow edit",
            )

        def propose_recomposition(self, **kw):
            return None

    res = rsi.improve(
        dom,
        method="pluginrsi",
        proposer=RogueProposer(),
        config={"iterations": 1, "branches_per_iter": 2, "batch_size": 4},
        out_dir=tmp_path / "rogue",
    )
    assert res.best == res.baseline
    assert res.meta["library_size"] == 0
    statuses = [n.status for n in res.ledger.nodes(kind="plugin_mutation")]
    assert statuses == ["rejected_surface", "rejected_surface"]

    with pytest.raises(SealedSplitError):
        rsi.improve(dom, method="pluginrsi", config={"search_split": "holdout", "iterations": 1})


def test_harnessworld_end_to_end_and_cross_policy_library_transfer(tmp_path: Path):
    dom_strong = make_domain(seed=2, policy=STRONG, n_evolve=30, n_holdout=30, n_ood_per_family=8)
    surfaces, wf_globs = resolve_surfaces(dom_strong, dom_strong.seed_artifact(), Config())
    assert {"client_tool", "skill", "memory", "subagent"} <= set(surfaces)
    assert "harness.md" in wf_globs

    out_strong = tmp_path / "strong_run"
    res_strong = rsi.improve(
        dom_strong,
        method="pluginrsi",
        config={"iterations": 3, "branches_per_iter": 6, "batch_size": 10, "failure_quota": 5, "seed": 2},
        out_dir=out_strong,
    )
    assert res_strong.meta["library_size"] >= 1
    assert res_strong.best != res_strong.baseline
    assert (
        dom_strong.expected(res_strong.best, "holdout")["S"] > dom_strong.expected(res_strong.baseline, "holdout")["S"]
    )

    # Transfer the discovered PluginLibrary to a fresh WEAK policy on the same HarnessWorld
    dom_weak = dom_strong.with_policy(WEAK)
    out_weak = tmp_path / "weak_transfer"
    res_weak = rsi.improve(
        dom_weak,
        method="pluginrsi",
        seed_library=res_strong.meta["plugin_library_dir"],
        config={"iterations": 2, "branches_per_iter": 4, "batch_size": 10, "seed": 5},
        out_dir=out_weak,
    )
    assert res_weak.meta["library_size"] >= res_strong.meta["library_size"]
    assert dom_weak.expected(res_weak.best, "evolve")["S"] >= dom_weak.expected(res_weak.baseline, "evolve")["S"]


def test_agentqa_offline_and_llm_proposer_protocol(tmp_path: Path):
    suite = make_suite(seed=0)
    dom = AgentQADomain(suite)
    sim = SimModel(suite)
    res = rsi.improve(
        dom,
        method="pluginrsi",
        llm_task=sim,
        config={"iterations": 2, "branches_per_iter": 3, "batch_size": 8, "failure_quota": 4},
        out_dir=tmp_path / "agentqa_prsi",
    )
    assert res.meta["library_size"] >= 1
    assert res.best != res.baseline

    class CannedJSONLLM(LLM):
        name = "canned-json"

        def _complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024, seed: int | None = None):
            if "Stage 1 Assignment" in prompt:
                payload = {
                    "hypothesis": "explicit ANSWER line reduces extraction errors",
                    "changes": "format task prompt",
                    "files": {
                        "prompts/task.md": (
                            "{question}\nThink step by step before answering.\n"
                            "End your reply with a final line of the form 'ANSWER: <value>'.\n"
                        )
                    },
                }
            else:
                payload = {
                    "selected_refs": ["prompt/v001"],
                    "hypothesis": "adopt verified prompt plugin",
                    "changes": "wire prompt/v001",
                    "workflow_files": {},
                }
            return LLMResponse(
                text=json.dumps(payload), model=self.name, usage=Usage(input_tokens=50, output_tokens=30)
            )

    llm_prop = CannedJSONLLM()
    res_llm = rsi.improve(
        dom,
        method="pluginrsi",
        llm_task=SimModel(suite),
        proposer=LLMPluginProposer(llm_prop),
        config={"iterations": 1, "branches_per_iter": 1, "plugin_components": ("prompt",), "batch_size": 8},
        out_dir=tmp_path / "agentqa_llm_prop",
    )
    assert res_llm.meta["library_size"] == 1
    assert res_llm.best != res_llm.baseline

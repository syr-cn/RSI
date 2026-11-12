"""PluginRSI two-stage evolutionary search loop (Shi et al., arXiv:2609.32423, Algorithm 1)."""

from __future__ import annotations

import random
from pathlib import Path

from ..core.artifact import Artifact
from ..core.critic import LeakageCritic
from ..core.domain import Domain
from ..core.evaluate import EvalResult, Evaluator
from ..core.ledger import Ledger, Node
from ..core.llm import LLM, Usage
from ..metaharness.validate import InterfaceValidator
from ..trace import RunTracer
from .config import Config
from .library import PluginLibrary
from .mutation import (
    balanced_batch,
    extract_plugin_patch,
    paired_gain,
    resolve_surfaces,
    validate_plugin_surface,
)
from .proposer import PluginProposer


class PluginRSILoop:
    """Alternates Stage 1 isolated plugin mutation on balanced minibatches with Stage 2 harness recomposition."""

    def __init__(
        self,
        domain: Domain,
        seed_artifact: Artifact,
        *,
        llm_task: LLM | None,
        proposer: PluginProposer,
        config: Config,
        out_dir: str | Path,
        seed_library: PluginLibrary | str | Path | None = None,
        tracer: RunTracer | None = None,
    ) -> None:
        self.domain = domain
        self.seed_artifact = seed_artifact
        self.llm_task = llm_task
        self.proposer = proposer
        self.cfg = config
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.out / "ledger.jsonl")
        self.library = PluginLibrary(self.out / "plugin_library")
        if seed_library is not None:
            self.library.import_from(seed_library)

        self.evaluator = Evaluator(domain, llm_task, workers=config.workers)
        self.validator = InterfaceValidator(config.validate_timeout_s, isolate=config.validate_in_subprocess)
        self.critic = LeakageCritic.for_domain(domain, split=config.search_split) if config.leakage_screen else None
        self.tr = tracer or RunTracer(None, "pluginrsi")
        self.rng = random.Random(f"pluginrsi-{config.seed}")

        self.plugin_surfaces, self.workflow_globs = resolve_surfaces(domain, seed_artifact, config)
        self.incumbent: Artifact = seed_artifact
        self.incumbent_ev: EvalResult | None = None
        self.incumbent_node_id: str = "prsi_seed"
        self.n_full_evals: int = 0
        self.n_minibatch_evals: int = 0
        self.proposer_usage = Usage()
        self.trajectory: list[dict] = []
        self.recompose_history: list[dict] = []
        self.stop_reason: str = "iterations"

    def _budget_exhausted(self) -> bool:
        return self.cfg.eval_budget is not None and self.n_full_evals >= self.cfg.eval_budget

    def _passes_guards(self, parent: Artifact, candidate: Artifact) -> tuple[bool, str]:
        if self.critic is not None:
            verdict = self.critic.check(parent, candidate)
            if not verdict.ok:
                return False, f"leakage critic rejected candidate: {verdict.reasons}"
        if self.cfg.validate:
            ok, msg = self.validator.validate(self.domain, candidate, self.llm_task)
            if not ok:
                return False, f"interface validation failed: {msg}"
        return True, "OK"

    def evaluate_seed(self) -> EvalResult:
        ev = self.evaluator.evaluate(self.incumbent, self.cfg.search_split, k=self.cfg.trials)
        self.n_full_evals += 1
        self.incumbent_ev = ev
        self.ledger.add(
            Node(
                id=self.incumbent_node_id,
                parent=None,
                round=0,
                kind="baseline",
                status="kept",
                score=ev.score,
                cost=ev.cost,
                change="seed harness",
                artifact_id=self.incumbent.id,
                metrics=ev.family_scores(),
            )
        )
        self.tr.evaluation(0, "seed", ev)
        self.tr.kept(0, "seed", self.incumbent, decision_score=ev.score)
        return ev

    def _run_stage1(self, iteration: int, surface_order: list[str]) -> list[dict]:
        assert self.incumbent_ev is not None
        parent_scores = self.incumbent_ev.task_scores()
        all_ids = self.incumbent_ev.task_ids
        branch_rows: list[dict] = []

        for b in range(self.cfg.branches_per_iter):
            surface = surface_order[(iteration * self.cfg.branches_per_iter + b) % len(surface_order)]
            allowed_globs = self.plugin_surfaces[surface]
            batch_ids = balanced_batch(
                all_ids,
                parent_scores,
                self.cfg.batch_size,
                self.cfg.failure_quota,
                self.rng,
                pass_score=self.cfg.pass_score,
            )
            batch_trials = [
                min(self.incumbent_ev.trials[tid], key=lambda tr: tr.score)
                for tid in batch_ids
                if self.incumbent_ev.trials.get(tid)
            ]
            prop = self.proposer.propose_mutation(
                domain=self.domain,
                incumbent=self.incumbent,
                surface=surface,
                allowed_globs=allowed_globs,
                workflow_globs=self.workflow_globs,
                batch_trials=batch_trials,
                library=self.library,
                history=branch_rows,
                iteration=iteration,
                branch=b,
                seed=self.cfg.seed,
            )
            cid = f"r{iteration}_m{b}_{surface}"
            if prop is None:
                branch_rows.append({"id": cid, "surface": surface, "status": "no_proposal"})
                continue
            self.proposer_usage = self.proposer_usage + prop.usage

            ok_surf, why_surf = validate_plugin_surface(
                self.incumbent, prop.artifact, allowed_globs, self.workflow_globs
            )
            if not ok_surf:
                self.ledger.add(
                    Node(
                        id=cid,
                        parent=self.incumbent_node_id,
                        round=iteration,
                        kind="plugin_mutation",
                        status="rejected_surface",
                        change=prop.changes,
                        artifact_id=prop.artifact.id,
                        meta={"surface": surface, "reason": why_surf},
                    )
                )
                self.tr.gate(iteration, cid, False, why_surf)
                branch_rows.append({"id": cid, "surface": surface, "status": "rejected_surface", "reason": why_surf})
                continue

            ok_guard, why_guard = self._passes_guards(self.incumbent, prop.artifact)
            if not ok_guard:
                self.ledger.add(
                    Node(
                        id=cid,
                        parent=self.incumbent_node_id,
                        round=iteration,
                        kind="plugin_mutation",
                        status="rejected_guard",
                        change=prop.changes,
                        artifact_id=prop.artifact.id,
                        meta={"surface": surface, "reason": why_guard},
                    )
                )
                self.tr.gate(iteration, cid, False, why_guard)
                branch_rows.append({"id": cid, "surface": surface, "status": "rejected_guard", "reason": why_guard})
                continue

            patch = extract_plugin_patch(self.incumbent, prop.artifact, allowed_globs)
            if self.library.has_patch(surface, patch):
                branch_rows.append({"id": cid, "surface": surface, "status": "duplicate_patch"})
                continue

            batch_tasks = [self.domain.tasks.get(tid) for tid in batch_ids]
            mb_ev = self.evaluator.evaluate(prop.artifact, batch_tasks, k=self.cfg.trials, label=f"minibatch_{cid}")
            self.n_minibatch_evals += 1
            child_scores = mb_ev.task_scores()
            gain, n_shared = paired_gain(parent_scores, child_scores, batch_ids)
            parent_mb = sum(parent_scores[t] for t in batch_ids) / len(batch_ids) if batch_ids else 0.0
            admitted = n_shared > 0 and gain > self.cfg.min_plugin_gain

            ref = None
            if admitted:
                entry = self.library.admit(
                    surface=surface,
                    files=patch,
                    paired_gain=gain,
                    parent_score=parent_mb,
                    child_score=mb_ev.score,
                    batch_ids=batch_ids,
                    hypothesis=prop.hypothesis,
                    changes=prop.changes,
                    parent_artifact=self.incumbent.id,
                    iteration=iteration,
                )
                ref = entry.ref

            self.ledger.add(
                Node(
                    id=cid,
                    parent=self.incumbent_node_id,
                    round=iteration,
                    kind="plugin_mutation",
                    status="admitted" if admitted else "rejected_gain",
                    score=mb_ev.score,
                    cost=mb_ev.cost,
                    change=prop.changes,
                    artifact_id=prop.artifact.id,
                    meta={"surface": surface, "paired_gain": round(gain, 5), "ref": ref, "n_shared": n_shared},
                )
            )
            self.tr.gate(
                iteration,
                cid,
                admitted,
                f"admitted as {ref} (paired_gain={gain:+.4f})"
                if admitted
                else f"paired_gain={gain:+.4f} <= {self.cfg.min_plugin_gain}",
                math={"paired_gain": gain, "parent_mb": parent_mb, "child_mb": mb_ev.score},
            )
            branch_rows.append(
                {
                    "id": cid,
                    "surface": surface,
                    "status": "admitted" if admitted else "rejected_gain",
                    "paired_gain": round(gain, 5),
                    "ref": ref,
                }
            )
        return branch_rows

    def _run_stage2(self, iteration: int) -> dict:
        assert self.incumbent_ev is not None
        if self._budget_exhausted():
            self.stop_reason = "eval_budget"
            return {"status": "eval_budget"}

        rec = self.proposer.propose_recomposition(
            domain=self.domain,
            incumbent=self.incumbent,
            incumbent_eval=self.incumbent_ev,
            workflow_globs=self.workflow_globs,
            plugin_surfaces=self.plugin_surfaces,
            library=self.library,
            history=self.recompose_history,
            iteration=iteration,
            seed=self.cfg.seed,
        )
        cid = f"r{iteration}_recompose"
        if rec is None:
            return {"id": cid, "status": "no_recomposition", "kept": False}
        self.proposer_usage = self.proposer_usage + rec.usage

        ok_guard, why_guard = self._passes_guards(self.incumbent, rec.artifact)
        if not ok_guard:
            self.ledger.add(
                Node(
                    id=cid,
                    parent=self.incumbent_node_id,
                    round=iteration,
                    kind="recomposition",
                    status="rejected_guard",
                    change=rec.changes,
                    artifact_id=rec.artifact.id,
                    meta={"selected_refs": rec.selected_refs, "reason": why_guard},
                )
            )
            self.tr.gate(iteration, cid, False, why_guard)
            row = {"id": cid, "status": "rejected_guard", "selected_refs": rec.selected_refs, "kept": False}
            self.recompose_history.append(row)
            return row

        ev = self.evaluator.evaluate(rec.artifact, self.cfg.search_split, k=self.cfg.trials)
        self.n_full_evals += 1
        self.tr.evaluation(iteration, cid, ev)

        prev_score = self.incumbent_ev.score
        delta = ev.score - prev_score
        kept = delta > self.cfg.min_recompose_gain
        prev_node = self.incumbent_node_id
        self.ledger.add(
            Node(
                id=cid,
                parent=prev_node,
                round=iteration,
                kind="recomposition",
                status="kept" if kept else "rejected_score",
                score=ev.score,
                cost=ev.cost,
                change=rec.changes,
                artifact_id=rec.artifact.id,
                metrics=ev.family_scores(),
                meta={"selected_refs": rec.selected_refs, "delta": round(delta, 5)},
            )
        )
        self.tr.gate(
            iteration,
            cid,
            kept,
            f"full split {ev.score:.4f} vs incumbent {prev_score:.4f} ({delta:+.4f})",
            math={"score": ev.score, "incumbent": prev_score, "delta": delta},
        )
        if kept:
            self.incumbent = rec.artifact
            self.incumbent_ev = ev
            self.incumbent_node_id = cid
            self.tr.kept(iteration, cid, self.incumbent, decision_score=ev.score)
        self.tr.decision(
            iteration, kept=cid if kept else None, incumbent_before=prev_node, incumbent_after=self.incumbent_node_id
        )
        row = {
            "id": cid,
            "status": "kept" if kept else "rejected_score",
            "selected_refs": rec.selected_refs,
            "score": round(ev.score, 5),
            "delta": round(delta, 5),
            "kept": kept,
        }
        self.recompose_history.append(row)
        return row

    def run(self) -> None:
        seed_ev = self.evaluate_seed()
        self.trajectory.append(
            {
                "round": 0,
                "score": round(seed_ev.score, 5),
                "cost": round(seed_ev.cost, 1),
                "incumbent": self.incumbent.short_id,
                "library_size": len(self.library),
                "admitted_in_round": 0,
                "kept": True,
            }
        )
        surfaces = list(self.plugin_surfaces)
        if not surfaces:
            self.stop_reason = "no_surfaces"
            return

        for t in range(1, self.cfg.iterations + 1):
            if self._budget_exhausted():
                self.stop_reason = "eval_budget"
                break
            lib_before = len(self.library)
            branches = self._run_stage1(t, surfaces)
            admitted_now = len(self.library) - lib_before
            stage2 = self._run_stage2(t)
            if stage2.get("status") == "eval_budget":
                break
            assert self.incumbent_ev is not None
            self.trajectory.append(
                {
                    "round": t,
                    "score": round(self.incumbent_ev.score, 5),
                    "cost": round(self.incumbent_ev.cost, 1),
                    "incumbent": self.incumbent.short_id,
                    "library_size": len(self.library),
                    "admitted_in_round": admitted_now,
                    "kept": bool(stage2.get("kept", False)),
                    "stage1": branches,
                    "stage2": stage2,
                }
            )

"""Public entry point :func:`run` for PluginRSI (``rsi.improve(..., method='pluginrsi')``)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..core.artifact import Artifact
from ..core.domain import Domain
from ..core.llm import LLM, CachedLLM, MockLLM, merged_usage
from ..core.run import ImprovementResult
from ..trace import RunTracer, ShadowMonitor
from .config import Config
from .library import PluginLibrary
from .loop import PluginRSILoop
from .mock import MockPluginProposer
from .proposer import LLMPluginProposer, PluginProposer


def _unwrap(llm: LLM | None) -> LLM | None:
    while isinstance(llm, CachedLLM):
        llm = llm.inner
    return llm


def make_proposer(domain: Domain, llm_propose: LLM | None, config: Config) -> PluginProposer:
    base = _unwrap(llm_propose)
    if llm_propose is None or type(base) is MockLLM:
        if getattr(domain, "world", None) is not None or "harness.py" in domain.seed_artifact():
            return MockPluginProposer(seed=config.seed)
        if llm_propose is None:
            raise ValueError(f"PluginRSI needs llm_propose (or a custom PluginProposer) for domain {domain.name!r}")
    return LLMPluginProposer(llm_propose, trace_chars=config.trace_chars)


def run(
    domain: Domain,
    seed_artifact: Artifact | None = None,
    *,
    llm_task: LLM | None = None,
    llm_propose: LLM | None = None,
    config: Config | None = None,
    out_dir: str | Path | None = None,
    proposer: PluginProposer | None = None,
    seed_library: PluginLibrary | str | Path | None = None,
    monitor: ShadowMonitor | None = None,
) -> ImprovementResult:
    """Run PluginRSI: Stage 1 balanced-minibatch plugin mutation into a reusable library + Stage 2 recomposition."""
    cfg = config or Config()
    seed = seed_artifact if seed_artifact is not None else domain.seed_artifact()
    out = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="pluginrsi_"))
    prop = proposer or make_proposer(domain, llm_propose, cfg)

    tracer = RunTracer(out if (out_dir is not None and cfg.trace) else None, "pluginrsi")
    if tracer.enabled:
        if monitor is not None:
            tracer.monitor = monitor
        elif cfg.shadow_monitor:
            tracer.monitor = ShadowMonitor(domain, llm_task, splits=cfg.shadow_splits, k=cfg.shadow_k)
        tracer.event("run_start", seed=seed.short_id, config=cfg.to_json())

    loop = PluginRSILoop(
        domain,
        seed,
        llm_task=llm_task,
        proposer=prop,
        config=cfg,
        out_dir=out,
        seed_library=seed_library,
        tracer=tracer,
    )
    loop.run()

    usage = merged_usage(llm_task, llm_propose)
    usage["proposer"] = loop.proposer_usage.to_dict()
    if llm_propose is None and loop.proposer_usage.calls > 0:
        for k, v in loop.proposer_usage.to_dict().items():
            usage["_total"][k] = usage["_total"].get(k, 0) + v

    return ImprovementResult(
        method="pluginrsi",
        baseline=seed,
        best=loop.incumbent,
        ledger=loop.ledger,
        trajectory=loop.trajectory,
        usage=usage,
        stop_reason=loop.stop_reason,
        out_dir=str(out),
        meta={
            "config": cfg.to_json(),
            "library_size": len(loop.library),
            "admitted_plugins": loop.library.catalog_summary(max_entries=100),
            "plugin_library_dir": str(out / "plugin_library"),
            "n_full_evals": loop.n_full_evals,
            "n_minibatch_evals": loop.n_minibatch_evals,
            "plugin_surfaces": {k: list(v) for k, v in loop.plugin_surfaces.items()},
            "workflow_globs": list(loop.workflow_globs),
        },
    )

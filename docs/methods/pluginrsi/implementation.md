# PluginRSI — Implementation in `rsi`

This page documents how the PluginRSI specification (`docs/methods/pluginrsi/paper-spec.md`) is implemented on top of
`rsi.core` and exposed through `rsi.improve(..., method="pluginrsi")` (aliases `plugin-rsi`, `plugin_rsi`).

## Module map

| Module | Responsibility |
|---|---|
| `rsi/pluginrsi/config.py` | `Config` dataclass: `iterations`, `branches_per_iter`, `batch_size`, `failure_quota`, `pass_score`, `plugin_components`, `workflow_components`, `min_plugin_gain`, `min_recompose_gain`, `eval_budget`, `leakage_screen`, `validate`. |
| `rsi/pluginrsi/mutation.py` | `resolve_surfaces`, `balanced_batch`, `paired_gain`, `validate_plugin_surface`, and `extract_plugin_patch`. |
| `rsi/pluginrsi/library.py` | `PluginEntry` and persistent `PluginLibrary` (`library.json` + per-version manifests, `apply`, and `import_from` for `seed_library` reuse). |
| `rsi/pluginrsi/proposer.py` | `PluginProposer` interface and `LLMPluginProposer` (JSON or `=== FILE: ... ===` protocol for Stage 1 and Stage 2). |
| `rsi/pluginrsi/mock.py` | Deterministic offline `MockPluginProposer` supporting `HarnessWorldDomain` and `AgentQADomain`-layout domains (`AgentQA`, `BBH`, `BBEH`). |
| `rsi/pluginrsi/loop.py` | `PluginRSILoop`: seed evaluation, Stage 1 isolated minibatch mutations, library admission, Stage 2 full-split recomposition, `Ledger` and `RunTracer` events. |
| `rsi/pluginrsi/api.py` | `run()` returning `ImprovementResult` with `library_size`, `admitted_plugins`, and `plugin_library_dir` in `meta`. |

## Deviations and package choices

1. **Domain-agnostic surface mapping:** The reference repo uses a `harness.yaml` manifest with versioned plugin
   directories (`new_plugins/<kind>/<name>/<version>/`). To work across every `rsi.core.Domain` without requiring a custom
   YAML manifest, `resolve_surfaces` maps `Domain.components` into `workflow_components` (default `("control_flow", "config")`)
   and `plugin_components` (all remaining declared components such as `client_tool`/`tool`, `skill`, `memory`, `subagent`,
   `prompt`), storing verified file-level patches per surface in `PluginLibrary`.
2. **Beam width `W = 1` incumbent tracking:** The reference repo supports beam width `top_w`; `rsi.pluginrsi` tracks the
   single best full-split incumbent `H*` (matching `W = 1`) while retaining every positively-gaining Stage 1 plugin in
   `PluginLibrary` across all iterations so no verified local discovery is lost when a Stage 2 recomposition is rejected.
3. **Optional safeguards:** Supports `rsi`'s `LeakageCritic` (`leakage_screen=True`), `InterfaceValidator` (`validate=True`),
   `eval_budget` (counting full search-split evaluations), and `ShadowMonitor` on sealed `holdout`/`ood` splits while
   keeping sealed splits strictly unreadable to the loop itself.

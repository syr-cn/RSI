# PluginRSI: modular plugin mutation + recomposition (`rsi.improve(..., method="pluginrsi")`)

Sources: Shi et al., *"PluginRSI: Recursive Improvement of Agent Harnesses with Reusable Plugins"*, arXiv:2609.32423,
summarised in `docs/methods/pluginrsi/paper-spec.md`; package notes in `docs/methods/pluginrsi/implementation.md` and
`rsi/pluginrsi/{config,mutation,library,proposer,mock,loop,api}.py`.

## 1. Core idea

PluginRSI represents an agent harness as `H(W, P)`—a global orchestration workflow `W` over modular plugin surfaces `P`
(roles/subagents, tools, skills, memory, prompts). Instead of rewriting the whole harness at once on every step, each
iteration runs two stages:

1. **Stage 1 (`plugin_mutation`):** For `branches_per_iter` branches, sample a balanced minibatch of failed
   (`score < pass_score`) and solved (`score >= pass_score`) tasks from the incumbent's search-split evaluation. Mutate
   **only** one assigned plugin surface while keeping workflow files and all sibling plugin surfaces frozen
   (`validate_plugin_surface`). Evaluate on that minibatch alone and admit the plugin patch into the persistent
   `PluginLibrary` iff its paired score gain on shared minibatch tasks exceeds `min_plugin_gain` (default `0.0`).
2. **Stage 2 (`harness_recomposition`):** Propose a complete harness that combines compatible positive-gain plugins from
   `PluginLibrary` and optionally updates workflow files `W`. Evaluate on the full `search_split` (`"evolve"`) and
   replace the incumbent iff the full-split score exceeds `incumbent_score + min_recompose_gain` (default `0.0`).

## 2. Key configuration (`rsi.pluginrsi.Config`)

- `iterations=4`, `branches_per_iter=4`: outer rounds and Stage 1 single-surface branches per round.
- `batch_size=8`, `failure_quota=4`, `pass_score=1.0`: balanced solved/failed minibatch sampling parameters.
- `workflow_components=("control_flow", "config")`, `plugin_components=None`: partitions `Domain.components` into `W`
  and `P` (`None` selects all non-workflow components declared by the domain).
- `min_plugin_gain=0.0`, `min_recompose_gain=0.0`: strict paired gain for Stage 1 library admission and strict full-split
  gain for Stage 2 incumbent replacement.
- `eval_budget=None`, `leakage_screen=False`, `validate=True`: optional full-evaluation cap, `LeakageCritic`, and
  `InterfaceValidator`.

## 3. Cross-run and cross-policy reuse

Pass `seed_library="runs/prev/plugin_library"` (or a `PluginLibrary` instance) to `rsi.improve(..., method="pluginrsi")`
to import verified plugin variants from an earlier run or another frozen policy before round 1.

```python
import rsi
from rsi.domains.harnessworld import WEAK, make_domain

dom_strong = make_domain(seed=0)
r1 = rsi.improve(dom_strong, method="pluginrsi", config={"iterations": 3}, out_dir="runs/prsi_strong")

dom_weak = dom_strong.with_policy(WEAK)
r2 = rsi.improve(
    dom_weak,
    method="pluginrsi",
    config={"iterations": 2},
    seed_library=r1.meta["plugin_library_dir"],
    out_dir="runs/prsi_weak",
)
```

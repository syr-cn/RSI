# PluginRSI — Paper & Code Specification

This file is a reimplementation spec: every statement is tagged with its source in Shi et al., *"PluginRSI: Recursive
Improvement of Agent Harnesses with Reusable Plugins"* (`arXiv:2609.32423`, cited as `[paper §x]`, `[paper Alg. 1]`) or
the official reference release `syr-cn/PluginRSI` (`src/pluginrsi/search/{loop,mutation,selection,store}.py` and
`src/pluginrsi/prompts/{plugin_mutation,harness_recomposition}.md`, cited as `[code <file>]`).

## 1. Formulation

- **Modular harness representation `[paper §3.1]`:** An agent harness is parameterized as `H(W, P)` around a frozen
  solver model `M`, where `W` is the global orchestration workflow and `P` is a collection of typed, reusable plugin
  modules (`roles`, `tools`, `skills`, `memory`).
- **Credit-assignment bottleneck `[paper §1, §3]`:** Monolithic full-harness rewrites (as in Meta-Harness) bundle edits
  across multiple components and require expensive full-split rollouts to validate every hypothesis, obscuring which
  local change fixed or broke a task.
- **Persistent plugin library `[paper §3.2]`:** Verified local improvements accumulate in a shared plugin library `L`
  that persists across search iterations and can be transferred across runs or solver models without re-discovering
  every mechanism from scratch.

## 2. Two-Stage Evolutionary Loop (`[paper Alg. 1]`, `[code search/loop.py]`)

Given seed harness `H_0 = H(W_0, P_0)`, held-in search split `X_search`, minibatch size `B`, failure quota `q_fail`, and
`N` Stage 1 branches per iteration:

1. **Seed evaluation `[code search/loop.py]`:** Evaluate `H_0` on the full search split `X_search` to record per-task
   scores and execution traces. Set incumbent `H* <- H_0`.
2. **Stage 1 — Isolated Plugin Mutation (`plugin_mutation`) `[paper §3.2, code search/mutation.py]`:**
   For each branch `b = 1..N`:
   - Assign one editable plugin surface `s_b` from `P`.
   - Sample a balanced minibatch `X_b` of size `B` (`balanced_batch` in `[code search/mutation.py]`): shuffle failed
     incumbent tasks (`score < 1`) and solved tasks (`score == 1`), take up to `q_fail` failed tasks and `B - q_fail`
     solved tasks, and backfill from remaining tasks if either pool is short.
   - Propose a mutated harness `H'_b` restricted to `s_b` (`[code prompts/plugin_mutation.md]`): keep workflow files
     `W` and all sibling plugin surfaces byte-for-byte unchanged (`validate_local_surface` in `[code search/mutation.py]`).
   - Evaluate `H'_b` only on minibatch `X_b`. Compute the paired score delta `paired_gain = mean_{x in X_b} (r(H'_b, x) - r(H*, x))`
     over shared valid tasks (`paired` admission policy in `[code search/loop.py]`).
   - Admit the mutated plugin into the shared library `L` iff `paired_gain > 0` (`matched_gain > 0` in `[code search/loop.py]`).
3. **Stage 2 — Global Harness Recomposition (`harness_recomposition`) `[paper §3.3, code prompts/harness_recomposition.md]`:**
   - Expose the incumbent `H*`, representative execution traces, and the library `L` (prioritizing positive-gain
     plugins alongside their local hypotheses and diffs).
   - Propose a recomposed harness `H_cand = H(W', P')` that selects compatible plugins `P'` from `L` and optionally
     updates workflow `W'` so the selected plugins are wired into execution.
   - Evaluate `H_cand` on the full held-in search split `X_search`.
   - Replace the incumbent `H* <- H_cand` iff the full search-split score strictly improves over `H*`.
4. **Cross-run & cross-model transfer `[paper §4]`:** Because `L` stores self-contained plugin patches with provenance
   and local paired-gain metadata, a library populated under one run or solver policy can seed `PluginLibrary` for a
   different solver policy (`seed_library`).

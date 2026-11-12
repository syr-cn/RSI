# PluginRSI

Recursive improvement of agent harnesses with reusable plugins (Shi et al., arXiv:2609.32423; code `syr-cn/PluginRSI`):
decomposes an agent harness $H(W, P)$ into a workflow $W$ over modular plugin surfaces $P$ (roles/subagents, tools,
skills, memory, prompts), isolates single-plugin mutations on balanced solved/failed minibatches in Stage 1 to admit
strictly improving variants into a persistent shared `PluginLibrary`, and recomposes library plugins plus workflow code
under full search-split evaluation in Stage 2.

1. [paper-spec.md](paper-spec.md): the reimplementation spec with section and reference-code citations.
2. [implementation.md](implementation.md): module map, API, deviations and limitations of `rsi/pluginrsi`.

The skill reference `.claude/skills/rsi-improve/methods/pluginrsi.md` is quote-checked in
`docs/skill-claims/ledger/pluginrsi.jsonl`.

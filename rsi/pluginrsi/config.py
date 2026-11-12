"""Configuration for PluginRSI (Shi et al., arXiv:2609.32423)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DEFAULT_ITERATIONS = 4
DEFAULT_BRANCHES_PER_ITER = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_FAILURE_QUOTA = 4
DEFAULT_PASS_SCORE = 1.0
DEFAULT_SEARCH_SPLIT = "evolve"
DEFAULT_TRIALS = 1
DEFAULT_WORKERS = 4
DEFAULT_WORKFLOW_COMPONENTS = ("control_flow", "config")
DEFAULT_VALIDATE_TIMEOUT_S = 30.0
DEFAULT_TRACE_CHARS = 3000


@dataclass
class Config:
    """Hyperparameters for PluginRSI's two-stage plugin mutation + harness recomposition loop."""

    iterations: int = DEFAULT_ITERATIONS
    branches_per_iter: int = DEFAULT_BRANCHES_PER_ITER
    batch_size: int = DEFAULT_BATCH_SIZE
    failure_quota: int = DEFAULT_FAILURE_QUOTA
    pass_score: float = DEFAULT_PASS_SCORE
    search_split: str = DEFAULT_SEARCH_SPLIT
    trials: int = DEFAULT_TRIALS
    workers: int = DEFAULT_WORKERS
    eval_budget: int | None = None
    plugin_components: tuple[str, ...] | None = None
    workflow_components: tuple[str, ...] = DEFAULT_WORKFLOW_COMPONENTS
    min_plugin_gain: float = 0.0
    min_recompose_gain: float = 0.0
    leakage_screen: bool = False
    validate: bool = True
    validate_in_subprocess: bool = False
    validate_timeout_s: float = DEFAULT_VALIDATE_TIMEOUT_S
    trace_chars: int = DEFAULT_TRACE_CHARS
    seed: int = 0
    trace: bool = True
    shadow_monitor: bool = False
    shadow_splits: tuple[str, ...] = ("holdout", "ood")
    shadow_k: int = 1

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.branches_per_iter < 0:
            raise ValueError(f"branches_per_iter must be >= 0, got {self.branches_per_iter}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.failure_quota < 0:
            raise ValueError(f"failure_quota must be >= 0, got {self.failure_quota}")
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if self.plugin_components is not None:
            self.plugin_components = tuple(self.plugin_components)
        self.workflow_components = tuple(self.workflow_components)
        self.shadow_splits = tuple(self.shadow_splits)

    def to_json(self) -> dict:
        d = asdict(self)
        if d["plugin_components"] is not None:
            d["plugin_components"] = list(d["plugin_components"])
        d["workflow_components"] = list(d["workflow_components"])
        d["shadow_splits"] = list(d["shadow_splits"])
        return d

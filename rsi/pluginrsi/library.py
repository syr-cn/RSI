"""Persistent shared Plugin Library ``L`` storing verified single-surface plugin variants."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.artifact import Artifact

LIBRARY_INDEX = "library.json"


@dataclass
class PluginEntry:
    """One verified plugin variant admitted from Stage 1 balanced-minibatch evaluation."""

    ref: str
    surface: str
    version: int
    files: dict[str, str | None]
    paired_gain: float
    parent_score: float
    child_score: float
    batch_ids: list[str] = field(default_factory=list)
    hypothesis: str = ""
    changes: str = ""
    parent_artifact: str = ""
    iteration: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> PluginEntry:
        return cls(
            ref=str(d["ref"]),
            surface=str(d["surface"]),
            version=int(d["version"]),
            files=dict(d.get("files") or {}),
            paired_gain=float(d["paired_gain"]),
            parent_score=float(d.get("parent_score", 0.0)),
            child_score=float(d.get("child_score", 0.0)),
            batch_ids=list(d.get("batch_ids") or []),
            hypothesis=str(d.get("hypothesis", "")),
            changes=str(d.get("changes", "")),
            parent_artifact=str(d.get("parent_artifact", "")),
            iteration=int(d.get("iteration", 0)),
        )


class PluginLibrary:
    """Accumulates validated plugin patches across iterations and exports/imports them across runs or policies."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self._entries: dict[str, PluginEntry] = {}
        self._next_ver: dict[str, int] = {}
        if self.root is not None and (self.root / LIBRARY_INDEX).is_file():
            self._load_disk()

    def _load_disk(self) -> None:
        assert self.root is not None
        raw = json.loads((self.root / LIBRARY_INDEX).read_text(encoding="utf-8"))
        for row in raw.get("entries", []):
            entry = PluginEntry.from_json(row)
            self._entries[entry.ref] = entry
            self._next_ver[entry.surface] = max(self._next_ver.get(entry.surface, 1), entry.version + 1)

    def save(self) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "entries": [e.to_json() for e in self.entries()]}
        tmp = self.root / f".{LIBRARY_INDEX}.tmp"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.root / LIBRARY_INDEX)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, ref: str) -> bool:
        return ref in self._entries

    def get(self, ref: str) -> PluginEntry:
        return self._entries[ref]

    def entries(self, surface: str | None = None) -> list[PluginEntry]:
        rows = [e for e in self._entries.values() if surface is None or e.surface == surface]
        return sorted(rows, key=lambda e: (-e.paired_gain, e.surface, e.version))

    def best_per_surface(self) -> dict[str, PluginEntry]:
        best: dict[str, PluginEntry] = {}
        for entry in self.entries():
            best.setdefault(entry.surface, entry)
        return best

    def has_patch(self, surface: str, files: dict[str, str | None]) -> bool:
        return any(e.surface == surface and e.files == files for e in self._entries.values())

    def admit(
        self,
        *,
        surface: str,
        files: dict[str, str | None],
        paired_gain: float,
        parent_score: float,
        child_score: float,
        batch_ids: Sequence[str],
        hypothesis: str = "",
        changes: str = "",
        parent_artifact: str = "",
        iteration: int = 0,
    ) -> PluginEntry:
        ver = self._next_ver.get(surface, 1)
        self._next_ver[surface] = ver + 1
        ref = f"{surface}/v{ver:03d}"
        entry = PluginEntry(
            ref=ref,
            surface=surface,
            version=ver,
            files=dict(files),
            paired_gain=float(paired_gain),
            parent_score=float(parent_score),
            child_score=float(child_score),
            batch_ids=list(batch_ids),
            hypothesis=hypothesis,
            changes=changes,
            parent_artifact=parent_artifact,
            iteration=iteration,
        )
        self._entries[ref] = entry
        if self.root is not None:
            snap_dir = self.root / surface / f"v{ver:03d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "manifest.json").write_text(json.dumps(entry.to_json(), indent=2) + "\n", encoding="utf-8")
        self.save()
        return entry

    def import_from(self, source: PluginLibrary | str | Path) -> int:
        """Import entries from another :class:`PluginLibrary` or directory (cross-run / cross-model plugin reuse)."""
        other = source if isinstance(source, PluginLibrary) else PluginLibrary(source)
        added = 0
        for entry in sorted(other._entries.values(), key=lambda e: (e.iteration, e.surface, e.version)):
            if self.has_patch(entry.surface, entry.files):
                continue
            self.admit(
                surface=entry.surface,
                files=entry.files,
                paired_gain=entry.paired_gain,
                parent_score=entry.parent_score,
                child_score=entry.child_score,
                batch_ids=entry.batch_ids,
                hypothesis=entry.hypothesis,
                changes=entry.changes,
                parent_artifact=entry.parent_artifact,
                iteration=entry.iteration,
            )
            added += 1
        return added

    def apply(self, base: Artifact, refs: Sequence[str]) -> Artifact:
        """Overlay selected library plugin patches onto ``base`` in the given order."""
        updates: dict[str, str | None] = {}
        for ref in refs:
            entry = self._entries[ref]
            updates.update(entry.files)
        return base.with_files(updates)

    def catalog_summary(self, max_entries: int = 24) -> list[dict]:
        return [
            {
                "ref": e.ref,
                "surface": e.surface,
                "paired_gain": round(e.paired_gain, 4),
                "hypothesis": e.hypothesis,
                "changes": e.changes,
                "paths": sorted(e.files),
            }
            for e in self.entries()[:max_entries]
        ]

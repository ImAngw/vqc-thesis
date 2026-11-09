"""Metrics logging: one flat dict {name: [values]}, flushed atomically to metrics.json."""

import json
import os
from pathlib import Path


class MetricsLogger:
    """Rows are appended per group (`train`, `eval`, `test`); names are stored as `group/key`.

    Every row of a group must carry the same keys, so all lists of a group stay equally long.
    """

    def __init__(self, log_dir: str | Path, run_name: str):
        self.run_dir = Path(log_dir) / run_name
        if self.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self.metrics: dict[str, list[float]] = {}
        self._group_keys: dict[str, set[str]] = {}

    @property
    def path(self) -> Path:
        return self.run_dir / "metrics.json"

    def log(self, group: str, row: dict[str, float]) -> None:
        keys = set(row)
        known = self._group_keys.setdefault(group, keys)
        if keys != known:
            raise ValueError(f"group {group!r} expects keys {sorted(known)}, got {sorted(keys)}")
        for key, value in row.items():
            self.metrics.setdefault(f"{group}/{key}", []).append(float(value))

    def flush(self) -> None:
        """Atomic write: a partially written file is never visible."""
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.metrics, f)  # NaN is written as a bare NaN literal (Python json reads it back)
        os.replace(tmp, self.path)

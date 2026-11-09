"""Optional training-side helper: per-block gradient statistics for the train-metrics monitor.

Copy or import this into the training project (the plugin never runs it and
never edits your code). Call it after ``loss.backward()`` and before gradient
clipping, so clipping cannot hide an exploding block::

    grad_log = GradLogger(model, every=50)          # prints the block grouping once
    ...
    loss.backward()
    grad_log.record(metrics, update)                # appends to the lists of your metrics dict
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

With mixed precision call ``scaler.unscale_(optimizer)`` first, otherwise the
norms are scaled. The keys follow the contract in SPEC-grad-monitor.md:

    grad/<block>/rms        L2 norm of the block's gradient / sqrt(numel)
    grad/<block>/zero_frac  fraction of exactly-zero gradient elements
    grad/<x_key>            x value of those points (default ``grad/update``)

Only the standard library is imported at module level; torch is needed by
``GradLogger.log`` alone.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

Metrics = dict[str, list[float]]


@dataclass(frozen=True)
class BlockStats:
    sumsq: float  # sum of squared gradient elements of the block
    numel: int  # number of trainable elements in the block
    zeros: int  # elements with an exactly-zero (or missing) gradient


def _default_block(name: str) -> str:
    """``encoder.layers.0.attn.weight`` -> ``encoder.layers.0``; ``head.bias`` -> ``head``."""
    parts = name.split(".")
    if len(parts) == 1:
        return "root"
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return parts[0]


def _matches(name: str, prefix: str) -> bool:
    prefix = prefix.rstrip(".")
    return name == prefix or name.startswith(prefix + ".")


def group_params(names: list[str], groups: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Map block name -> parameter names.

    Default: the prefix up to the first numbered module (``layers.3``), else the
    first component. ``groups`` overrides it: block -> name prefixes, first match
    wins; names matching no override keep the default grouping.
    """
    result: dict[str, list[str]] = {}
    for name in names:
        block = next(
            (b for b, prefixes in (groups or {}).items() if any(_matches(name, p) for p in prefixes)),
            None,
        ) or _default_block(name)
        result.setdefault(block, []).append(name)
    return result


def entries(stats: dict[str, BlockStats], step: float, x_key: str = "update", prefix: str = "grad") -> dict[str, float]:
    """Flat ``{key: value}`` for one checkpoint, in the monitor's key contract."""
    out: dict[str, float] = {f"{prefix}/{x_key}": float(step)}
    for block, s in stats.items():
        out[f"{prefix}/{block}/rms"] = math.sqrt(s.sumsq / s.numel) if s.numel else 0.0
        out[f"{prefix}/{block}/zero_frac"] = s.zeros / s.numel if s.numel else 1.0
    return out


def append_entries(metrics: Metrics, new: dict[str, float]) -> None:
    """Append each value to its list in ``metrics`` (created on first use)."""
    for key, value in new.items():
        metrics.setdefault(key, []).append(value)


class GradLogger:
    """Collects per-block gradient statistics every ``every`` steps."""

    def __init__(
        self,
        model,
        every: int = 50,
        prefix: str = "grad",
        x_key: str = "update",
        groups: dict[str, list[str]] | None = None,
        print_fn: Callable[[str], None] = print,
    ):
        if every < 1:
            raise ValueError("every must be >= 1")
        self.every, self.prefix, self.x_key = every, prefix, x_key
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._blocks = {b: [params[n] for n in ns] for b, ns in group_params(list(params), groups).items()}
        print_fn(
            f"[grad_logger] {len(self._blocks)} blocks every {every} steps: "
            + ", ".join(f"{b} ({len(ps)} tensors)" for b, ps in self._blocks.items())
        )

    def _collect(self) -> dict[str, BlockStats]:
        import torch  # imported here so the pure helpers work without torch

        flat, meta = [], []
        for ps in self._blocks.values():
            grads = [p.grad.detach() for p in ps if p.grad is not None]
            device = ps[0].device
            sumsq = sum((g.float().pow(2).sum() for g in grads), torch.zeros((), device=device))
            zeros = sum(((g == 0).sum().float() for g in grads), torch.zeros((), device=device))
            flat.append(torch.stack([sumsq, zeros]))
            meta.append((sum(p.numel() for p in ps), sum(p.numel() for p in ps if p.grad is None)))
        values = torch.stack(flat).cpu().tolist()  # one device sync for all blocks
        return {
            block: BlockStats(sumsq=v[0], numel=numel, zeros=int(round(v[1])) + missing)
            for block, (v, (numel, missing)) in zip(self._blocks, zip(values, meta))
        }

    def log(self, step: int) -> dict[str, float] | None:
        """Entries for this step, or None when ``step`` is not a multiple of ``every``."""
        if step % self.every:
            return None
        return entries(self._collect(), step, self.x_key, self.prefix)

    def record(self, metrics: Metrics, step: int) -> bool:
        """Append this step's entries to ``metrics``; True if the step was logged."""
        new = self.log(step)
        if new is None:
            return False
        append_entries(metrics, new)
        return True

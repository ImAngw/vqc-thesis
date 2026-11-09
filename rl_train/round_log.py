"""Round log in the text format read by scripts/summarize_gen_pilot.py and scripts/plot_gen_curve.py.

One block per round (`inner_steps` updates):

    round 3/50
      node    steps  steps/s       loss  sync/round  state  <columns>
      local     150     16.4     0.5644        0.0s  active  <values>
      TOTAL     150     16.4     0.5644        0.0s          <values>

The readers take the TOTAL row and match its values to the header names by order; "-" means not available."""

from pathlib import Path


class RoundLog:
    def __init__(self, path: Path, total_rounds: int, columns: list[str]):
        self.path = Path(path)
        self.total_rounds = total_rounds
        self.columns = columns
        self.path.write_text("")

    def write(self, rnd: int, steps: int, seconds: float, loss: float, values: dict[str, float]) -> None:
        """Append round `rnd` (1-based). `values` maps column -> number; a missing column is written as "-"."""
        cells = [f"{values[c]:.4g}" if c in values else "-" for c in self.columns]
        header = "".join(f"  {c:>12s}" for c in self.columns)
        speed = steps / seconds if seconds > 0 else 0.0
        row = f"{steps:>7d} {speed:>8.1f} {loss:>10.4f} {'0.0s':>11s}"
        body = "".join(f"  {c:>12s}" for c in cells)
        text = (f"round {rnd}/{self.total_rounds}\n"
                f"  node    steps  steps/s       loss  sync/round  state{header}\n"
                f"  local {row}  active{body}\n"
                f"  TOTAL {row}        {body}\n")
        with open(self.path, "a") as f:
            f.write(text)

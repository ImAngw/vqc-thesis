"""
Base class of the generated VQC modules.

A VQC module is a function features (B, F) -> logits (B, A). The base class owns the plumbing that does not
depend on the circuit; a subclass defines the circuit and the readout:

    n_qubits, n_measured   class attributes. The first n_measured qubits (big-endian) are the measured
                           register, the others are never measured (ancillas): they get no readout error.
    fixed_blocks()         optional, list of 2^n x 2^n matrices of the data-independent gate blocks
    state(features)        clean path: statevector (B, 2^n), real or complex
    density_matrix(features, noise)
                           noisy path: rho (B, 2^n, 2^n), depolarizing noise after every nominal gate
    readout(probs)         probs (B, 2^n_measured), the Born probabilities of the measured register (with
                           readout error in the noisy path) -> logits

forward(features) runs the clean path; forward(features, noise=NoiseParams(...)) the density-matrix path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vqc_core import density as dm
from vqc_core import noise as nz
from vqc_core import statevector as sv


class VQCBase(nn.Module):
    n_qubits: int
    n_measured: int

    def __init__(self):
        super().__init__()
        self._blocks_key, self._blocks = None, None

    # --- to be defined by the circuit -----------------------------------------------------------
    def fixed_blocks(self) -> list[torch.Tensor]:
        """Matrices of the blocks of gates that depend only on the parameters (none by default)."""
        return []

    def state(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def density_matrix(self, features: torch.Tensor, noise: nz.NoiseParams) -> torch.Tensor:
        raise NotImplementedError

    def readout(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # --- shared plumbing ------------------------------------------------------------------------
    def blocks(self) -> list[torch.Tensor]:
        """With gradient: rebuilt at every forward. Without: cached until a parameter changes (the _version
        of a tensor grows at every in-place update, e.g. optimizer.step or load_state_dict)."""
        if torch.is_grad_enabled():
            return self.fixed_blocks()
        params = list(self.parameters())
        key = tuple(p._version for p in params) + (params[0].device,)
        if key != self._blocks_key:
            self._blocks, self._blocks_key = self.fixed_blocks(), key
        return self._blocks

    def measured_probabilities(self, features: torch.Tensor, noise: nz.NoiseParams | None = None) -> torch.Tensor:
        """Born probabilities of the measured register, (B, 2^n_measured); in the noisy path the readout
        error (noise.p01, noise.p10: float, scalar tensor or one value per measured qubit) is applied."""
        if noise is None:
            return sv.register_probabilities(self.state(features), self.n_measured)
        probs = dm.register_probabilities(self.density_matrix(features, noise), self.n_measured)
        return nz.apply_readout(probs, noise.p01, noise.p10)

    def forward(self, features: torch.Tensor, noise: nz.NoiseParams | None = None) -> torch.Tensor:
        return self.readout(self.measured_probabilities(features, noise))

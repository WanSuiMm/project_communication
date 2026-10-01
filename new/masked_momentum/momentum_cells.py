"""Generic momentum using the frozen mask operator and parameter factory."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'masked_medium'))
from masked_cells import MaskedCell, masked_laplacian, frozen_cells

ARM = 'masked_momentum_nca'


class MaskedMomentum(MaskedCell):
    def step(self, state, x):
        h, v = state
        lh = masked_laplacian(h, x[:, :1])
        force = self.base.eta * self.base.program(torch.cat((h, lh, x), dim=1))
        beta, _ = self.base.coefficients()
        v = beta * v + force
        return h + v, v


def make_cell(arm=ARM, channels=16, reference_hidden=128):
    if arm != ARM:
        raise ValueError(arm)
    return MaskedMomentum(frozen_cells.make_cell('momentum_nca', channels, reference_hidden), arm)

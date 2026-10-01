"""Ordinary state-matched NCA on the frozen binary input-mask graph."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'masked_medium'))
from masked_cells import MaskedCell, masked_laplacian, frozen_cells

ARM = 'masked_state_nca'


class MaskedState(MaskedCell):
    def step(self,state,x):
        h,v = state
        if v is not None:
            raise ValueError('First-order state model has no velocity')
        lh = masked_laplacian(h,x[:,:1])
        force = self.base.eta*self.base.program(torch.cat((h,lh,x),dim=1))
        return h+force,None


def make_cell(arm=ARM,channels=16,reference_hidden=128):
    if arm != ARM:
        raise ValueError(arm)
    return MaskedState(frozen_cells.make_cell('nca_state_matched',channels,reference_hidden),arm)

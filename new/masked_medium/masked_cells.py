"""Frozen structured cells with only the input-derived diffusion graph changed."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

FROZEN=Path(__file__).resolve().parents[1]/'nca_inertial_wind_tunnel'
if str(FROZEN) not in sys.path:
    sys.path.insert(0,str(FROZEN))
import cells as frozen_cells

ARMS=('masked_rd_nca','masked_inertial_rd')
BASE_ARMS={'masked_rd_nca':'rd_nca','masked_inertial_rd':'inertial_rd'}


def masked_laplacian(h,mask):
    """L_m h = sum_j m_i*m_j*(h_i-h_j), with replicated outer ghosts.

    For binary input masks the operator is symmetric PSD, lambda_max<=8.
    Wall nodes have no transport edges; their local state is otherwise intact.
    """
    z=F.pad(h,(1,1,1,1),mode='replicate')
    m=F.pad(mask,(1,1,1,1),mode='replicate')
    return (mask*m[:,:,1:-1,:-2]*(h-z[:,:,1:-1,:-2])
            +mask*m[:,:,1:-1,2:]*(h-z[:,:,1:-1,2:])
            +mask*m[:,:,:-2,1:-1]*(h-z[:,:,:-2,1:-1])
            +mask*m[:,:,2:,1:-1]*(h-z[:,:,2:,1:-1]))


class MaskedCell(nn.Module):
    def __init__(self,base,arm):
        super().__init__()
        self.base=base
        self.arm=arm

    def initial(self,x):
        return self.base.initial(x)

    def step(self,state,x):
        h,v=state
        b,d=self.base.coefficients()
        force=self.base.eta*self.base.program(torch.cat((h,x),dim=1))
        force=force-d*masked_laplacian(h,x[:,:1])
        if self.base.inertial:
            v=b*v+force
            return h+v,v
        return h+force,None

    def logits(self,state):
        return self.base.logits(state)

    def rollout(self,x,steps,state=None):
        if steps<0:
            raise ValueError('steps must be nonnegative')
        state=self.initial(x) if state is None else state
        for _ in range(steps):
            state=self.step(state,x)
        return state

    def metadata(self):
        return {**self.base.metadata(),'arm':self.arm,'base_arm':self.base.arm,
                'medium':'binary_input_mask_edges_m_i_times_m_j',
                'operator_parameter_count':0}


def make_cell(arm,channels=16,reference_hidden=128):
    if arm not in BASE_ARMS:
        raise ValueError(f'Unknown masked arm: {arm}')
    # Preserve the original factory's dummy construction and every RNG draw.
    base=frozen_cells.make_cell(BASE_ARMS[arm],channels,reference_hidden)
    return MaskedCell(base,arm)

"""One fixed spatial carry path; all learned components remain frozen in form."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/workspace_revision'))
from revision_cells import RevisionCell, masked_laplacian

VARIANTS = ('baseline', 'carry')
RHO = 0.5


def masked_degree(mask):
    """Count real open neighbors; replicated ghosts are NOT graph edges."""
    m = F.pad(mask, (1, 1, 1, 1))
    return mask * (m[:, :, 1:-1, :-2] + m[:, :, 1:-1, 2:]
                   + m[:, :, :-2, 1:-1] + m[:, :, 2:, 1:-1])


def spatial_carry(workspace, mask, rho=RHO, laplacian=None):
    if not 0 <= rho <= 1:
        raise ValueError('Carry weight must be a convex weight')
    lap = masked_laplacian(workspace, mask) if laplacian is None else laplacian
    # Binary walls and isolated nodes have L(W)=0 and therefore retain W.
    return workspace - rho * lap / masked_degree(mask).clamp_min(1)


class DirectCarryCell(RevisionCell):
    def __init__(self, rho=RHO, **kwargs):
        super().__init__('ws_additive', **kwargs)
        self.rho = rho

    def step(self, state, x):
        self._validate_x(x)
        workspace, latent = state
        features = self._features(workspace, latent, x)
        force = self.f_out(torch.tanh(self.f_in(features)))
        # Reuse exactly the old L(W) feature; do not change F's inputs.
        offset = self.workspace_channels + self.latent_channels
        lap = features[:, offset:offset + self.workspace_channels]
        carried = spatial_carry(workspace, x[:, :1], self.rho, lap)
        workspace_new = carried + self.eta * force
        candidate = self._candidate(workspace_new, latent, x)
        return workspace_new, latent + self.alpha * candidate

    def metadata(self):
        return {**super().metadata(), 'architecture': 'direct_spatial_carry',
                'rho': self.rho, 'carry_parameters': 0,
                'workspace_update': 'W_new = W - rho*D_dagger*L(W) + eta*F',
                'carry_degree': 'real binary-mask neighbors; no ghost self-edges',
                'carry_only_linf_nonexpansive': True, 'stability_guarantee': False}


def make_variant(variant):
    if variant == 'baseline':
        return RevisionCell('ws_additive')
    if variant == 'carry':
        return DirectCarryCell()
    raise ValueError(variant)

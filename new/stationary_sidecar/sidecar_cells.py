"""Exactly nested StreamingCell with persistent or instantaneous local side state."""
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/streaming_carry'))
from stream_cells import StreamingCell, stream

VARIANTS = ('stream', 'memory', 'stateless')
PARAMETERS = {'stream': 5033, 'memory': 7989, 'stateless': 7989}
CORE_NAMES = ('encoder.weight', 'encoder.bias', 'f_in.weight', 'f_in.bias',
              'f_out.weight', 'f_out.bias', 'q_in.weight', 'q_in.bias',
              'q_out.weight', 'q_out.bias', 'readout.weight', 'readout.bias')


class SidecarCell(StreamingCell):
    """Keep the original W24/Z8 core; add H12 without transporting H.

    Both variants consume the same current H* in F/Q. Only the persistent
    variant retains the previous H in H*. G deliberately does not read H
    directly: every G input parameter is active in both variants.
    """

    def __init__(self, persistent=True):
        # Original modules are constructed first, preserving their RNG draws.
        super().__init__()
        self.persistent = persistent
        self.h_channels = 12
        self.g_in = nn.Conv2d(67, 32, 1)
        self.g_out = nn.Conv2d(32, self.h_channels, 1)
        self.feedback_f = nn.Conv2d(self.h_channels, 24, 1, bias=False)
        self.feedback_q = nn.Conv2d(self.h_channels, 8, 1, bias=False)
        nn.init.normal_(self.g_out.weight, std=.02)
        nn.init.zeros_(self.g_out.bias)
        nn.init.zeros_(self.feedback_f.weight)
        nn.init.zeros_(self.feedback_q.weight)

    def initial(self, x):
        w, z = super().initial(x)
        h = w.new_zeros((w.shape[0], self.h_channels, *w.shape[-2:]))
        return w, z, h

    def step(self, state, x):
        self._validate_x(x)
        w, z, h = state
        old_features = self._features(w, z, x)
        write = .1 * self.g_out(torch.tanh(self.g_in(old_features)))
        active_h = h + write if self.persistent else write
        incoming = stream(w, x[:, :1])
        features = torch.cat((incoming, old_features[:, 24:]), dim=1)
        injection_f = self.feedback_f(active_h)
        injection_q = self.feedback_q(active_h)
        force = (self.f_out(torch.tanh(self.f_in(features))) +
                 injection_f)
        w_new = incoming + self.eta * force
        q = self._candidate(w_new, z, x) + injection_q
        z_new = z + self.alpha * q
        stored_h = active_h if self.persistent else torch.zeros_like(h)
        # Observational tensors only: no buffers, state feedback, or graph retention.
        self.last_active_h = active_h.detach()
        self.last_injection_f = injection_f.detach()
        self.last_injection_q = injection_q.detach()
        return w_new, z_new, stored_h

    def logits(self, state):
        return self.readout(state[1])

    def metadata(self):
        return {**super().metadata(), 'architecture': 'stationary_sidecar_v1',
                'persistent_sidecar': self.persistent, 'sidecar_channels': 12,
                'stationary_state': 'Z8 and H12 (H reset each step for stateless)',
                'sidecar_write': '.1*G(W,Z,L_M(W),L_M(Z),X); G67->32->12',
                'sidecar_update': 'H*=H+write' if self.persistent else 'H*=write; stored H=0',
                'workspace_update': 'W_new=T_M(W)+.1*(F(T_M(W),Z,L_M(W),L_M(Z),X)+P_F(H*))',
                'latent_update': 'Z_new=Z+.5*(Q(W_new,Z,L_M(W_new),L_M(Z),X)+P_Q(H*))',
                'sidecar_initialization': 'H0=0; G_in default; G_out weight N(0,.02), bias0; P_F/P_Q weight0, no bias',
                'neutral_point': 'P_F=P_Q=0: projected W/Z/logits match StreamingCell',
                'neutral_optimizer_update_equivalence': False,
                'macro_graph_radius_upper_bound': 2,
                'K8_graph_radius_upper_bound': 16,
                'parameter_count': sum(p.numel() for p in self.parameters())}


def core_state_dict(model):
    state = model.state_dict()
    return {name: state[name] for name in CORE_NAMES}


def make_variant(variant):
    if variant == 'stream':
        return StreamingCell()
    if variant == 'memory':
        return SidecarCell(persistent=True)
    if variant == 'stateless':
        return SidecarCell(persistent=False)
    raise ValueError(variant)

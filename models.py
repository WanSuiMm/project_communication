"""Small target-readout models for the ReactionTransport qualification."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .transport import apply_transport, prepare_transport


def _conv(dim: int):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]


def _channel_norm(h: Tensor) -> Tensor:
    """Layer-normalize channels independently at each spatial location."""
    return F.layer_norm(h.movedim(1, -1), (h.shape[1],)).movedim(-1, 1)


def _coordinates(x: Tensor) -> Tensor:
    axes = [
        torch.linspace(-1.0, 1.0, n, device=x.device, dtype=x.dtype)
        for n in x.shape[2:]
    ]
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grid, dim=0).unsqueeze(0).expand(x.shape[0], -1, *x.shape[2:])


def _terminal_mask(spatial: Sequence[int], axis: int, *, device, dtype) -> Tensor:
    shape = [1, 1, *spatial]
    idx = torch.arange(spatial[axis], device=device)
    mask = (idx < spatial[axis] - 1).to(dtype=dtype)
    view = [1] * (len(spatial) + 2)
    view[axis + 2] = spatial[axis]
    return mask.reshape(view).expand(shape)


class _EdgeBuilder(nn.Module):
    """Store symmetric undirected edges once, in positive-axis slots."""

    def __init__(self, dim: int, width: int, groups: int):
        super().__init__()
        conv = _conv(dim)
        self.features = conv(2 * width, 8, kernel_size=1)
        self.edge = conv(16, groups, kernel_size=1)

    def forward(self, e: Tensor, s: Tensor) -> List[Tensor]:
        f = self.features(torch.cat((e, s), dim=1))
        edges: List[Tensor] = []
        spatial = f.shape[2:]
        for axis, length in enumerate(spatial):
            if length <= 1:
                edges.append(f.new_zeros((f.shape[0], self.edge.out_channels, *spatial)))
                continue
            left_slice = [slice(None)] * f.ndim
            right_slice = [slice(None)] * f.ndim
            left_slice[axis + 2] = slice(0, max(length - 1, 0))
            right_slice[axis + 2] = slice(1, length)
            left = f[tuple(left_slice)]
            right = f[tuple(right_slice)]
            pair = torch.cat((left + right, torch.abs(left - right)), dim=1)
            logits = self.edge(pair).sigmoid()
            pad_shape = list(logits.shape)
            pad_shape[axis + 2] = 1
            terminal = logits.new_zeros(pad_shape)
            edges.append(torch.cat((logits, terminal), dim=axis + 2))
        return edges


class _Reaction(nn.Module):
    def __init__(self, dim: int, width: int, message_width: int, *, full_state: bool):
        super().__init__()
        conv = _conv(dim)
        self.width = width
        self.dim = dim
        self.full_state = full_state
        self.emit = nn.Identity() if full_state else conv(width, message_width, kernel_size=1)
        self.local = conv(width, width, kernel_size=3, padding=1, groups=width)
        self.delta = nn.Sequential(
            conv(3 * width + 2 * message_width, width, kernel_size=1),
            nn.SiLU(),
            conv(width, width, kernel_size=1),
        )
        self.gate = conv(2 * width + message_width, width, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, width, *([1] * dim)), 0.1))

    def forward(self, h: Tensor, e: Tensor, m: Tensor, q: Tensor | None = None):
        s = _channel_norm(h)
        if q is None:
            q = self.emit(h if self.full_state else s)
        local = self.local(s)
        z = torch.cat((s, local, e, m, m - q), dim=1)
        delta = self.delta(z)
        gate = self.gate(torch.cat((s, e, m), dim=1)).sigmoid()
        return q, self.gamma * gate * delta


class _GlobalMessage(nn.Module):
    """Global, multi-head scaled dot-product attention over spatial positions."""

    def __init__(self, dim: int, width: int, message_width: int, groups: int):
        super().__init__()
        conv = _conv(dim)
        self.message_width = message_width
        self.groups = groups
        self.head_width = message_width // groups
        self.qkv = conv(width, 3 * message_width, kernel_size=1)
        self.out = conv(message_width, message_width, kernel_size=1)

    def forward(self, s: Tensor) -> Tensor:
        b, _, *spatial = s.shape
        n = math.prod(spatial)
        q, k, v = self.qkv(s).reshape(
            b, 3, self.groups, self.head_width, n
        ).unbind(dim=1)
        q, k, v = (t.transpose(-2, -1) for t in (q, k, v))  # [B,G,N,D]
        m = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        m = m.transpose(-2, -1).reshape(b, self.message_width, *spatial)
        return self.out(m)


class GateModel(nn.Module):
    """A target-only classifier with reaction and optional message transport.

    ``message_full`` and ``state_full`` use width-sized transported states;
    the other variants use a ``message_width`` bottleneck.  ``state_full``
    uses the transported message as the residual base, isolating full-width
    state retention from the default message-residual model.
    """

    VARIANTS = {
        "cnn", "nca", "vit", "constant", "learned", "oracle",
        "message_full", "state_full"
    }

    def __init__(
        self,
        dim: int,
        variant: str,
        width: int = 64,
        message_width: int = 32,
        groups: int = 4,
        steps: int = 8,
        classes: int = 2,
    ):
        super().__init__()
        variant = variant.lower()
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(self.VARIANTS)}")
        if width < 1 or message_width < 1 or groups < 1 or classes < 1:
            raise ValueError("width, message_width, groups, and classes must be positive")
        if steps < 2 or steps % 2:
            raise ValueError("steps must be an even integer of at least two")

        full_state = variant in {"message_full", "state_full"}
        r = width if full_state else message_width
        if variant == "vit" or variant in {
            "constant", "learned", "oracle", "message_full", "state_full"
        }:
            if r % groups:
                raise ValueError("message channels must be divisible by groups")
        if dim == 1 and variant in {
            "constant", "learned", "oracle", "message_full", "state_full"
        }:
            raise ValueError("transport variants require dim=2 or dim=3")

        self.dim = dim
        self.variant = variant
        self.width = width
        self.message_width = r
        self.groups = groups
        self.steps = steps
        conv = _conv(dim)
        self.stem = conv(5 + dim, width, kernel_size=1)
        self.init_state = conv(width, width, kernel_size=1)

        n_cells = steps if variant == "cnn" else 2
        self.cells = nn.ModuleList(
            _Reaction(dim, width, r, full_state=full_state)
            for _ in range(n_cells)
        )
        if variant == "vit":
            self.global_messages = nn.ModuleList(
                _GlobalMessage(dim, width, r, groups) for _ in range(2)
            )
        else:
            self.global_messages = nn.ModuleList()

        if variant in {"learned", "message_full", "state_full"}:
            self.edge_builders = nn.ModuleList(
                _EdgeBuilder(dim, width, groups) for _ in range(2)
            )
        else:
            self.edge_builders = nn.ModuleList()

        if variant in {"constant", "learned", "oracle", "message_full", "state_full"}:
            initial_tau = torch.logspace(0.0, math.log10(4096.0), groups)
            self.log_tau = nn.Parameter(initial_tau.log().repeat(2, 1))
        else:
            self.register_parameter("log_tau", None)
        self.readout = nn.Linear(width, classes)

    def _transport_edges(self, phase: int, e: Tensor, h: Tensor) -> List[Tensor]:
        spatial = h.shape[2:]
        if self.variant in {"learned", "message_full", "state_full"}:
            return self.edge_builders[phase](e, _channel_norm(h))
        edges = []
        for axis in range(self.dim):
            edge = h.new_ones((h.shape[0], self.groups, *spatial))
            edges.append(edge * _terminal_mask(spatial, axis, device=h.device, dtype=h.dtype))
        return edges

    def _oracle_phase_edges(self, oracle_edges, h: Tensor) -> List[Tensor]:
        if oracle_edges is None:
            raise ValueError("variant='oracle' requires oracle_edges")
        # The same oracle edge list is reused for both phases.
        if not isinstance(oracle_edges, (list, tuple)) or len(oracle_edges) != self.dim:
            raise ValueError("oracle_edges must provide one tensor per spatial axis")
        spatial = h.shape[2:]
        result = []
        for axis, edge in enumerate(oracle_edges):
            if not isinstance(edge, Tensor) or edge.ndim != self.dim + 2:
                raise ValueError("each oracle edge must have shape [B,1,*spatial]")
            if edge.shape[0] != h.shape[0] or edge.shape[1] != 1 or tuple(edge.shape[2:]) != tuple(spatial):
                raise ValueError("each oracle edge must have shape [B,1,*spatial]")
            edge = edge.to(device=h.device, dtype=h.dtype)
            edge = edge.expand(-1, self.groups, *spatial)
            result.append(edge * _terminal_mask(spatial, axis, device=h.device, dtype=h.dtype))
        return result

    def forward(self, x: Tensor, target_index: Tensor, oracle_edges=None) -> Tensor:
        if x.ndim != self.dim + 2 or x.shape[1] != 5:
            raise ValueError(f"x must have shape [B,5,*spatial] for dim={self.dim}")
        if target_index.ndim != 1 or target_index.shape[0] != x.shape[0]:
            raise ValueError("target_index must have shape [B]")

        e = self.stem(torch.cat((x, _coordinates(x)), dim=1))
        h = self.init_state(e)
        spatial_size = math.prod(x.shape[2:])
        index = target_index.to(device=x.device, dtype=torch.long)
        if torch.any(index < 0) or torch.any(index >= spatial_size):
            raise ValueError("target_index is outside the flattened spatial range")

        steps = self.steps
        if steps < 2 or steps % 2:
            raise ValueError("self.steps must remain an even integer of at least two")
        phase_steps = steps // 2
        if self.variant == "cnn" and steps > len(self.cells):
            raise ValueError("CNN step count cannot exceed its unshared cell count")

        if oracle_edges is not None and self.variant != "oracle":
            raise ValueError("oracle_edges are accepted only by variant='oracle'")
        for phase in range(2):
            prepared = None
            if self.variant in {"constant", "learned", "message_full", "state_full"}:
                edges = self._transport_edges(phase, e, h)
                tau = self.log_tau[phase].exp()
                prepared = prepare_transport(edges, tau)
            elif self.variant == "oracle":
                edges = self._oracle_phase_edges(oracle_edges, h)
                tau = self.log_tau[phase].exp()
                prepared = prepare_transport(edges, tau)
            first_step = phase * phase_steps
            for local_step in range(phase_steps):
                cell_index = first_step + local_step if self.variant == "cnn" else phase
                cell = self.cells[cell_index]
                s = _channel_norm(h)
                q = cell.emit(h if cell.full_state else s)
                if prepared is not None:
                    m = apply_transport(q, prepared)
                elif self.variant == "vit":
                    m = self.global_messages[phase](s)
                else:
                    m = q
                _, update = cell(h, e, m, q=q)
                base = m if self.variant == "state_full" else h
                h = base + update

        target = h.flatten(start_dim=2).gather(
            2, index[:, None, None].expand(-1, self.width, 1)
        ).squeeze(-1)
        return self.readout(target)

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


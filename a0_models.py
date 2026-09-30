"""A0 models: paired confidence-weighted emission, with/without normalization.

The v1 model and solver are deliberately imported without modification. All
A0 arms start with the same common parameters for a seed. This is a new
architecture screen, not a rerun or replacement of the frozen v1 experiment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import GateModel, _EdgeBuilder, _channel_norm, _conv, _coordinates
from .transport import apply_transport, prepare_transport


VARIANTS = ("attention", "constant_raw", "constant_normalized",
            "learned_raw", "learned_normalized")


def transport_pair(q, confidence, prepared):
    """Transport q and c in the SAME group-specific kernel and factorization."""
    b, r, *shape = q.shape
    groups = confidence.shape[1]
    per_group = r // groups
    packed = torch.cat((q.reshape(b, groups, per_group, *shape),
                        confidence.unsqueeze(2)), dim=2)
    packed = apply_transport(packed.flatten(1, 2), prepared)
    packed = packed.reshape(b, groups, per_group + 1, *shape)
    return packed[:, :, :per_group].flatten(1, 2), packed[:, :, per_group]


class SharedValueAttention(nn.Module):
    """Only query/key are projected here; values are the supplied emitted q."""
    def __init__(self, dim, width, message_width, groups):
        super().__init__()
        self.groups = groups
        self.head_width = message_width // groups
        self.query_key = _conv(dim)(width, 2 * message_width, 1)

    def forward(self, state, emitted):
        b, r, *shape = emitted.shape
        n = math.prod(shape)
        query, key = self.query_key(state).reshape(
            b, 2, self.groups, self.head_width, n).unbind(1)
        value = emitted.reshape(b, self.groups, self.head_width, n)
        output = F.scaled_dot_product_attention(
            query.transpose(-1, -2), key.transpose(-1, -2),
            value.transpose(-1, -2), dropout_p=0.0)
        return output.transpose(-1, -2).reshape(b, r, *shape)


class A0Model(GateModel):
    def __init__(self, dim, variant, width=64, message_width=32, groups=4,
                 steps=8, epsilon=1e-6):
        if variant not in VARIANTS:
            raise ValueError(variant)
        # Initial common parameters (including readout) do not depend on arm.
        super().__init__(dim, "constant", width, message_width, groups, steps)
        self.variant = variant
        self.epsilon = epsilon
        self.confidence = nn.ModuleList(
            _conv(dim)(width, groups, 1) for _ in range(2))
        for layer in self.confidence:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)  # c=0.5 initially, in every arm.
        if variant.startswith("learned"):
            self.edge_builders = nn.ModuleList(
                _EdgeBuilder(dim, width, groups) for _ in range(2))
            for builder in self.edge_builders:
                nn.init.zeros_(builder.edge.weight)
                nn.init.zeros_(builder.edge.bias)
        if variant == "attention":
            self.global_messages = nn.ModuleList(
                SharedValueAttention(dim, width, message_width, groups)
                for _ in range(2))
            self.register_parameter("log_tau", None)

    def forward(self, x, target_index, oracle_edges=None, diagnostics=False):
        if oracle_edges is not None:
            raise ValueError("A0 learned models do not accept oracle geometry")
        if x.ndim != self.dim + 2 or x.shape[1] != 5:
            raise ValueError("Expected [B,5,*spatial]")
        e = self.stem(torch.cat((x, _coordinates(x)), 1))
        h = self.init_state(e)
        index = target_index.to(device=x.device, dtype=torch.long)
        if index.shape != (x.shape[0],) or torch.any(index < 0) or torch.any(index >= math.prod(x.shape[2:])):
            raise ValueError("Invalid target index")
        stats = {}

        def gather(field):
            return field.flatten(2).gather(2, index[:, None, None].expand(-1, field.shape[1], 1)).squeeze(-1)

        for phase in range(2):
            cell = self.cells[phase]
            prepared = None
            if self.variant != "attention":
                if self.variant.startswith("learned"):
                    # 2*sigmoid(0)=1: exactly match the constant medium at init.
                    edges = [2 * edge for edge in self.edge_builders[phase](e, _channel_norm(h))]
                else:
                    edges = self._transport_edges(phase, e, h)
                # Frozen v1 axis order shared by every RT arm. No equivariance claim.
                prepared = prepare_transport(edges, self.log_tau[phase].exp())
            for step in range(self.steps // 2):
                state = _channel_norm(h)
                value = cell.emit(state)
                confidence = self.confidence[phase](state).sigmoid()
                expanded = confidence.repeat_interleave(self.message_width // self.groups, 1)
                q = expanded * value
                if prepared is None:
                    message = self.global_messages[phase](state, q)
                else:
                    numerator, denominator = transport_pair(q, confidence, prepared)
                    message = numerator
                    if self.variant.endswith("normalized"):
                        message = numerator / (denominator.repeat_interleave(
                            self.message_width // self.groups, 1) + self.epsilon)
                _, update = cell(h, e, message, q=q)
                h = h + update
                if diagnostics and phase == 1 and step == self.steps // 2 - 1:
                    source_mask = x[:, 1:2]
                    source_count = source_mask.sum() * self.groups
                    background_count = (1 - source_mask).sum() * self.groups
                    stats = {
                        "source_confidence_mean": float((confidence * source_mask).sum() / source_count),
                        "background_confidence_mean": float((confidence * (1 - source_mask)).sum() / background_count),
                        "target_message_rms": float(gather(message).square().mean().sqrt()),
                        "target_emission_rms": float(gather(q).square().mean().sqrt()),
                    }
                    if prepared is not None:
                        target_d = gather(denominator)
                        source_mass = apply_transport(confidence * source_mask, prepared)
                        fraction = gather(source_mass) / target_d.clamp_min(self.epsilon)
                        stats.update(target_denominator_min=float(target_d.min()),
                                     target_denominator_mean=float(target_d.mean()),
                                     target_denominator_p10=float(target_d.flatten().quantile(.1)),
                                     target_below_epsilon_fraction=float((target_d <= self.epsilon).float().mean()),
                                     target_source_mass_fraction_mean=float(fraction.mean()),
                                     tau=self.log_tau.detach().exp().cpu().tolist())
        logits = self.readout(gather(h))
        if not self.training and not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite A0 evaluation logits")
        return (logits, stats) if diagnostics else logits

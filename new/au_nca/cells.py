"""Original and accumulated-update NCA cells.

The AU state stores a visible base ``b`` and an update accumulator ``e``.  Its
visible state is reconstructed as ``b + W e``, where the last channel of ``e``
is a constant one so the projection can express a trainable output bias.
"""
from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Mode: TypeAlias = str
CellState: TypeAlias = Tensor | tuple[Tensor, Tensor]


class FixedPerception(nn.Module):
    """Depthwise identity and Sobel perceptions with zero-padded boundaries."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        # One grouped convolution computes both derivatives. The identity
        # perception is the input itself and needs no convolution launch.
        pair = torch.stack((sobel_x, sobel_y)).unsqueeze(1).repeat(channels, 1, 1, 1)
        self.register_buffer("sobel_pair", pair, persistent=True)
        self.channels = channels

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 4 or state.shape[1] != self.channels:
            raise ValueError(
                f"perception expects Bx{self.channels}xHxW, got {tuple(state.shape)}"
            )
        derivatives = F.conv2d(state, self.sobel_pair, padding=1, groups=self.channels)
        derivatives = derivatives.reshape(state.shape[0], self.channels, 2, *state.shape[-2:])
        return torch.cat((state, derivatives[:, :, 0], derivatives[:, :, 1]), dim=1)


class NCACell(nn.Module):
    """NCA cell with original-state and accumulated-update representations."""

    def __init__(self, channels: int = 16, hidden: int = 128, alpha_channel: int = 3) -> None:
        super().__init__()
        if channels <= 0 or hidden <= 0:
            raise ValueError("channels and hidden must be positive")
        if not 0 <= alpha_channel < channels:
            raise ValueError("alpha_channel must index a visible state channel")

        self.channels = int(channels)
        self.hidden = int(hidden)
        self.alpha_channel = int(alpha_channel)
        self.perception = FixedPerception(self.channels)
        self.feature = nn.Conv2d(3 * self.channels, self.hidden, kernel_size=1, bias=True)
        self.projection = nn.Conv2d(self.hidden + 1, self.channels, kernel_size=1, bias=False)
        self.reset_parameters()
        self.last_masks: dict[str, Tensor] | None = None

    def reset_parameters(self) -> None:
        # Kaiming initialization is suitable for the ReLU feature map.  The
        # accumulator projection starts at zero so both arms start identically.
        nn.init.kaiming_uniform_(self.feature.weight, nonlinearity="relu")
        nn.init.zeros_(self.feature.bias)
        nn.init.zeros_(self.projection.weight)

    @property
    def state_channels(self) -> dict[str, int]:
        return {"original": self.channels, "au_base": self.channels, "au_accumulator": self.hidden + 1}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_mode(self, mode: Mode) -> None:
        if mode not in ("original", "au"):
            raise ValueError(f"mode must be 'original' or 'au', got {mode!r}")

    def _validate_visible(self, seed_visible: Tensor) -> None:
        if seed_visible.ndim != 4 or seed_visible.shape[1] != self.channels:
            raise ValueError(
                f"seed_visible expects Bx{self.channels}xHxW, got {tuple(seed_visible.shape)}"
            )
        if not seed_visible.is_floating_point():
            raise TypeError("seed_visible must have a floating-point dtype")

    def perceive(self, state: Tensor) -> Tensor:
        """Return identity, x-Sobel, and y-Sobel channels in that order."""
        return self.perception(state)

    def initialize(self, seed_visible: Tensor, mode: Mode = "original") -> CellState:
        self._validate_mode(mode)
        self._validate_visible(seed_visible)
        base = seed_visible.clone()
        if mode == "original":
            return base
        batch, _, height, width = seed_visible.shape
        accumulator = seed_visible.new_zeros((batch, self.hidden + 1, height, width))
        return base, accumulator

    def visible(self, state: CellState, mode: Mode = "original") -> Tensor:
        self._validate_mode(mode)
        if mode == "original":
            if not isinstance(state, Tensor):
                raise TypeError("original mode state must be a Tensor")
            self._validate_visible(state)
            return state
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("au mode state must be a (base, accumulator) tuple")
        base, accumulator = state
        self._validate_visible(base)
        if accumulator.ndim != 4 or accumulator.shape[1] != self.hidden + 1:
            raise ValueError(
                f"AU accumulator expects Bx{self.hidden + 1}xHxW, got {tuple(accumulator.shape)}"
            )
        if base.shape[0] != accumulator.shape[0] or base.shape[-2:] != accumulator.shape[-2:]:
            raise ValueError("AU base and accumulator batch/spatial shapes must match")
        return base + self.projection(accumulator)

    def detach(self, state: CellState, mode: Mode = "original") -> CellState:
        self._validate_mode(mode)
        if mode == "original":
            if not isinstance(state, Tensor):
                raise TypeError("original mode state must be a Tensor")
            return state.detach()
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("au mode state must be a (base, accumulator) tuple")
        return state[0].detach(), state[1].detach()

    def _alive(self, visible: Tensor) -> Tensor:
        alpha = visible[:, self.alpha_channel : self.alpha_channel + 1]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def step(
        self,
        state: CellState,
        fire: Tensor,
        mode: Mode = "original",
        alive_override: Tensor | None = None,
    ) -> CellState:
        """Apply one NCA update, optionally using a fixed qualification mask.

        ``alive_override`` is the final boolean mask applied to the candidate
        state.  When omitted, the usual intersection of pre- and post-update
        alive masks is used.
        """
        self._validate_mode(mode)
        visible_before = self.visible(state, mode)
        expected_fire_shape = (visible_before.shape[0], 1, *visible_before.shape[-2:])
        if tuple(fire.shape) != expected_fire_shape:
            raise ValueError(f"fire must have shape {expected_fire_shape}, got {tuple(fire.shape)}")
        fire = fire.to(device=visible_before.device, dtype=visible_before.dtype)
        pre_alive = self._alive(visible_before)

        features = F.relu(self.feature(self.perceive(visible_before)))
        expanded = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
        if mode == "original":
            if not isinstance(state, Tensor):  # kept explicit for type checkers and callers
                raise TypeError("original mode state must be a Tensor")
            candidate = state + fire * self.projection(expanded)
            base_candidate = None
            accumulator_candidate = None
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise TypeError("au mode state must be a (base, accumulator) tuple")
            base, accumulator = state
            accumulator_candidate = accumulator + fire * expanded
            base_candidate = base
            candidate = base + self.projection(accumulator_candidate)

        post_alive = self._alive(candidate)
        computed_alive = pre_alive & post_alive
        if alive_override is None:
            alive = computed_alive
        else:
            if tuple(alive_override.shape) != tuple(computed_alive.shape):
                raise ValueError(
                    f"alive_override must have shape {tuple(computed_alive.shape)}, "
                    f"got {tuple(alive_override.shape)}"
                )
            alive = alive_override.to(device=candidate.device).bool()
        mask = alive.to(dtype=candidate.dtype)
        self.last_masks = {
            "pre_alive": pre_alive.detach(),
            "post_alive": post_alive.detach(),
            "computed_alive": computed_alive.detach(),
            "alive": alive.detach(),
        }

        if mode == "original":
            return candidate * mask
        assert base_candidate is not None and accumulator_candidate is not None
        return base_candidate * mask, accumulator_candidate * mask


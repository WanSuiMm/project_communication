"""Conditioned NCA cells for coarse-logit retinal refinement.

This is a conditioned NCA adaptation: RGB is a fixed per-rollout input, while
the coarse probability map is used only to initialize the working state.  The
cell does not reproduce the original Reaction--Transport architecture.
"""
from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Mode: TypeAlias = str
CellState: TypeAlias = Tensor | tuple[Tensor, Tensor]


class ConditionalNCACell(nn.Module):
    """Shared original/AU cell with a fixed FOV and static RGB conditioning.

    Channel zero of the visible state is the segmentation logit.  The other
    channels are learned working state.  In AU mode, ``state`` is ``(b, e)``
    and its visible value is ``b + projection(e)``; the 129th accumulator
    channel accumulates the appended per-update constant used by the shared
    bias-free projection.
    """

    def __init__(
        self,
        channels: int = 16,
        hidden: int = 128,
        perception_width: int = 64,
        clamp_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden <= 0 or perception_width <= 0:
            raise ValueError("channels, hidden, and perception_width must be positive")
        if not 0.0 < clamp_epsilon < 0.5:
            raise ValueError("clamp_epsilon must be between 0 and 0.5")

        self.channels = int(channels)
        self.hidden = int(hidden)
        self.perception_width = int(perception_width)
        self.clamp_epsilon = float(clamp_epsilon)

        self.state_perception = nn.Conv2d(
            self.channels, self.perception_width, kernel_size=3, padding=1
        )
        self.image_perception = nn.Conv2d(3, self.perception_width, kernel_size=3, padding=1)
        self.fuse = nn.Conv2d(2 * self.perception_width, self.hidden, kernel_size=1)
        self.projection = nn.Conv2d(self.hidden + 1, self.channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.projection.weight)

    @property
    def state_channels(self) -> dict[str, int]:
        return {
            "original": self.channels,
            "au_base": self.channels,
            "au_accumulator": self.hidden + 1,
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_mode(self, mode: Mode) -> None:
        if mode not in ("original", "au"):
            raise ValueError(f"mode must be 'original' or 'au', got {mode!r}")

    def _validate_visible(self, visible: Tensor, name: str = "visible") -> None:
        if visible.ndim != 4 or visible.shape[1] != self.channels:
            raise ValueError(
                f"{name} expects Bx{self.channels}xHxW, got {tuple(visible.shape)}"
            )
        if not visible.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")

    @staticmethod
    def _validate_mask(mask: Tensor, expected: tuple[int, int, int, int], name: str) -> None:
        if mask.ndim != 4 or tuple(mask.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(mask.shape)}")

    def encode_image(self, rgb: Tensor) -> Tensor:
        """Encode the static RGB image once for reuse across recurrent steps."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb expects Bx3xHxW, got {tuple(rgb.shape)}")
        if not rgb.is_floating_point():
            raise TypeError("rgb must have a floating-point dtype")
        return self.image_perception(rgb)

    def initialize(
        self, probability: Tensor, fov: Tensor, mode: Mode = "original"
    ) -> CellState:
        """Initialize channel zero from coarse probabilities and mask the FOV."""
        self._validate_mode(mode)
        if probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError(
                f"probability expects Bx1xHxW, got {tuple(probability.shape)}"
            )
        if not probability.is_floating_point():
            raise TypeError("probability must have a floating-point dtype")
        expected = (probability.shape[0], 1, *probability.shape[-2:])
        self._validate_mask(fov, expected, "fov")

        logits = torch.logit(
            probability.clamp(min=self.clamp_epsilon, max=1.0 - self.clamp_epsilon)
        )
        visible = probability.new_zeros(
            probability.shape[0], self.channels, *probability.shape[-2:]
        )
        visible[:, :1] = logits
        mask = fov.to(device=probability.device, dtype=probability.dtype)
        visible = visible * mask
        if mode == "original":
            return visible
        accumulator = probability.new_zeros(
            probability.shape[0], self.hidden + 1, *probability.shape[-2:]
        )
        return visible, accumulator

    def visible(self, state: CellState, mode: Mode = "original") -> Tensor:
        """Return the visible working state for either representation."""
        self._validate_mode(mode)
        if mode == "original":
            if not isinstance(state, Tensor):
                raise TypeError("original mode state must be a Tensor")
            self._validate_visible(state, "state")
            return state
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("au mode state must be a (base, accumulator) tuple")
        base, accumulator = state
        self._validate_visible(base, "AU base")
        expected = (base.shape[0], self.hidden + 1, *base.shape[-2:])
        if accumulator.ndim != 4 or tuple(accumulator.shape) != expected:
            raise ValueError(
                f"AU accumulator expects {expected}, got {tuple(accumulator.shape)}"
            )
        return base + self.projection(accumulator)

    def detach(self, state: CellState, mode: Mode = "original") -> CellState:
        """Detach recurrent state while preserving the complete AU accumulator."""
        self._validate_mode(mode)
        if mode == "original":
            if not isinstance(state, Tensor):
                raise TypeError("original mode state must be a Tensor")
            return state.detach()
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("au mode state must be a (base, accumulator) tuple")
        return state[0].detach(), state[1].detach()

    def canonicalize_visible(self, visible: Tensor, mode: Mode = "original") -> CellState:
        """Create a mode-specific state for a visible-state intervention.

        AU interventions start with a zero accumulator because only the supplied
        visible tensor is known.  Ordinary truncated rollouts should use
        :meth:`detach` so that their accumulated history is retained.
        """
        self._validate_mode(mode)
        self._validate_visible(visible)
        base = visible.clone()
        if mode == "original":
            return base
        accumulator = visible.new_zeros(
            visible.shape[0], self.hidden + 1, *visible.shape[-2:]
        )
        return base, accumulator

    def _augmented_update(self, visible: Tensor, image_features: Tensor) -> Tensor:
        self._validate_visible(visible)
        expected = (visible.shape[0], self.perception_width, *visible.shape[-2:])
        if image_features.ndim != 4 or tuple(image_features.shape) != expected:
            raise ValueError(
                f"image_features must have shape {expected}, got {tuple(image_features.shape)}"
            )
        state_features = self.state_perception(visible)
        hidden = F.relu(self.fuse(torch.cat((state_features, image_features), dim=1)))
        constant = torch.ones_like(hidden[:, :1])
        return torch.cat((hidden, constant), dim=1)

    def step(
        self,
        state: CellState,
        image_features: Tensor,
        fire: Tensor,
        fov: Tensor,
        mode: Mode = "original",
    ) -> CellState:
        """Apply one scalar-per-cell Bernoulli update under the fixed FOV."""
        self._validate_mode(mode)
        visible_before = self.visible(state, mode)
        expected_mask = (visible_before.shape[0], 1, *visible_before.shape[-2:])
        self._validate_mask(fire, expected_mask, "fire")
        self._validate_mask(fov, expected_mask, "fov")
        fire_value = fire.to(device=visible_before.device, dtype=visible_before.dtype)
        mask = fov.to(device=visible_before.device, dtype=visible_before.dtype)
        update = self._augmented_update(visible_before, image_features)

        if mode == "original":
            if not isinstance(state, Tensor):
                raise TypeError("original mode state must be a Tensor")
            return mask * (state + fire_value * self.projection(update))

        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("au mode state must be a (base, accumulator) tuple")
        base, accumulator = state
        next_base = mask * base
        next_accumulator = mask * (accumulator + fire_value * update)
        return next_base, next_accumulator


__all__ = ["CellState", "ConditionalNCACell", "Mode"]

"""Original, accumulated-update, and ReLU input-lift NCA cell modes.

The ``relu_lift`` mode keeps the visible base ``b`` and stores each hidden
unit's accumulated augmented perception in ``U``.  Its visible update is
decoded with the inherited feature weights and bias, so all three modes share
the exact same trainable parameters.
"""
from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor
from torch.nn import functional as F

from new.au_nca.cells import CellState, Mode, NCACell


ReLULiftState: TypeAlias = tuple[Tensor, Tensor, Tensor]
ExtendedCellState: TypeAlias = CellState | ReLULiftState


class ReLUInputLiftCell(NCACell):
    """NCA cell with a ReLU activation-conditioned input-lift state.

    ``U[b,j,k,y,x]`` accumulates the augmented perception ``p_aug`` for hidden
    feature ``j`` when that feature's pre-update affine activation is positive.
    The third state tensor accumulates the fire mask.  In the default
    16-channel, 128-hidden configuration this mode carries 6,289 floats per
    spatial cell: 16 base values, 128 * 49 lifted values, and one count.

    Under ``torch.no_grad()``, ``step`` updates the rollout-owned state tensors
    in place to avoid allocating another full-sized lifted state.  The
    differentiable path is entirely out of place.
    """

    @property
    def state_channels(self) -> dict[str, int]:
        augmented_features = 3 * self.channels + 1
        return {
            "original": self.channels,
            "au_base": self.channels,
            "au_accumulator": self.hidden + 1,
            "relu_lift_base": self.channels,
            "relu_lift_U": self.hidden * augmented_features,
            "relu_lift_count": 1,
        }

    def _validate_mode(self, mode: Mode) -> None:
        if mode not in ("original", "au", "relu_lift"):
            raise ValueError(
                f"mode must be 'original', 'au', or 'relu_lift', got {mode!r}"
            )

    def _validate_lift_state(self, state: ExtendedCellState) -> ReLULiftState:
        if not isinstance(state, tuple) or len(state) != 3:
            raise TypeError("relu_lift mode state must be a (base, U, count) tuple")
        base, update, count = state
        self._validate_visible(base)
        batch, _, height, width = base.shape
        augmented_features = 3 * self.channels + 1
        expected_update = (batch, self.hidden, augmented_features, height, width)
        if update.ndim != 5 or tuple(update.shape) != expected_update:
            raise ValueError(
                f"relu_lift U expects shape {expected_update}, got {tuple(update.shape)}"
            )
        expected_count = (batch, 1, height, width)
        if tuple(count.shape) != expected_count:
            raise ValueError(
                f"relu_lift count expects shape {expected_count}, got {tuple(count.shape)}"
            )
        return base, update, count

    def _augmented_feature_matrix(self) -> Tensor:
        """Return inherited feature weights and bias as ``[hidden, 3C+1]``."""
        weights = self.feature.weight[:, :, 0, 0]
        return torch.cat((weights, self.feature.bias[:, None]), dim=1)

    def _lifted_feature_sum(self, update: Tensor) -> Tensor:
        # U's adjacent (hidden,input) axes flatten as a view. Each group reads
        # one hidden unit's 49 input sums; this avoids einsum's large layout
        # conversion and keeps exactly the same row contraction and gradients.
        weights = self._augmented_feature_matrix()[:, :, None, None]
        return F.conv2d(update.flatten(1, 2), weights, groups=self.hidden)

    def initialize(self, seed_visible: Tensor, mode: Mode = "original") -> ExtendedCellState:
        self._validate_mode(mode)
        if mode != "relu_lift":
            return super().initialize(seed_visible, mode)
        self._validate_visible(seed_visible)
        batch, _, height, width = seed_visible.shape
        augmented_features = 3 * self.channels + 1
        base = seed_visible.clone()
        update = seed_visible.new_zeros(
            (batch, self.hidden, augmented_features, height, width)
        )
        count = seed_visible.new_zeros((batch, 1, height, width))
        return base, update, count

    def visible(self, state: ExtendedCellState, mode: Mode = "original") -> Tensor:
        self._validate_mode(mode)
        if mode != "relu_lift":
            return super().visible(state, mode)
        base, update, count = self._validate_lift_state(state)
        lifted_features = self._lifted_feature_sum(update)
        projected = self.projection(torch.cat((lifted_features, count), dim=1))
        return base + projected

    def detach(self, state: ExtendedCellState, mode: Mode = "original") -> ExtendedCellState:
        self._validate_mode(mode)
        if mode != "relu_lift":
            return super().detach(state, mode)
        base, update, count = self._validate_lift_state(state)
        return base.detach(), update.detach(), count.detach()

    def step(
        self,
        state: ExtendedCellState,
        fire: Tensor,
        mode: Mode = "original",
        alive_override: Tensor | None = None,
    ) -> ExtendedCellState:
        if mode != "relu_lift":
            return super().step(state, fire, mode, alive_override)
        self._validate_mode(mode)
        base, update, count = self._validate_lift_state(state)
        visible_before = self.visible(state, mode)
        expected_fire_shape = (visible_before.shape[0], 1, *visible_before.shape[-2:])
        if tuple(fire.shape) != expected_fire_shape:
            raise ValueError(f"fire must have shape {expected_fire_shape}, got {tuple(fire.shape)}")
        fire = fire.to(device=visible_before.device, dtype=visible_before.dtype)
        pre_alive = self._alive(visible_before)
        if alive_override is not None and tuple(alive_override.shape) != tuple(pre_alive.shape):
            raise ValueError(
                f"alive_override must have shape {tuple(pre_alive.shape)}, "
                f"got {tuple(alive_override.shape)}"
            )

        perceived = self.perceive(visible_before)
        firing = self.feature(perceived) > 0
        p_aug = torch.cat((perceived, torch.ones_like(perceived[:, :1])), dim=1)
        gated_fire = fire * firing.to(dtype=fire.dtype)

        if torch.is_grad_enabled():
            update_candidate = torch.addcmul(
                update,
                gated_fire.unsqueeze(2),
                p_aug.unsqueeze(1),
            )
            count_candidate = count + fire
            candidate_state: ReLULiftState = (base, update_candidate, count_candidate)
        else:
            # The rollout owns these tensors (initialize cloned the seed), so
            # the no-grad path can update them without copying a 128x49 state.
            update.addcmul_(gated_fire.unsqueeze(2), p_aug.unsqueeze(1))
            count.add_(fire)
            candidate_state = (base, update, count)

        candidate = self.visible(candidate_state, mode)
        post_alive = self._alive(candidate)
        computed_alive = pre_alive & post_alive
        if alive_override is None:
            alive = computed_alive
        else:
            alive = alive_override.to(device=candidate.device).bool()

        mask = alive.to(dtype=candidate.dtype)
        self.last_masks = {
            "pre_alive": pre_alive.detach(),
            "post_alive": post_alive.detach(),
            "computed_alive": computed_alive.detach(),
            "alive": alive.detach(),
        }

        if torch.is_grad_enabled():
            return (
                base * mask,
                update_candidate * mask.unsqueeze(2),
                count_candidate * mask,
            )

        base.mul_(mask)
        update.mul_(mask.unsqueeze(2))
        count.mul_(mask)
        return base, update, count


__all__ = ["ReLULiftState", "ExtendedCellState", "ReLUInputLiftCell"]

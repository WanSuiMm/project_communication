"""Matched native and input-factorized streaming cells, with an R2 arm.

The native core is constructed with the same ``StreamingCell`` implementation
used by ``new/streaming_carry/stream_cells.py``.  Factorized arms retain every
native tensor and add one identity-initialized 8x8 basis per F/Q input layer.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("native", "factorized", "native_r2", "factorized_r2")
DIRECTIONS = ("N", "E", "S", "W")
INPUT_CHANNELS = 3
C_CHANNELS = 8
Z_CHANNELS = 8
R_CHANNELS = 2
F_HIDDEN = 40
Q_HIDDEN = 16
ETA = 0.1
ALPHA = 0.5
BASE_FEATURE_CHANNELS = 2 * C_CHANNELS + 2 * Z_CHANNELS + INPUT_CHANNELS
R2_FEATURE_CHANNELS = BASE_FEATURE_CHANNELS + R_CHANNELS
R2_G_INPUT_CHANNELS = C_CHANNELS + R_CHANNELS + Z_CHANNELS + C_CHANNELS + Z_CHANNELS + INPUT_CHANNELS


_IMPORT_LOCK = threading.RLock()
_REFERENCE_STREAMING_MODULE: ModuleType | None = None


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module at {path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_streaming_reference() -> ModuleType:
    """Load the frozen streaming implementation without generic-name clashes.

    The older files import ``cells``, ``masked_cells`` and ``revision_cells``
    by generic names.  Temporarily bind those imports to private module objects,
    then restore the caller's module table and import path.
    """
    global _REFERENCE_STREAMING_MODULE
    with _IMPORT_LOCK:
        if _REFERENCE_STREAMING_MODULE is not None:
            return _REFERENCE_STREAMING_MODULE

        aliases = ("cells", "masked_cells", "revision_cells")
        missing = object()
        prior_modules = {name: sys.modules.get(name, missing) for name in aliases}
        prior_path = list(sys.path)
        try:
            frozen = _load_module(
                "reaction_transport_latent_factorial_frozen_cells",
                ROOT / "new/nca_inertial_wind_tunnel/cells.py",
            )
            sys.modules["cells"] = frozen
            masked = _load_module(
                "reaction_transport_latent_factorial_masked_cells",
                ROOT / "new/masked_medium/masked_cells.py",
            )
            sys.modules["masked_cells"] = masked
            revision = _load_module(
                "reaction_transport_latent_factorial_revision_cells",
                ROOT / "new/workspace_revision/revision_cells.py",
            )
            sys.modules["revision_cells"] = revision
            _REFERENCE_STREAMING_MODULE = _load_module(
                "reaction_transport_latent_factorial_stream_cells",
                ROOT / "new/streaming_carry/stream_cells.py",
            )
        finally:
            sys.path[:] = prior_path
            for name, prior in prior_modules.items():
                if prior is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior  # type: ignore[assignment]
        return _REFERENCE_STREAMING_MODULE


_StreamingCell = _load_streaming_reference().StreamingCell


class FactorizedInput(nn.Module):
    """1x1 convolution whose C and L(C) blocks are right-multiplied by U.

    The full native weight stays as the base tensor B.  The learned U acts on
    the already-transported C feature (or the already-computed L(C) feature),
    so it is not moved across the directional stream operator.
    """

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        if source.kernel_size != (1, 1) or source.in_channels < 24:
            raise ValueError("FactorizedInput expects a 1x1 layer with C and L(C) blocks")
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        if self.groups != 1:
            raise ValueError("FactorizedInput requires groups=1")
        self.padding_mode = source.padding_mode
        self.weight = nn.Parameter(source.weight.detach().clone())
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source.bias.detach().clone())
        self.U = nn.Parameter(torch.eye(C_CHANNELS, dtype=source.weight.dtype,
                                       device=source.weight.device))

    def effective_weight(self) -> Tensor:
        """Return the differentiable effective [out,in,1,1] convolution weight."""
        matrix = self.weight[..., 0, 0]
        blocks = (
            matrix[:, 0:8] @ self.U,
            matrix[:, 8:16],
            matrix[:, 16:24] @ self.U,
            matrix[:, 24:],
        )
        return torch.cat(blocks, dim=1)[:, :, None, None]

    def forward(self, features: Tensor) -> Tensor:
        weight = self.effective_weight()
        if self.padding_mode != "zeros":
            features = F.pad(features, self._reversed_padding_repeated_twice,
                             mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(features, weight, self.bias, self.stride, padding,
                        self.dilation, self.groups)


class LatentFactorialCell(_StreamingCell):
    """C8/Z8 streaming cell with optional identity-initialized R2 sidecar."""

    def __init__(self, arm: str) -> None:
        super().__init__(
            streaming=True,
            workspace_channels=C_CHANNELS,
            latent_channels=Z_CHANNELS,
            workspace_hidden=F_HIDDEN,
            candidate_hidden=Q_HIDDEN,
        )
        self.arm = arm
        self.has_r2 = arm.endswith("_r2")
        self.is_factorized = arm.startswith("factorized")
        self.initial_seed: int | None = None

    def initial(self, x: Tensor) -> tuple[Tensor, ...]:
        c, z = super().initial(x)
        if not self.has_r2:
            return c, z
        r = c.new_zeros((c.shape[0], R_CHANNELS, *c.shape[-2:]))
        return c, z, r

    def _unpack_state(self, state: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor | None]:
        expected = 3 if self.has_r2 else 2
        if len(state) != expected:
            raise ValueError(f"{self.arm} state must contain {expected} tensors")
        c, z = state[0], state[1]
        r = state[2] if self.has_r2 else None
        if c.ndim != 4 or c.shape[1] != C_CHANNELS:
            raise ValueError("C must have shape [batch, 8, height, width]")
        if z.shape != c.shape:
            raise ValueError("Z must have the same shape as C")
        if r is not None and r.shape != (c.shape[0], R_CHANNELS, *c.shape[-2:]):
            raise ValueError("R must have shape [batch, 2, height, width]")
        return c, z, r

    def step(self, state: Sequence[Tensor], x: Tensor) -> tuple[Tensor, ...]:
        self._validate_x(x)
        c, z, r = self._unpack_state(state)
        if x.shape[0] != c.shape[0] or x.shape[-2:] != c.shape[-2:]:
            raise ValueError("x and state batch/spatial dimensions must match")

        # The local perception is computed from the pre-stream C and Z.  Only
        # the leading C block of the F input is replaced by incoming T(C).
        old_features = self._features(c, z, x)
        incoming = _REFERENCE_STREAMING_MODULE.stream(c, x[:, :1])
        force_features = torch.cat((incoming, old_features[:, C_CHANNELS:]), dim=1)

        r_plus: Tensor | None = None
        if self.has_r2:
            assert r is not None
            lc = old_features[:, 2 * C_CHANNELS:3 * C_CHANNELS]
            lz = old_features[:, 3 * C_CHANNELS:4 * C_CHANNELS]
            g_features = torch.cat((c, r, z, lc, lz, x), dim=1)
            r_plus = r + self.eta * self.g_out(torch.tanh(self.g_in(g_features)))
            force_features = torch.cat((force_features, r_plus), dim=1)

        force = self.f_out(torch.tanh(self.f_in(force_features)))
        c_new = incoming + self.eta * force

        # Q sees the post-force C and its one-hop perception, while Z and L(Z)
        # remain on the current macro-step's stationary side.
        q_features = self._features(c_new, z, x)
        if r_plus is not None:
            q_features = torch.cat((q_features, r_plus), dim=1)
        q = self.q_out(torch.tanh(self.q_in(q_features)))
        z_new = z + self.alpha * q
        if r_plus is None:
            return c_new, z_new
        return c_new, z_new, r_plus

    def logits(self, state: Sequence[Tensor]) -> Tensor:
        if len(state) < 2:
            raise ValueError("state must contain C and Z")
        return self.readout(state[1])

    def rollout(
        self,
        x: Tensor,
        steps: int,
        state: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        self._validate_x(x)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        current = self.initial(x) if state is None else tuple(state)
        for _ in range(steps):
            current = self.step(current, x)
        return tuple(current)

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "arm": self.arm,
            "architecture": "latent_factorial_streaming_cell",
            "input_channels": INPUT_CHANNELS,
            "workspace_input_channels": R2_FEATURE_CHANNELS if self.has_r2 else BASE_FEATURE_CHANNELS,
            "candidate_input_channels": R2_FEATURE_CHANNELS if self.has_r2 else BASE_FEATURE_CHANNELS,
            "workspace_update": (
                "C_new = T(C) + eta * F(T(C),Z,L(C),L(Z),X,R_plus)"
                if self.has_r2 else
                "C_new = T(C) + eta * F(T(C),Z,L(C),L(Z),X)"
            ),
            "latent_update": "Z_new = Z + alpha * Q",
            "stationary_state": "Z and R" if self.has_r2 else "Z",
            "initialization_seeds": {
                "core_seed": _seed_value(self.initial_seed) if self.initial_seed is not None else None,
                "r2_extension_seed_offset": 104729 if self.has_r2 else None,
                "r2_extension_seed": (
                    _seed_value(self.initial_seed, 104729)
                    if self.has_r2 and self.initial_seed is not None else None
                ),
            },
            "geometry": {
                "spatial_dimensions": 2,
                "directions": list(DIRECTIONS),
                "mask": "x[:, :1]",
                "transport": "masked port permutation with blocked-link bounce-back",
                "carrier_lanes": 4,
                "payload_channels_per_lane": C_CHANNELS // 4,
            },
            "clocks": {
                "transport": "T(C) once per macro-step",
                "F": "L(C), L(Z) are pre-stream; T(C) is the leading F feature",
                "Q": "uses C_new and L(C_new); Z and L(Z) are current-step values",
                "R2": (
                    "R_plus is computed before F from C,R,Z,L(C),L(Z),X; "
                    "R_plus is then held as R for the next macro-step"
                    if self.has_r2 else None
                ),
            },
            "state_shapes": {
                "C": "[B, 8, H, W]",
                "Z": "[B, 8, H, W]",
                **({"R": "[B, 2, H, W]"} if self.has_r2 else {}),
            },
            "feature_shapes": {
                "F_input": R2_FEATURE_CHANNELS if self.has_r2 else BASE_FEATURE_CHANNELS,
                "Q_input": R2_FEATURE_CHANNELS if self.has_r2 else BASE_FEATURE_CHANNELS,
                "G_input": R2_G_INPUT_CHANNELS if self.has_r2 else None,
                "G_hidden": 16 if self.has_r2 else None,
                "G_output": R_CHANNELS if self.has_r2 else None,
            },
            "factorization": {
                "enabled": self.is_factorized,
                "input_blocks": {"C": [0, 8], "L(C)": [16, 24]},
                "effective_weight": "B_C @ U on C; B_L @ U on L(C)",
                "basis_parameters": ["f_in.U", "q_in.U"] if self.is_factorized else [],
                "basis_initialization": "U_F = U_Q = I_8" if self.is_factorized else None,
                "basis_applied_after_transport": True,
            },
            "r2_sidecar": {
                "enabled": self.has_r2,
                "update": "R_plus = R + 0.1 * G_out(tanh(G_in(C,R,Z,L(C),L(Z),X)))",
                "G_out_initialization": "zero weight and bias" if self.has_r2 else None,
                "transport": "none" if self.has_r2 else None,
                "readout_input": "Z only",
            },
            "parameter_shapes": {
                name: list(parameter.shape) for name, parameter in self.named_parameters()
            },
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "stability_guarantee": False,
        })
        return result


def _seed_value(seed: int, offset: int = 0) -> int:
    # Keep explicit CPU construction deterministic for all ordinary Python ints.
    return (int(seed) + offset) % (2**63 - 1)


def _extend_with_r2(model: LatentFactorialCell, seed: int) -> None:
    """Append two R columns and create G using a separate deterministic stream."""
    old_f, old_q = model.f_in, model.q_in
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed_value(seed, 104729))
        wide_f = nn.Conv2d(R2_FEATURE_CHANNELS, F_HIDDEN, kernel_size=1)
        wide_q = nn.Conv2d(R2_FEATURE_CHANNELS, Q_HIDDEN, kernel_size=1)
        g_in = nn.Conv2d(R2_G_INPUT_CHANNELS, 16, kernel_size=1)
        g_out = nn.Conv2d(16, R_CHANNELS, kernel_size=1)

    with torch.no_grad():
        wide_f.weight[:, :BASE_FEATURE_CHANNELS].copy_(old_f.weight)
        wide_f.bias.copy_(old_f.bias)
        wide_q.weight[:, :BASE_FEATURE_CHANNELS].copy_(old_q.weight)
        wide_q.bias.copy_(old_q.bias)
        nn.init.zeros_(g_out.weight)
        nn.init.zeros_(g_out.bias)
    model.f_in = wide_f
    model.q_in = wide_q
    model.g_in = g_in
    model.g_out = g_out


def _factorize_input_layers(model: LatentFactorialCell) -> None:
    model.f_in = FactorizedInput(model.f_in)
    model.q_in = FactorizedInput(model.q_in)


def make_model(arm: str, seed: int) -> LatentFactorialCell:
    """Construct an arm on CPU from an exactly matched native C8/Z8 core."""
    if arm not in ARMS:
        raise ValueError(f"Unknown latent-factorial arm: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed_value(seed))
        model = LatentFactorialCell(arm)
    model.initial_seed = int(seed)
    model.cpu()
    if model.has_r2:
        _extend_with_r2(model, seed)
    if model.is_factorized:
        _factorize_input_layers(model)
    return model


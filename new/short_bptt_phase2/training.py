"""Fixed loss cadence with independently selectable temporal cut cadence."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/workspace_revision'))
from revision_cells import make_cell
from tasks import balanced_loss


def backward_trajectory(model, data, gradient_horizon, steps=64, loss_every=8,
                        budget=lambda: None):
    """Accumulate all gradients; caller clips and updates once after return."""
    if not (0 < loss_every <= gradient_horizon <= steps):
        raise ValueError('Invalid horizons')
    if gradient_horizon % loss_every or steps % gradient_horizon:
        raise ValueError('Windows must partition the fixed loss clock')
    state = model.initial(data['x'])
    pending = []
    total = state[0].new_zeros(())
    backwards = cuts = 0
    for t in range(1, steps + 1):
        if (t - 1) % loss_every == 0:
            budget()
        state = model.step(state, data['x'])
        if t % loss_every == 0:
            loss = balanced_loss(model.logits(state), data['y'], data['mask']) / (steps // loss_every)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite trajectory loss')
            pending.append(loss)
            total = total + loss.detach()
        if t % gradient_horizon == 0:
            torch.stack(pending).sum().backward()
            backwards += 1
            pending.clear()
            state = tuple(v.detach() for v in state)
            cuts += int(t < steps)
    return total, state, {'backward_calls': backwards, 'interior_detach_boundaries': cuts,
                          'forward_steps': steps, 'loss_count': steps // loss_every}

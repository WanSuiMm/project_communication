"""Matched forward trajectories with full or detached temporal gradients."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"new/workspace_revision"))
from revision_cells import make_cell
from tasks import balanced_loss


def backward_trajectory(model, data, gradient_horizon, steps=64, loss_every=8,
                        budget=lambda:None):
    """Accumulate gradients only; caller owns zero_grad/clip/optimizer.step.

    Parameters remain unchanged across the entire trajectory. In the K8
    condition detach removes history, never the numerical persistent state.
    """
    if gradient_horizon not in (loss_every,steps) or steps%loss_every:
        raise ValueError("Use one-loss window or full trajectory")
    state=model.initial(data["x"])
    loss_terms=[]; total=state[0].new_zeros(()); backward_calls=0; cuts=0
    count=steps//loss_every
    for t in range(1,steps+1):
        if (t-1)%loss_every==0:budget()
        state=model.step(state,data["x"])
        if t%loss_every==0:
            loss=balanced_loss(model.logits(state),data["y"],data["mask"])/count
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite trajectory loss")
            total=total+loss.detach()
            if gradient_horizon==steps:loss_terms.append(loss)
            else:
                loss.backward();backward_calls+=1
                state=tuple(v.detach() for v in state)
                if t<steps:cuts+=1
    if gradient_horizon==steps:
        torch.stack(loss_terms).sum().backward();backward_calls+=1
    return total,tuple(v.detach() for v in state),{"backward_calls":backward_calls,
        "interior_detach_boundaries":cuts,"forward_steps":steps,"loss_count":count}

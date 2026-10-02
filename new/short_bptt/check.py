"""Small CPU checks for the matched BPTT experiment."""
import copy
import json
import torch
from training import backward_trajectory, make_cell
from tasks import bank, balanced_loss


def main():
    torch.set_num_threads(2);torch.manual_seed(801)
    data=bank(8,2,805)
    errors=[]
    for arm in ("ws_additive","ws_revision"):
        model=make_cell(arm)
        # Activate state coupling so detach checks are nontrivial, unlike
        # the zero-output-layer initialization used for actual training.
        with torch.no_grad():
            model.f_out.weight.normal_(std=.03)
            model.q_out.weight.normal_(std=.03)
        full,short,reference=[copy.deepcopy(model) for _ in range(3)]
        before={k:v.clone() for k,v in model.state_dict().items()}
        lf,sf,tf=backward_trajectory(full,data,8,steps=8,loss_every=2)
        ls,ss,ts=backward_trajectory(short,data,2,steps=8,loss_every=2)
        for a,b in zip(sf,ss):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(lf,ls,rtol=1e-6,atol=1e-7)
        for m in (full,short):
            assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
        # Reference full graph: all losses, one reverse pass.
        state=reference.initial(data["x"]); losses=[]
        for t in range(1,9):
            state=reference.step(state,data["x"])
            if t%2==0:losses.append(balanced_loss(reference.logits(state),data["y"],data["mask"]))
        torch.stack(losses).mean().backward()
        err=max(float((a.grad-b.grad).abs().max()) for a,b in zip(full.parameters(),reference.parameters()))
        assert err<2e-7
        assert any(float((a.grad-b.grad).abs().max())>1e-5 for a,b in zip(full.parameters(),short.parameters()))
        assert tf["backward_calls"]==1 and ts["backward_calls"]==4 and ts["interior_detach_boundaries"]==3
        # Only a final-window loss: cutting both state blocks removes every
        # route to the encoder, while the local program still gets gradient.
        late=copy.deepcopy(model); s=late.initial(data["x"])
        for _ in range(2):s=late.step(s,data["x"])
        s=tuple(v.detach() for v in s)
        for _ in range(2):s=late.step(s,data["x"])
        balanced_loss(late.logits(s),data["y"],data["mask"]).backward()
        assert late.encoder.weight.grad is None
        assert float(late.q_out.weight.grad.norm())>0
        # Explicit parameter-gradient sum from four separately constructed
        # windows must equal the accumulated K2 result.
        separate=copy.deepcopy(model); sums=[torch.zeros_like(p) for p in separate.parameters()]
        s=separate.initial(data["x"])
        for _ in range(4):
            separate.zero_grad(set_to_none=True)
            for _ in range(2):s=separate.step(s,data["x"])
            (balanced_loss(separate.logits(s),data["y"],data["mask"])/4).backward()
            for acc,p in zip(sums,separate.parameters()):
                if p.grad is not None:acc.add_(p.grad)
            s=tuple(v.detach() for v in s)
        for acc,p in zip(sums,short.parameters()):torch.testing.assert_close(acc,p.grad,rtol=1e-5,atol=1e-7)
        errors.append({"arm":arm,"full_gradient_reference_error":err,
                       "forward_identical":True,"detached_encoder_blocked":True,"window_sum_matches":True})
    print(json.dumps({"status":"PASS","checks":errors},indent=2))


if __name__=="__main__":main()

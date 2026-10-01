"""Exact pairing, nontrivial mask isolation and finite first-order training."""
import unittest
import torch
from state_cells import make_cell, frozen_cells
from tasks import bank, balanced_loss


class Tests(unittest.TestCase):
    def test_pairing_and_all_open_equivalence(self):
        torch.manual_seed(0)
        original=frozen_cells.make_cell('nca_state_matched')
        next_rng=torch.rand(8)
        torch.manual_seed(0)
        model=make_cell()
        self.assertTrue(torch.equal(next_rng,torch.rand(8)))
        for k,v in original.state_dict().items():
            self.assertTrue(torch.equal(v,model.base.state_dict()[k]))
        meta=model.metadata()
        self.assertEqual((meta['content_channels'],meta['persistent_scalars_per_cell'],meta['hidden_width'],meta['parameters']),(32,32,48,4993))
        torch.nn.init.normal_(model.base.program[-1].weight,std=.02)
        x=torch.randn(2,3,8,8); x[:,0]=1
        state=(torch.randn(2,32,8,8),None)
        got=model.step(state,x); expected=model.base.step(state,x)
        torch.testing.assert_close(got[0],expected[0]);self.assertIsNone(got[1])

    def test_mask_isolation_and_positive_control(self):
        torch.manual_seed(23)
        model=make_cell(channels=4,reference_hidden=16).double()
        torch.nn.init.normal_(model.base.program[-1].weight,std=.02)
        x=torch.zeros(1,3,9,9,dtype=torch.float64)
        x[:,0]=1;x[:,0,:,4]=0;x[:,1,4,3]=1
        flip=x.clone();flip[:,1,4,3]=0;flip[:,2,4,3]=1
        with torch.no_grad():
            for cell,masked in [(model,True),(model.base,False)]:
                diff=cell.logits(cell.rollout(x,12))-cell.logits(cell.rollout(flip,12))
                leak=float(diff[:,:,:,5:].abs().max())
                if masked:self.assertEqual(leak,0.)
                else:self.assertGreater(leak,1e-12)

    def test_finite_backward_and_zero_initial_update(self):
        torch.manual_seed(1)
        model=make_cell(channels=4,reference_hidden=16)
        data=bank(8,2,77)
        state=model.initial(data['x']);after=model.step(state,data['x'])
        torch.testing.assert_close(state[0],after[0],rtol=0,atol=0)
        self.assertIsNone(after[1])
        loss=balanced_loss(model.logits(model.rollout(data['x'],8)),data['y'],data['mask'])
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

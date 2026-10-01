"""Small operator, matched-initialization and cross-region isolation checks."""
import unittest

import torch
from masked_cells import ARMS,BASE_ARMS,make_cell,masked_laplacian,frozen_cells
from tasks import bank,balanced_loss


class Tests(unittest.TestCase):
    def test_operator_and_spectral_bound(self):
        torch.manual_seed(71)
        h=torch.randn(2,3,7,9,dtype=torch.float64)
        z=torch.randn_like(h)
        mask=(torch.rand(2,1,7,9)>.3).double()
        lh=masked_laplacian(h,mask)
        lz=masked_laplacian(z,mask)
        self.assertLess(abs(float((h*lz).sum()-(lh*z).sum())),1e-10)
        self.assertGreaterEqual(float((h*lh).sum()),-1e-10)
        self.assertEqual(float(masked_laplacian(torch.ones_like(h),mask).abs().max()),0.)
        torch.testing.assert_close(masked_laplacian(h,torch.ones_like(mask)),frozen_cells.laplacian(h))
        n=5
        m=(torch.rand(1,1,n,n)>.2).double()
        basis=torch.eye(n*n,dtype=torch.float64).reshape(n*n,1,n,n)
        matrix=masked_laplacian(basis,m).reshape(n*n,n*n).T
        eigenvalues=torch.linalg.eigvalsh(matrix)
        self.assertGreaterEqual(float(eigenvalues.min()),-1e-12)
        self.assertLessEqual(float(eigenvalues.max()),8.+1e-12)

    def test_exact_initialization_and_rng(self):
        for arm in ARMS:
            torch.manual_seed(0)
            original=frozen_cells.make_cell(BASE_ARMS[arm])
            original_next=torch.rand(8)
            torch.manual_seed(0)
            masked=make_cell(arm)
            self.assertTrue(torch.equal(original_next,torch.rand(8)))
            for key,value in original.state_dict().items():
                self.assertTrue(torch.equal(value,masked.base.state_dict()[key]))
            self.assertEqual(sum(p.numel() for p in original.parameters()),sum(p.numel() for p in masked.parameters()))

    def test_forward_backward_and_all_open_equivalence(self):
        data=bank(8,2,77)
        for arm in ARMS:
            torch.manual_seed(0)
            m=make_cell(arm,4,16)
            state=m.rollout(data['x'],8)
            loss=balanced_loss(m.logits(state),data['y'],data['mask'])
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
            x=data['x'].clone(); x[:,0]=1
            h,v=m.initial(x)
            masked=m.step((h,v),x)
            original=m.base.step((h,v),x)
            for a,b in zip(masked,original):
                if a is not None: torch.testing.assert_close(a,b)
            beta,d=m.base.coefficients()
            self.assertTrue(bool((8*d<2*(1+beta)).all()))

    def test_whole_model_cannot_cross_wall(self):
        # Wall column4 separates two open regions. The source changes on its left.
        x=torch.zeros(1,3,9,9,dtype=torch.float64)
        x[:,0]=1; x[:,0,:,4]=0; x[:,1,4,3]=1
        xf=x.clone(); xf[:,1,4,3]=0; xf[:,2,4,3]=1
        for arm in ARMS:
            torch.manual_seed(23)
            m=make_cell(arm,4,16).double()
            torch.nn.init.normal_(m.base.program[-1].weight,std=.02)
            with torch.no_grad():
                first=m.rollout(x,12); second=m.rollout(xf,12)
                self.assertEqual(float((m.logits(first)-m.logits(second))[:,:,:,5:].abs().max()),0.)
                first=m.base.rollout(x,12); second=m.base.rollout(xf,12)
                self.assertGreater(float((m.base.logits(first)-m.base.logits(second))[:,:,:,5:].abs().max()),1e-10)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

"""Small correctness tests; these are not architecture efficacy tests."""
import unittest
import numpy as np
import torch
from cells import ARMS,Cell,laplacian,make_cell
from tasks import bank,balanced_loss,damage,metrics
from math_checks import block_matrix
from run_wind_tunnel import sustained_threshold


class Tests(unittest.TestCase):
    def test_sustained_threshold(self):
        def curve(values):
            return {str(t):{'balanced_accuracy':v} for t,v in zip((16,32,64),values)}
        self.assertIsNone(sustained_threshold(curve((.96,.94,.94))))
        self.assertEqual(sustained_threshold(curve((.96,.94,.96))),64)
        self.assertEqual(sustained_threshold(curve((.96,.97,.98))),16)

    def test_single_class_after_source_flip(self):
        # A valid flip can make a two-region image entirely positive or negative.
        for label in (0.,1.):
            y=torch.full((2,1,4,4),label)
            logits=torch.full_like(y,12. if label else -12.)
            report=metrics(logits,y,torch.ones_like(y))
            self.assertEqual(report['balanced_accuracy'],1.)
            expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
            self.assertAlmostEqual(report['bce'],float(expected),places=7)

    def test_laplacian(self):
        h=torch.randn(2,3,7,9,dtype=torch.float64)
        z=torch.randn_like(h)
        self.assertLess(abs(float((h*laplacian(z)).sum()-(laplacian(h)*z).sum())),1e-10)
        self.assertGreaterEqual(float((h*laplacian(h)).sum()),-1e-10)
        self.assertLess(float(laplacian(torch.ones_like(h)).abs().max()),1e-12)

    def test_beta_zero_equivalence(self):
        torch.manual_seed(0)
        model=Cell('inertial_rd',4,8)
        x=torch.randn(2,3,8,8)
        h,v=model.initial(x)
        b,d=model.coefficients()
        force=model.eta*model.program(torch.cat([h,x],dim=1))-d*laplacian(h)
        # At beta=0 the inertial state update equals the direct RD step.
        model.coefficients=lambda: (torch.zeros_like(b),d)
        actual_h,actual_v=model.step((h,torch.randn_like(h)),x)
        self.assertTrue(torch.equal(actual_h,h+force))
        self.assertTrue(torch.equal(actual_v,force))

    def test_pure_transport_bounds(self):
        for arm in ARMS:
            m=make_cell(arm,4,16)
            b,d=m.coefficients()
            if m.structured:
                self.assertTrue(bool((8*d<2*(1+b)).all()))

    def test_task_and_backward(self):
        data=bank(8,2,77)
        self.assertTrue(bool(((data['x'][:,1:]+data['x_flip'][:,1:]).sum()>0)))
        self.assertTrue(torch.equal(data['x'][:,0],data['x_flip'][:,0]))
        self.assertTrue(torch.equal(data['y_flip'], data['y']*(1-data['changed'])+(1-data['y'])*data['changed']))
        for arm in ARMS:
            m=make_cell(arm,4,16)
            state=m.rollout(data['x'],4)
            loss=balanced_loss(m.logits(state),data['y'],data['mask'])
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
            damaged=damage(state,.25,np.random.default_rng(12))
            if m.inertial:
                # Every erased content location also has erased velocity.
                zeros=(damaged[0].abs().sum(dim=1,keepdim=True)==0)
                self.assertTrue(bool((damaged[1]*zeros).abs().max()==0))

    def test_source_light_cone(self):
        torch.manual_seed(2)
        x=torch.zeros(1,3,13,13)
        xf=x.clone(); xf[0,1,6,6]=1
        yy,xx=torch.meshgrid(torch.arange(13),torch.arange(13),indexing='ij')
        outside=(yy-6).abs()+(xx-6).abs()>2
        for arm in ARMS:
            m=make_cell(arm,4,16)
            # Nonzero local program, so this also tests recurrent source injection.
            torch.nn.init.normal_(m.program[-1].weight,std=.02)
            s=m.rollout(x,2); f=m.rollout(xf,2)
            self.assertEqual(float((m.logits(s)-m.logits(f))[0,0,outside].detach().abs().max()),0.)

    def test_jacobian(self):
        q=-.2; b=.9
        def step(s):
            h,v=s[0],s[1]
            vv=b*v+q*h
            return torch.stack((h+vv,vv))
        s=torch.tensor([.3,-.1],dtype=torch.float64,requires_grad=True)
        J=torch.autograd.functional.jacobian(step,s).detach().numpy()
        self.assertLess(float(np.max(np.abs(J-block_matrix(b,[[q]])))),1e-14)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()

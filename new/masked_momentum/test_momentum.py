"""Minimal tests for the new generic mask perception, without training."""
import unittest
import torch
from momentum_cells import make_cell, frozen_cells
from tasks import bank, balanced_loss


class Tests(unittest.TestCase):
    def test_matching_and_all_open_nonzero_program(self):
        torch.manual_seed(0)
        original = frozen_cells.make_cell('momentum_nca')
        next_random = torch.rand(8)
        torch.manual_seed(0)
        model = make_cell()
        self.assertTrue(torch.equal(next_random, torch.rand(8)))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, model.base.state_dict()[name]))
        self.assertEqual(model.metadata()['parameters'], 4689)
        torch.nn.init.normal_(model.base.program[-1].weight, std=.02)
        x = torch.randn(2, 3, 8, 8); x[:, 0] = 1
        state = (torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8))
        for a, b in zip(model.step(state, x), model.base.step(state, x)):
            torch.testing.assert_close(a, b)

    def test_wall_isolation_with_unmasked_positive_control(self):
        torch.manual_seed(23)
        model = make_cell(channels=4, reference_hidden=16).double()
        torch.nn.init.normal_(model.base.program[-1].weight, std=.02)
        x = torch.zeros(1, 3, 9, 9, dtype=torch.float64)
        x[:, 0] = 1; x[:, 0, :, 4] = 0; x[:, 1, 4, 3] = 1
        flip = x.clone(); flip[:, 1, 4, 3] = 0; flip[:, 2, 4, 3] = 1
        with torch.no_grad():
            for cell, masked in [(model, True), (model.base, False)]:
                diff = cell.logits(cell.rollout(x, 12))-cell.logits(cell.rollout(flip, 12))
                leak = float(diff[:, :, :, 5:].abs().max())
                if masked: self.assertEqual(leak, 0.)
                else: self.assertGreater(leak, 1e-10)

    def test_finite_backward_and_no_explicit_transport_initially(self):
        torch.manual_seed(1)
        model = make_cell(channels=4, reference_hidden=16)
        data = bank(8, 2, 77)
        initial = model.initial(data['x'])
        first = model.step(initial, data['x'])
        torch.testing.assert_close(first[0], initial[0], rtol=0, atol=0)
        self.assertEqual(float(first[1].abs().max()), 0.)
        loss = balanced_loss(model.logits(model.rollout(data['x'], 8)), data['y'], data['mask'])
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

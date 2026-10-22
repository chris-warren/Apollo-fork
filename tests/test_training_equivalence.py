"""Batch 2 with per-sample losses == two batch-1 steps averaged (the paper's 8 GPUs x 1).

DDP averages each GPU's gradient, so the paper's step is the mean of 8 single-sample
loss gradients. training_step evaluates the GAN losses per sample (per_sample_mean),
so k samples per GPU give the same step. Checked here for both the discriminator and
the generator update, on the real model, discriminator and losses.
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

try:
    import torch
    from look2hear.discriminators.frequencydis import MultiFrequencyDiscriminator
    from look2hear.losses.gan_losses import MultiFrequencyDisLoss, MultiFrequencyGenLoss
    from look2hear.models.apollo import Apollo
    from look2hear.system.audio_litmodule import per_sample_mean
except ImportError:  # pragma: no cover
    torch = None

SR = 44100


@unittest.skipIf(torch is None, "training stack (torch, pytorch_lightning) not installed")
class PerSampleLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = Apollo(sr=SR, win=20, feature_dim=256, layer=6)
        cls.disc = MultiFrequencyDiscriminator(nch=2, window=[32, 64, 128, 256, 512, 1024, 2048])
        # Converge spectral norm's power iteration as training does, then eval() so that
        # repeated forward passes see the same discriminator. (From a fresh init in eval
        # mode its scale estimate is arbitrary and the losses explode to ~1e13.)
        with torch.no_grad():
            for _ in range(30):
                cls.disc(torch.randn(2, 2, int(0.1 * SR)), sample_rate=SR)
        cls.disc.eval()
        cls.loss_g, cls.loss_d = MultiFrequencyGenLoss(eps=1e-8), MultiFrequencyDisLoss(eps=1e-8)
        n = int(0.1 * SR)
        cls.ori = torch.randn(2, 2, n) * torch.tensor([0.05, 0.5]).view(2, 1, 1)  # different loudness
        cls.codec = cls.ori + 0.01 * torch.randn(2, 2, n)

    def step(self, ori, codec, combine):
        """One discriminator + generator update's losses and gradients (no optimiser step)."""
        for p in list(self.model.parameters()) + list(self.disc.parameters()):
            p.grad = None
        output = self.model(codec)
        est, _ = self.disc(output.detach(), sample_rate=SR)
        tgt, _ = self.disc(ori, sample_rate=SR)
        loss_d = combine(self.loss_d, tgt, est)
        loss_d.backward()
        grad_d = torch.cat([p.grad.flatten() for p in self.disc.parameters() if p.grad is not None]).clone()
        for p in self.disc.parameters():
            p.grad = None
        est, est_fm = self.disc(output, sample_rate=SR)
        _, tgt_fm = self.disc(ori, sample_rate=SR)
        loss_g = combine(self.loss_g, est, est_fm, tgt_fm, output, ori)
        loss_g.backward()
        grad_g = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None]).clone()
        return loss_d.item(), loss_g.item(), grad_d, grad_g

    def reference(self):
        """Two separate batch-1 steps, averaged: what DDP does across two GPUs."""
        runs = [self.step(self.ori[b:b + 1], self.codec[b:b + 1], lambda fn, *a: fn(*a)) for b in range(2)]
        return [sum(r[i] for r in runs) / 2 for i in range(4)]

    @staticmethod
    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    def test_per_sample_batch_matches_averaged_single_samples(self):
        loss_d, loss_g, grad_d, grad_g = self.step(self.ori, self.codec, per_sample_mean)
        ref_d, ref_g, ref_grad_d, ref_grad_g = self.reference()
        self.assertAlmostEqual(loss_d, ref_d, delta=1e-5 * abs(ref_d))
        self.assertAlmostEqual(loss_g, ref_g, delta=1e-5 * abs(ref_g))
        self.assertLess(self.rel(grad_d, ref_grad_d), 1e-4)  # float32 rounding: measured ~1.5e-5
        self.assertLess(self.rel(grad_g, ref_grad_g), 1e-4)  # measured ~1.4e-6; plain batching: ~0.5

    def test_plain_batched_loss_does_not_match(self):
        # Why per_sample_mean exists: the released loss on a batch of 2 is a different loss.
        _, loss_g, _, grad_g = self.step(self.ori, self.codec, lambda fn, *a: fn(*a))
        _, ref_g, _, ref_grad_g = self.reference()
        self.assertGreater(abs(loss_g - ref_g) / abs(ref_g), 1e-3)
        self.assertGreater(self.rel(grad_g, ref_grad_g), 1e-3)

    def test_losses_are_in_a_sane_range(self):
        loss_d, loss_g, _, _ = self.step(self.ori, self.codec, per_sample_mean)
        self.assertLess(loss_d, 10)
        self.assertLess(loss_g, 10)

    def test_batch_of_one_is_the_released_call(self):
        args = (self.disc(self.ori[:1], sample_rate=SR)[0], self.disc(self.codec[:1], sample_rate=SR)[0])
        self.assertEqual(per_sample_mean(self.loss_d, *args).item(), self.loss_d(*args).item())


if __name__ == "__main__":
    unittest.main()

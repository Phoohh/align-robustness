"""Small numerical tests of the new kernel; no datasets, GPUs or downloads."""

import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    raise unittest.SkipTest('Selection-free numerical tests require PyTorch') from error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.adv_train import setup
from training.adv_train import saar


class ToyModel(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.first = nn.Linear(2, 4)
        self.norm = nn.BatchNorm1d(4) if batch_norm else nn.Identity()
        self.last = nn.Linear(4, 3)
        with torch.no_grad():
            self.last.bias.copy_(torch.tensor([3., 0., -3.]))
        self.calls = []

    def forward(self, x):
        self.calls.append((x.detach().clone(), self.training))
        return self.last(torch.tanh(self.norm(self.first(x))))


class Scheduler:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


class SelectionFreeLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.P = SimpleNamespace(lam=1.3, T=.5, BD_alpha=.75,
                                 optimizer='sgd', n_gpus=1)
        self.x1 = torch.tensor([[.1, .3], [.4, .2], [.7, .1]])
        self.x2 = torch.tensor([[.8, .2], [.3, .7], [.1, .9]])
        self.labels = torch.tensor([2, 2, 2])
        self.env = {'SAAR_ATTACK_VIEWS': 'independent', 'AWP_TRAIN': '0',
                    'AWP_OBJECTIVE': 'align_only', 'AWP_GAMMA': '.02',
                    'AWP_START': '0'}

    def attack(self, x, y):
        # A nonlinear input-dependent attack distinguishes shared and independent.
        return (x + .1 * (x > .35).float() - .03).clamp(0, 1)

    def run_batch(self, model, optimizer, **env):
        scheduler = Scheduler()
        with mock.patch.dict(os.environ, {**self.env, **env}), \
                mock.patch.object(saar.np.random, 'beta', return_value=.25), \
                mock.patch('builtins.print'):
            saar.train(self.P, 1, model, nn.CrossEntropyLoss(), optimizer,
                       scheduler, [([self.x1, self.x2], self.labels)], self.attack)
        self.assertEqual(scheduler.calls, 1)

    def test_attack_independent_and_shared_are_distinct(self):
        attack = mock.Mock(side_effect=self.attack)
        pair = saar.adversarial_views(self.x1, self.x2, self.labels, attack,
                                      'independent')
        attack.assert_called_once()
        self.assertTrue(torch.equal(attack.call_args.args[0],
                                    torch.cat((self.x1, self.x2))))
        self.assertTrue(torch.equal(attack.call_args.args[1], self.labels.repeat(2)))
        self.assertTrue(torch.equal(pair, self.attack(torch.cat((self.x1, self.x2)),
                                                     self.labels.repeat(2))))
        attack.reset_mock()
        shared = saar.adversarial_views(self.x1, self.x2, self.labels, attack, 'shared')
        self.assertTrue(torch.equal(attack.call_args.args[0], self.x1))
        adv1 = self.attack(self.x1, self.labels)
        torch.testing.assert_close(shared[3:], (self.x2 + adv1 - self.x1).clamp(0, 1))
        self.assertFalse(torch.equal(shared[3:], pair[3:]))
        self.assertFalse(pair.requires_grad)
        self.assertFalse(shared.requires_grad)

    def test_wrong_clean_predictions_still_use_every_sample_and_full_ce(self):
        model = ToyModel()
        with torch.no_grad():
            self.assertTrue((model(torch.cat((self.x1, self.x2))).argmax(1)
                             != self.labels.repeat(2)).all())
        model.calls.clear()
        before = {k: p.clone() for k, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        self.run_batch(model, optimizer)
        # Exactly CE(2B), interpolated alignment(B), clean alignment(2B).
        # There are no extra eval forwards for boundary/correctness diagnostics.
        self.assertEqual([len(x) for x, _ in model.calls], [6, 3, 6])
        self.assertTrue(all(mode for _, mode in model.calls))
        adv_pair = self.attack(torch.cat((self.x1, self.x2)), self.labels.repeat(2))
        self.assertTrue(torch.equal(model.calls[0][0], adv_pair))
        self.assertTrue(torch.equal(model.calls[1][0], .75 * adv_pair[:3] + .25 * adv_pair[3:]))
        self.assertTrue(any(not torch.equal(p, before[k]) for k, p in model.named_parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_no_awp_matches_explicit_all_sample_gradient(self):
        model = ToyModel()
        expected = copy.deepcopy(model)
        adv = self.attack(torch.cat((self.x1, self.x2)), self.labels.repeat(2))
        ce, con = saar.batch_losses(self.P, expected, nn.CrossEntropyLoss(),
                                    torch.cat((self.x1, self.x2)), adv, self.labels, .75)
        expected_optimizer = torch.optim.SGD(expected.parameters(), lr=.05)
        (ce + con).backward()
        expected_optimizer.step()
        self.run_batch(model, torch.optim.SGD(model.parameters(), lr=.05),
                       AWP_TRAIN='1', AWP_OBJECTIVE='none')
        for actual, reference in zip(model.parameters(), expected.parameters()):
            self.assertTrue(torch.equal(actual, reference))

    def test_start0_uses_awp_on_first_epoch_and_disabled_awp_is_start_invariant(self):
        initial = ToyModel()
        start0, start10 = copy.deepcopy(initial), copy.deepcopy(initial)
        self.run_batch(start0, torch.optim.SGD(start0.parameters(), lr=.05),
                       AWP_TRAIN='1', AWP_START='0')
        self.run_batch(start10, torch.optim.SGD(start10.parameters(), lr=.05),
                       AWP_TRAIN='1', AWP_START='10')
        self.assertEqual(len(start0.calls), 6)
        self.assertEqual(len(start10.calls), 3)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(start0.parameters(), start10.parameters())))
        disabled0, disabled10 = copy.deepcopy(initial), copy.deepcopy(initial)
        self.run_batch(disabled0, torch.optim.SGD(disabled0.parameters(), lr=.05),
                       AWP_TRAIN='1', AWP_OBJECTIVE='none', AWP_START='0')
        self.run_batch(disabled10, torch.optim.SGD(disabled10.parameters(), lr=.05),
                       AWP_TRAIN='1', AWP_OBJECTIVE='none', AWP_START='10')
        for a, b in zip(disabled0.state_dict().values(), disabled10.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_awp_source_is_alignment_or_ce_and_weights_restore_exactly(self):
        for objective in ('align_only', 'ce_only'):
            with self.subTest(objective=objective):
                model = ToyModel()
                reference = copy.deepcopy(model)
                adv = self.attack(torch.cat((self.x1, self.x2)), self.labels.repeat(2))
                ce, con = saar.batch_losses(self.P, reference, nn.CrossEntropyLoss(),
                                            torch.cat((self.x1, self.x2)), adv, self.labels, .75)
                expected_source = con if objective == 'align_only' else ce
                source_values = []
                original_grad = torch.autograd.grad

                def record_grad(source, *args, **kwargs):
                    source_values.append(source.detach().clone())
                    return original_grad(source, *args, **kwargs)

                originals = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.SGD(model.parameters(), lr=0.)
                with mock.patch.object(torch.autograd, 'grad', side_effect=record_grad):
                    self.run_batch(model, optimizer, AWP_TRAIN='1', AWP_OBJECTIVE=objective)
                self.assertEqual(len(source_values), 1)
                torch.testing.assert_close(source_values[0], expected_source.detach())
                self.assertEqual([len(x) for x, _ in model.calls], [6, 3, 6, 6, 3, 6])
                for param, original in zip(model.parameters(), originals):
                    self.assertTrue(torch.equal(param, original))
                    self.assertTrue(torch.isfinite(param.grad).all())

    def test_awp_restores_weights_before_step_and_on_recompute_failure(self):
        model = ToyModel()
        originals = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        source = model(self.x1).square().mean()

        def fail():
            self.assertTrue(any(not torch.equal(p, before)
                                for p, before in zip(model.parameters(), originals)))
            raise RuntimeError('deliberate recompute failure')

        with mock.patch.object(optimizer, 'step') as step:
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                saar._weight_perturbed_backward(model, optimizer, source, .02, fail)
            step.assert_not_called()
        for param, original in zip(model.parameters(), originals):
            self.assertTrue(torch.equal(param, original))

        source = model(self.x1).square().mean()

        def recompute():
            loss = model(self.x2).square().mean()
            return loss, loss * .1

        real_step = optimizer.step

        def assert_restored_then_step():
            for param, original in zip(model.parameters(), originals):
                self.assertTrue(torch.equal(param, original))
            real_step()

        with mock.patch.object(optimizer, 'step', side_effect=assert_restored_then_step):
            saar._weight_perturbed_backward(model, optimizer, source, .02, recompute)
        self.assertTrue(any(not torch.equal(p, before)
                            for p, before in zip(model.parameters(), originals)))

    def test_awp_is_relative_per_weight_tensor_and_excludes_bias(self):
        model = ToyModel()
        adv = self.attack(torch.cat((self.x1, self.x2)), self.labels.repeat(2))
        ce, source = saar.batch_losses(self.P, model, nn.CrossEntropyLoss(),
                                       torch.cat((self.x1, self.x2)), adv, self.labels, .75)
        weights = [p for p in model.parameters() if p.dim() > 1]
        gradients = torch.autograd.grad(source, weights, retain_graph=True)
        originals = [p.detach().clone() for p in model.parameters()]
        expected = {}
        for p, grad in zip(weights, gradients):
            expected[id(p)] = p.detach() + .02 * p.detach().norm() * grad / (grad.norm() + 1e-12)

        def recompute():
            for param, original in zip(model.parameters(), originals):
                torch.testing.assert_close(param, expected.get(id(param), original))
            return saar.batch_losses(self.P, model, nn.CrossEntropyLoss(),
                                      torch.cat((self.x1, self.x2)), adv, self.labels, .75)

        saar._weight_perturbed_backward(model, torch.optim.SGD(model.parameters(), lr=0.),
                                        source, .02, recompute)
        for param, original in zip(model.parameters(), originals):
            self.assertTrue(torch.equal(param, original))

    def test_awp_preserves_training_bn_updates_and_checks_nonfinite_loss(self):
        model = ToyModel(batch_norm=True)
        self.run_batch(model, torch.optim.SGD(model.parameters(), lr=.01), AWP_TRAIN='1')
        self.assertEqual(model.norm.num_batches_tracked.item(), 6)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        source = model(self.x1).sum() * float('nan')
        with self.assertRaises(FloatingPointError):
            saar._weight_perturbed_backward(model, optimizer, source, .02, lambda: None)

    def test_dispatch_is_protocol_scoped(self):
        P = SimpleNamespace(dataset='cifar10', model='resnet18', adv_method='pgd',
                            distance='Linf', augment_type='base', consistency=False,
                            BD=True, experiment_protocol='selection_free_v2', seed=0,
                            suffix=None)
        function, filename = setup('adv_train', P)
        self.assertIs(function, saar.train)
        self.assertIn('selection_free_v2', filename)


if __name__ == '__main__':
    unittest.main()

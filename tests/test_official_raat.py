"""Numerical comparison with the pinned upstream RAAT implementation.

The embedded fixtures are verbatim upstream MIT-licensed source, identified by
SHA-256 below. No network or private audit directory is needed to run the tests.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM_KERNEL = "import time\nimport torch.optim\nfrom training import _jensen_shannon_div\nfrom utils.utils import AverageMeter\n\nimport numpy as np\nfrom copy import deepcopy\n\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\n\ndef train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):\n\n    if logger is None:\n        log_ = print\n    else:\n        log_ = logger.log\n\n    batch_time = AverageMeter()\n    data_time = AverageMeter()\n\n    losses = dict()\n    losses['cls'] = AverageMeter()\n    losses['con'] = AverageMeter()\n\n    check = time.time()\n    for n, (images, labels) in enumerate(loader):\n        model.train()\n        count = n * P.n_gpus  # number of trained samples\n\n        data_time.update(time.time() - check)\n        check = time.time()\n\n        batch_size = images[0].size(0)\n        labels = labels.to(device)\n\n        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)\n        images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B\n        images_adv = adversary(images_pair, labels.repeat(2))\n\n        # --- divide non-boundary, boundary and misclassified samples --- #\n        model.eval()\n\n        eval_output_benign = model(images_pair)\n        eval_predicted_labels_benign = torch.argmax(eval_output_benign, 1).cpu().data.numpy()\n\n        eval_output_adv = model(images_adv)\n        eval_predicted_labels_adv = torch.argmax(eval_output_adv, 1).cpu().data.numpy()\n\n        adv_perturb = images_adv.clone().detach() - images_pair.clone().detach()\n        adv_input_reduced = images_pair.clone().detach() + P.BD_boundary_range * adv_perturb\n        eval_output_adv_reduced = model(adv_input_reduced)\n        eval_predicted_labels_adv_reduced = torch.argmax(eval_output_adv_reduced, 1).cpu().data.numpy()\n\n        model.train()\n\n        labels_np = deepcopy(labels.repeat(2)).cpu().numpy()\n        all_index = np.array([i for i in range(len(labels_np))])\n        misclassified_index = all_index[eval_predicted_labels_benign != labels_np]\n        boundary_index = all_index[(eval_predicted_labels_benign == labels_np)\n                                   & (eval_predicted_labels_adv != labels_np)\n                                   & (eval_predicted_labels_adv_reduced != labels_np)]\n        non_boundary_index = np.sort(\n            np.array(list(set(list(all_index)) - set(list(misclassified_index)) - set(list(boundary_index)))))\n\n        # reconstruct supervised set for CE loss\n        data_ce = torch.cat((\n            images_adv[non_boundary_index], adv_input_reduced[boundary_index], images_adv[misclassified_index]))\n        labels_ce = torch.cat((\n            labels.repeat(2)[non_boundary_index], labels.repeat(2)[boundary_index], labels.repeat(2)[misclassified_index]))\n\n        outputs_ce = model(data_ce)\n        loss_ce = criterion(outputs_ce, labels_ce)\n        loss = loss_ce\n\n        ### ICT regularization ###\n        ori_train_len = int(len(labels_np) / 2)\n        ori_all_index = [idx for idx in range(ori_train_len)]\n        misclassified_index_1 = [idx for idx in misclassified_index if idx < ori_train_len]\n        misclassified_index_2 = [idx - ori_train_len for idx in misclassified_index if idx >= ori_train_len]\n        ori_misclassified_index_set = set(misclassified_index_1) | set(misclassified_index_2)\n        ICT_index = np.sort(np.array(list(set(ori_all_index) - ori_misclassified_index_set)))\n\n        if len(ICT_index) > 0:\n            _lambda = np.random.beta(P.BD_alpha, P.BD_alpha)\n            mixup_rate = max(_lambda, (1 - _lambda))\n            images_adv1, images_adv2 = images_adv.chunk(2)\n            ICT_data = (mixup_rate * images_adv1[ICT_index] + (1 - mixup_rate) * images_adv2[ICT_index])\n            ICT_output_1 = model(ICT_data)\n\n            outputs_benign = model(images_pair)\n            outputs_benign1, outputs_benign2 = outputs_benign.chunk(2)\n            ICT_output_2 = (mixup_rate * outputs_benign1[ICT_index] + (1 - mixup_rate) * outputs_benign2[ICT_index])\n\n            loss_con = P.lam * _jensen_shannon_div(ICT_output_1, ICT_output_2, P.T)\n            loss += loss_con\n\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n\n        lr = optimizer.param_groups[0]['lr']\n\n        batch_time.update(time.time() - check)\n\n        ### Log losses ###\n        losses['cls'].update(loss_ce.item(), batch_size)\n        if len(ICT_index) > 0:\n            losses['con'].update(loss_con.item(), batch_size)\n\n        if count % 50 == 0:\n            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\\n'\n                 '[LossC %f] [LossCon %f]' %\n                 (epoch, count, batch_time.value, data_time.value, lr,\n                  losses['cls'].value, losses['con'].value))\n\n        check = time.time()\n\n    if P.optimizer == 'sgd':\n        scheduler.step()\n\n    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossCon %f]' %\n         (batch_time.average, data_time.average,\n          losses['cls'].average, losses['con'].average))\n\n    if logger is not None:\n        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)\n        logger.scalar_summary('train/loss_con', losses['con'].average, epoch)\n        logger.scalar_summary('train/batch_time', batch_time.average, epoch)\n"
UPSTREAM_HELPER = "import torch\nimport torch.nn.functional as F\n\n\ndef _kl_div(logit1, logit2):\n    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction='batchmean')\n\n\ndef _jensen_shannon_div(logit1, logit2, T=1.):\n    prob1 = F.softmax(logit1/T, dim=1)\n    prob2 = F.softmax(logit2/T, dim=1)\n    mean_prob = 0.5 * (prob1 + prob2)\n\n    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))\n    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')\n    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')\n    return jsd * 0.5\n\n\ndef _jensen_shannon_div_without_reduction(logit1, logit2, T=1.):\n    prob1 = F.softmax(logit1/T, dim=1)\n    prob2 = F.softmax(logit2/T, dim=1)\n    mean_prob = 0.5 * (prob1 + prob2)\n\n    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))\n    jsd = F.kl_div(logsoftmax, prob1, reduction='none')\n    jsd += F.kl_div(logsoftmax, prob2, reduction='none')\n    return jsd * 0.5\n"

KERNEL_SHA256 = 'b38de0879c4c018493946f4cde9b2b320861ede1dd99e0575ceca49bf388e091'
HELPER_SHA256 = 'a168b4148d6e91a78a4a66a5973ed59109c90ee853234b850e8613a21d39ce91'


class UpstreamFixtureTests(unittest.TestCase):
    def test_verbatim_upstream_fixtures(self):
        self.assertEqual(hashlib.sha256(UPSTREAM_KERNEL.encode()).hexdigest(), KERNEL_SHA256)
        self.assertEqual(hashlib.sha256(UPSTREAM_HELPER.encode()).hexdigest(), HELPER_SHA256)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is needed for official RAAT numerical checks')
class OfficialRAATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        torch.set_num_threads(1)
        # AverageMeter itself has no tensorboard dependency. Extract the actual
        # utility so a CPU test does not need the unrelated logger package.
        tree = ast.parse((ROOT / 'utils/utils.py').read_text())
        meter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AverageMeter')
        utilities = ModuleType('utils.utils')
        exec(compile(ast.Module(body=[meter], type_ignores=[]), '<AverageMeter>', 'exec'), utilities.__dict__)
        helper = {}
        exec(compile(UPSTREAM_HELPER, '<pinned upstream JS>', 'exec'), helper)
        training = ModuleType('training')
        training._jensen_shannon_div = helper['_jensen_shannon_div']
        with patch.dict(sys.modules, {'utils.utils': utilities, 'training': training}):
            source = importlib.util.spec_from_file_location('_official_raat_under_test', ROOT / 'training/adv_train/raat_official.py')
            cls.portable = importlib.util.module_from_spec(source)
            source.loader.exec_module(cls.portable)
            cls.raw = {}
            exec(compile(UPSTREAM_KERNEL, '<pinned upstream RAAT>', 'exec'), cls.raw)
        cls.raw['device'] = torch.device('cpu')
        cls.raw_helper = staticmethod(helper['_jensen_shannon_div'])

    def args(self):
        return SimpleNamespace(n_gpus=1, optimizer='sgd', BD_boundary_range=.1, BD_alpha=.75, T=.5, lam=1.)

    def environment(self, **extra):
        values = dict(AWP_TRAIN='0', AWP_GAMMA='0', SAAR_ATTACK_VIEWS='independent',
                      ALIGN_ALL_SAMPLES='1')
        values.update(extra)
        return patch.dict('os.environ', values)

    def model(self, features=2):
        torch = self.torch
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(features, 3 if features == 2 else 2).double()
                self.calls = []
            def forward(self, x):
                self.calls.append((self.training, x.detach().clone()))
                return self.fc(x)
        model = RecordingModel()
        with torch.no_grad():
            if features == 2:
                model.fc.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., -1.]]))
                model.fc.bias.copy_(torch.tensor([.4, .2, -.5]))
            else:
                model.fc.weight.copy_(torch.tensor([[-1.], [1.]]))
                model.fc.bias.copy_(torch.tensor([.5, -.5]))
        return model

    def run_kernel(self, kernel, model, views, labels, attack=None):
        torch = self.torch
        calls = []
        def adversary(images, target):
            calls.append((images.detach().clone(), target.detach().clone()))
            if attack is not None:
                return attack(images)
            # Each row obtains its own input-dependent perturbation.
            return (images.detach() + .2 * (.25 - images.detach())).clamp(0., 1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9, weight_decay=.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 105], gamma=.1)
        with self.environment(), redirect_stdout(io.StringIO()):
            self.np.random.seed(11)
            kernel(self.args(), 1, model, torch.nn.CrossEntropyLoss(), optimizer,
                   scheduler, [(views, labels)], adversary)
        return calls, optimizer, scheduler

    def test_one_step_matches_raw_for_all_some_and_no_alignment_samples(self):
        torch = self.torch
        views = [torch.tensor([[.95, .1], [.4, .7], [.8, .2]], dtype=torch.float64),
                 torch.tensor([[.85, .2], [.3, .8], [.6, .25]], dtype=torch.float64)]
        for label_values, eligible in (([0, 1, 0], 3), ([0, 0, 2], 1), ([2, 0, 2], 0)):
            with self.subTest(eligible=eligible):
                labels = torch.tensor(label_values)
                original = self.model()
                portable = deepcopy(original)
                raw_calls, _, raw_schedule = self.run_kernel(self.raw['train'], original, views, labels)
                calls, _, schedule = self.run_kernel(self.portable.train, portable, views, labels)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0].shape[0], 2 * len(labels))
                self.assertTrue(torch.equal(calls[0][0], torch.cat(views)))
                self.assertTrue(torch.equal(calls[0][1], labels.repeat(2)))
                self.assertTrue(torch.equal(calls[0][0], raw_calls[0][0]))
                for name, tensor in original.state_dict().items():
                    torch.testing.assert_close(tensor, portable.state_dict()[name], rtol=1e-13, atol=1e-14, msg=name)
                for original_param, portable_param in zip(original.parameters(), portable.parameters()):
                    torch.testing.assert_close(original_param.grad, portable_param.grad, rtol=1e-13, atol=1e-14)
                # No AWP source/recomputation pass is inserted into RAAT.
                self.assertEqual(len(portable.calls), 6 if eligible else 4)
                train_calls = [data for is_train, data in portable.calls if is_train]
                if eligible:
                    self.assertEqual(len(train_calls[1]), eligible)
                self.assertEqual(schedule.last_epoch, raw_schedule.last_epoch)

    def test_official_boundary_scaling_and_correctness_alignment_are_retained(self):
        torch = self.torch
        model = self.model(features=1)
        view = torch.tensor([[.51], [.9], [.2]], dtype=torch.float64)
        self.run_kernel(self.portable.train, model, [view, view.clone()],
                        torch.ones(3, dtype=torch.long),
                        attack=lambda x: (x.detach() - .4).clamp(0., 1.))
        train_calls = [data for is_train, data in model.calls if is_train]
        # Non-boundary full attack; boundary 0.1-scaled attack; clean-wrong full attack.
        expected = torch.tensor([[.5], [.5], [.47], [.47], [0.], [0.]], dtype=torch.float64)
        self.assertTrue(torch.allclose(train_calls[0], expected, atol=1e-14, rtol=0))
        self.assertEqual(train_calls[1].shape[0], 2)  # the clean-wrong pair is excluded
        self.assertEqual(len(train_calls), 3)

    def test_official_js_uses_unclipped_logits_and_matches_upstream_gradients(self):
        torch = self.torch
        first = torch.tensor([[100., 90., 0.]], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([[90., 100., 0.]], dtype=torch.float64, requires_grad=True)
        expected = self.raw_helper(first, second, .5)
        actual = self.portable._jensen_shannon_div(first, second, .5)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
        self.assertGreater(float(actual.detach()), .69)  # clipping both leading logits to 50 would erase this JS
        expected_grad = torch.autograd.grad(expected, [first, second], retain_graph=True)
        actual_grad = torch.autograd.grad(actual, [first, second])
        for a, b in zip(expected_grad, actual_grad):
            torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-14)

    def test_underflowed_probability_targets_have_finite_logit_gradients(self):
        torch = self.torch
        for dtype, magnitude in ((torch.float32, 100.), (torch.float64, 1000.)):
            with self.subTest(dtype=dtype):
                first = torch.tensor([[magnitude, 0., -magnitude]], dtype=dtype, requires_grad=True)
                second = torch.tensor([[0., magnitude, -magnitude]], dtype=dtype, requires_grad=True)
                self.assertTrue((torch.softmax(first / .5, 1) == 0).any())
                raw = self.raw_helper(first, second, .5)
                raw_grad = torch.autograd.grad(raw, [first, second])
                self.assertTrue(torch.isfinite(raw))
                self.assertTrue(any(not torch.isfinite(gradient).all() for gradient in raw_grad))
                actual = self.portable._jensen_shannon_div(first, second, .5)
                gradients = torch.autograd.grad(actual, [first, second])
                torch.testing.assert_close(actual, raw.detach(), rtol=1e-6, atol=1e-7)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_stable_js_matches_upstream_on_ordinary_random_logits(self):
        torch = self.torch
        torch.manual_seed(27)
        for dtype in (torch.float32, torch.float64):
            first = torch.randn(7, 10, dtype=dtype, requires_grad=True)
            second = torch.randn(7, 10, dtype=dtype, requires_grad=True)
            expected = self.raw_helper(first, second, .5)
            actual = self.portable._jensen_shannon_div(first, second, .5)
            reference_grad = torch.autograd.grad(expected, [first, second])
            gradients = torch.autograd.grad(actual, [first, second])
            tolerance = dict(rtol=1e-5, atol=1e-7) if dtype == torch.float32 else dict(rtol=1e-13, atol=1e-14)
            torch.testing.assert_close(actual, expected.detach(), **tolerance)
            for gradient, reference in zip(gradients, reference_grad):
                torch.testing.assert_close(gradient, reference, **tolerance)

    def test_extreme_logits_train_update_is_finite(self):
        torch = self.torch
        model = self.model().float()
        with torch.no_grad():
            model.fc.weight.mul_(100.)
        views = [torch.tensor([[.95, .1], [.4, .7], [.8, .2]]),
                 torch.tensor([[.85, .2], [.3, .8], [.6, .25]])]
        labels = torch.tensor([0, 1, 0])
        before = deepcopy(model.state_dict())
        self.run_kernel(self.portable.train, model, views, labels)
        self.assertTrue(all(torch.isfinite(parameter).all() and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()))

    def test_awp_and_shared_attack_requests_are_rejected(self):
        torch = self.torch
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100])
        for settings in ({'AWP_TRAIN': '1'}, {'AWP_GAMMA': '.002'}, {'SAAR_ATTACK_VIEWS': 'shared'}):
            with self.environment(**settings), self.assertRaises(ValueError):
                self.portable.train(self.args(), 1, model, torch.nn.CrossEntropyLoss(), optimizer,
                                    scheduler, [], adversary=lambda *a: None)

    def test_nonfinite_attack_stops_before_parameter_update(self):
        torch = self.torch
        model = self.model()
        before = deepcopy(model.state_dict())
        view = torch.tensor([[.9, .1]], dtype=torch.float64)
        with self.assertRaises(FloatingPointError):
            self.run_kernel(self.portable.train, model, [view, view.clone()],
                            torch.tensor([0]), attack=lambda x: torch.full_like(x, float('nan')))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]))


if __name__ == '__main__':
    unittest.main()

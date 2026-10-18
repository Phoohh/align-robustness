"""Selection-free SAAR kernel, kept separate from archived experiment code.

Every original sample contributes both adversarial views to classification and
one pair to alignment. No prediction-dependent screening is performed here.
"""

import math
import os
import time

import numpy as np
import torch

from training import _jensen_shannon_div


def adversarial_views(view1, view2, labels, adversary, attack_views):
    """Construct independent attacks, or the explicitly labeled shared ablation."""
    if attack_views == 'independent':
        # A single 2B call optimizes a separate input perturbation for every view.
        return adversary(torch.cat((view1, view2)), labels.repeat(2)).detach()
    if attack_views == 'shared':
        adv1 = adversary(view1, labels).detach()
        delta = (adv1 - view1).detach()
        adv2 = (view2.detach() + delta).clamp(0., 1.)
        return torch.cat((adv1, adv2))
    raise ValueError(f'Unknown SAAR_ATTACK_VIEWS: {attack_views}')


def batch_losses(P, model, criterion, clean_pair, adversarial_pair, labels, mixup_rate):
    """Full-budget CE and all-pair logit alignment; neither branch masks samples."""
    loss_ce = criterion(model(adversarial_pair), labels.repeat(2))
    adv1, adv2 = adversarial_pair.chunk(2)
    mixed_adv = mixup_rate * adv1 + (1. - mixup_rate) * adv2
    mixed_output = model(mixed_adv)
    clean1, clean2 = model(clean_pair).chunk(2)
    clean_target = mixup_rate * clean1 + (1. - mixup_rate) * clean2
    # Keep both sides differentiable and preserve the recorded temperature/JS.
    loss_con = P.lam * _jensen_shannon_div(mixed_output, clean_target, P.T)
    return loss_ce, loss_con


def _require_finite(value, description):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'Non-finite {description} in selection_free_v2')


def _check_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _require_finite(parameter.grad, f'training gradient {name}')


def _weight_perturbed_backward(model, optimizer, source, gamma, recompute):
    """Differentiate at perturbed weights, restoring exact originals on failure."""
    named_params = [(name, p) for name, p in model.named_parameters()
                    if p.requires_grad and p.dim() > 1]
    optimizer.zero_grad()
    _require_finite(source, 'weight-perturbation source loss')
    grads = torch.autograd.grad(source, [p for _, p in named_params],
                                allow_unused=True) if named_params else []
    restored = []
    try:
        with torch.no_grad():
            for (name, param), grad in zip(named_params, grads):
                if grad is None:
                    continue
                _require_finite(grad, f'weight-perturbation gradient {name}')
                perturbation = gamma * param.norm() * grad / (grad.norm() + 1e-12)
                _require_finite(perturbation, f'weight perturbation {name}')
                restored.append((param, param.detach().clone()))
                param.add_(perturbation)
        loss_ce, loss_con = recompute()
        loss = loss_ce + loss_con
        _require_finite(loss, 'perturbed training loss')
        optimizer.zero_grad()
        loss.backward()
        _check_gradients(model)
        # As in the archived kernel, training forwards update BatchNorm buffers.
        # Only trainable weights are restored, not those running statistics.
    finally:
        with torch.no_grad():
            for param, original in restored:
                param.copy_(original)
    optimizer.step()
    return loss


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):
    log = print if logger is None else logger.log
    attack_views = os.environ.get('SAAR_ATTACK_VIEWS', 'independent')
    if attack_views not in ('independent', 'shared'):
        raise ValueError(f'Unknown SAAR_ATTACK_VIEWS: {attack_views}')
    objective = os.environ.get('AWP_OBJECTIVE', 'align_only').lower()
    if objective not in ('align_only', 'ce_only', 'none'):
        raise ValueError(f'Unsupported selection-free AWP objective: {objective}')
    gamma = float(os.environ.get('AWP_GAMMA', '0.005'))
    awp_start = int(os.environ.get('AWP_START', '10'))
    if not math.isfinite(gamma) or gamma < 0 or awp_start < 0:
        raise ValueError('AWP_GAMMA and AWP_START must be finite and nonnegative')
    use_awp = (os.environ.get('AWP_TRAIN', '0') == '1'
               and objective != 'none' and gamma > 0 and epoch >= awp_start)
    if not math.isfinite(float(P.BD_alpha)) or P.BD_alpha <= 0:
        raise ValueError('The Beta distribution shape must be positive')
    device = next(model.parameters()).device
    total_ce = total_con = total_examples = total_batch_time = 0.
    last_ce = last_con = last_batch_time = last_data_time = 0.
    check = time.time()
    log(f'[SAAR] protocol=selection_free_v2 attack_views={attack_views} '
        f'alignment_selection=none boundary_handling=none objective={objective}')

    for n, (images, labels) in enumerate(loader):
        model.train()
        last_data_time = time.time() - check
        check = time.time()
        view1, view2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        clean_pair = torch.cat((view1, view2))
        adversarial_pair = adversarial_views(view1, view2, labels, adversary, attack_views)
        beta = np.random.beta(P.BD_alpha, P.BD_alpha)
        mixup_rate = max(beta, 1. - beta)

        def recompute():
            return batch_losses(P, model, criterion, clean_pair,
                                adversarial_pair, labels, mixup_rate)

        loss_ce, loss_con = recompute()
        loss = loss_ce + loss_con
        _require_finite(loss, 'training loss')
        if use_awp:
            source = loss_con if objective == 'align_only' else loss_ce
            perturbed_loss = _weight_perturbed_backward(
                model, optimizer, source, gamma, recompute)
            if n % 50 == 0:
                log(f'[AWPTrain] objective={objective} gamma={gamma} '
                    f'start={awp_start} loss_awp={perturbed_loss.item():.6f}')
        else:
            optimizer.zero_grad()
            loss.backward()
            _check_gradients(model)
            optimizer.step()

        last_ce, last_con = loss_ce.item(), loss_con.item()
        total_ce += last_ce * batch_size
        total_con += last_con * batch_size
        total_examples += batch_size
        last_batch_time = time.time() - check
        total_batch_time += last_batch_time
        if n * P.n_gpus % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            log('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                '[LossC %f] [LossCon %f]' %
                (epoch, n * P.n_gpus, last_batch_time, last_data_time,
                 lr, last_ce, last_con))
        check = time.time()

    if not total_examples:
        raise ValueError('selection_free_v2 received an empty training loader')
    if P.optimizer == 'sgd':
        scheduler.step()
    mean_ce, mean_con = total_ce / total_examples, total_con / total_examples
    mean_batch_time = total_batch_time / (n + 1)
    log('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossCon %f]' %
        (mean_batch_time, last_data_time, mean_ce, mean_con))
    if logger is not None:
        logger.scalar_summary('train/loss_cls', mean_ce, epoch)
        logger.scalar_summary('train/loss_con', mean_con, epoch)
        logger.scalar_summary('train/batch_time', mean_batch_time, epoch)

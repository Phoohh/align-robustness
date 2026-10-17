"""Official RAAT objective, isolated from the archived shared-attack variant.

Adapted from FlaAI/RAAT, commit 61de35b7e2db141e2904eee59460a6d98964525d,
RAAT_Code_(Sec_5.1&5.2)/training/adv_train/adv_train_bd.py. The upstream MIT
license is retained at the repository root. See docs/OFFICIAL_RAAT_AUDIT.md.
"""
import math
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim

from utils.utils import AverageMeter


def _jensen_shannon_div(logit1, logit2, T=1.):
    """Upstream JS objective with a finite logit-space entropy derivative.

    Softmax can round a target probability to zero. Differentiating KL's
    probability-target entropy then evaluates log(0), although the composed
    logit-space derivative has a finite limit. Use log_softmax for that entropy
    term, preserving the probabilities and the upstream mixture floor exactly.
    No logits/target probabilities are clipped, detached, or sanitized.
    """
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    log_prob1 = F.log_softmax(logit1/T, dim=1)
    log_prob2 = F.log_softmax(logit2/T, dim=1)
    jsd = (prob1 * (log_prob1 - logsoftmax)).sum() / logit1.size(0)
    jsd += (prob2 * (log_prob2 - logsoftmax)).sum() / logit2.size(0)
    return jsd * 0.5


def _require_finite(value, description):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'Non-finite {description} in official RAAT')


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):
    # The driver selects this kernel explicitly; inherited exploratory flags
    # must never silently turn the official baseline into an AWP/shared variant.
    if os.environ.get('AWP_TRAIN', '0') != '0' or float(os.environ.get('AWP_GAMMA', '0')) != 0:
        raise ValueError('Official RAAT has no adversarial weight perturbation')
    if os.environ.get('SAAR_ATTACK_VIEWS', 'independent') != 'independent':
        raise ValueError('Official RAAT independently attacks both augmented views')
    for name in ('BD_alpha', 'T'):
        value = float(getattr(P, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Official RAAT requires finite positive {name}')
    if not math.isfinite(float(P.lam)) or P.lam < 0:
        raise ValueError('Official RAAT requires a finite nonnegative alignment coefficient')
    if not math.isfinite(float(P.BD_boundary_range)) or not 0 <= P.BD_boundary_range <= 1:
        raise ValueError('Official RAAT boundary scaling must be in [0, 1]')
    device = next(model.parameters()).device

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = dict()
    losses['cls'] = AverageMeter()
    losses['con'] = AverageMeter()
    check = time.time()
    batches = 0
    log_('[RAAT] baseline=raat_official attack_views=independent '
         'alignment_selection=clean_correct boundary_handling=official weight_perturbation=none')
    for n, (images, labels) in enumerate(loader):
        batches += 1
        model.train()
        count = n * P.n_gpus
        data_time.update(time.time() - check)
        check = time.time()
        batch_size = images[0].size(0)
        labels = labels.to(device)

        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_pair = torch.cat([images_aug1, images_aug2], dim=0)
        # One 2B call creates a separate optimized perturbation for each view.
        images_adv = adversary(images_pair, labels.repeat(2))
        _require_finite(images_adv, 'adversarial input')

        model.eval()
        eval_output_benign = model(images_pair)
        _require_finite(eval_output_benign, 'clean screening logits')
        eval_predicted_labels_benign = torch.argmax(eval_output_benign, 1).cpu().data.numpy()
        eval_output_adv = model(images_adv)
        _require_finite(eval_output_adv, 'adversarial screening logits')
        eval_predicted_labels_adv = torch.argmax(eval_output_adv, 1).cpu().data.numpy()
        adv_perturb = images_adv.clone().detach() - images_pair.clone().detach()
        adv_input_reduced = images_pair.clone().detach() + P.BD_boundary_range * adv_perturb
        eval_output_adv_reduced = model(adv_input_reduced)
        _require_finite(eval_output_adv_reduced, 'reduced-adversarial screening logits')
        eval_predicted_labels_adv_reduced = torch.argmax(eval_output_adv_reduced, 1).cpu().data.numpy()
        model.train()

        labels_np = deepcopy(labels.repeat(2)).cpu().numpy()
        all_index = np.array([i for i in range(len(labels_np))])
        misclassified_index = all_index[eval_predicted_labels_benign != labels_np]
        boundary_index = all_index[(eval_predicted_labels_benign == labels_np)
                                   & (eval_predicted_labels_adv != labels_np)
                                   & (eval_predicted_labels_adv_reduced != labels_np)]
        non_boundary_index = np.sort(
            np.array(list(set(list(all_index)) - set(list(misclassified_index)) - set(list(boundary_index)))))

        data_ce = torch.cat((
            images_adv[non_boundary_index], adv_input_reduced[boundary_index], images_adv[misclassified_index]))
        labels_ce = torch.cat((
            labels.repeat(2)[non_boundary_index], labels.repeat(2)[boundary_index], labels.repeat(2)[misclassified_index]))
        outputs_ce = model(data_ce)
        loss_ce = criterion(outputs_ce, labels_ce)
        loss = loss_ce

        ori_train_len = int(len(labels_np) / 2)
        ori_all_index = [idx for idx in range(ori_train_len)]
        misclassified_index_1 = [idx for idx in misclassified_index if idx < ori_train_len]
        misclassified_index_2 = [idx - ori_train_len for idx in misclassified_index if idx >= ori_train_len]
        ori_misclassified_index_set = set(misclassified_index_1) | set(misclassified_index_2)
        ICT_index = np.sort(np.array(list(set(ori_all_index) - ori_misclassified_index_set)))

        if len(ICT_index) > 0:
            _lambda = np.random.beta(P.BD_alpha, P.BD_alpha)
            mixup_rate = max(_lambda, (1 - _lambda))
            images_adv1, images_adv2 = images_adv.chunk(2)
            ICT_data = (mixup_rate * images_adv1[ICT_index] + (1 - mixup_rate) * images_adv2[ICT_index])
            ICT_output_1 = model(ICT_data)
            outputs_benign = model(images_pair)
            outputs_benign1, outputs_benign2 = outputs_benign.chunk(2)
            ICT_output_2 = (mixup_rate * outputs_benign1[ICT_index] + (1 - mixup_rate) * outputs_benign2[ICT_index])
            loss_con = P.lam * _jensen_shannon_div(ICT_output_1, ICT_output_2, P.T)
            loss += loss_con

        _require_finite(loss, 'training loss')
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                _require_finite(parameter.grad, f'gradient {name}')
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        batch_time.update(time.time() - check)
        losses['cls'].update(loss_ce.item(), batch_size)
        if len(ICT_index) > 0:
            losses['con'].update(loss_con.item(), batch_size)
        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossCon %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['con'].value))
        check = time.time()

    if not batches:
        raise ValueError('Official RAAT received an empty training loader')
    if P.optimizer == 'sgd':
        scheduler.step()
    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossCon %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['con'].average))
    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_con', losses['con'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

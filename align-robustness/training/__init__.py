import torch
import torch.nn.functional as F


def _safe_logits(x):
    return torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


def _kl_div(logit1, logit2):
    logit1 = _safe_logits(logit1)
    logit2 = _safe_logits(logit2)
    return F.kl_div(
        F.log_softmax(logit1, dim=1),
        F.softmax(logit2, dim=1),
        reduction='batchmean'
    )


def _jensen_shannon_div(logit1, logit2, T=1.):
    T = float(T)
    logit1 = _safe_logits(logit1 / T)
    logit2 = _safe_logits(logit2 / T)

    log_prob1 = F.log_softmax(logit1, dim=1)
    log_prob2 = F.log_softmax(logit2, dim=1)
    prob1 = log_prob1.exp()
    prob2 = log_prob2.exp()

    mean_prob = 0.5 * (prob1 + prob2)
    mean_prob = mean_prob.clamp(min=1e-8)
    log_mean_prob = mean_prob.log()

    jsd = F.kl_div(log_mean_prob, prob1, reduction='batchmean')
    jsd += F.kl_div(log_mean_prob, prob2, reduction='batchmean')
    jsd = jsd * 0.5

    return torch.nan_to_num(jsd, nan=0.0, posinf=1e4, neginf=0.0)


def _jensen_shannon_div_without_reduction(logit1, logit2, T=1.):
    T = float(T)
    logit1 = _safe_logits(logit1 / T)
    logit2 = _safe_logits(logit2 / T)

    log_prob1 = F.log_softmax(logit1, dim=1)
    log_prob2 = F.log_softmax(logit2, dim=1)
    prob1 = log_prob1.exp()
    prob2 = log_prob2.exp()

    mean_prob = 0.5 * (prob1 + prob2)
    mean_prob = mean_prob.clamp(min=1e-8)
    log_mean_prob = mean_prob.log()

    jsd = F.kl_div(log_mean_prob, prob1, reduction='none')
    jsd += F.kl_div(log_mean_prob, prob2, reduction='none')
    jsd = jsd * 0.5

    return torch.nan_to_num(jsd, nan=0.0, posinf=1e4, neginf=0.0)

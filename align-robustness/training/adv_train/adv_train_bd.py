import os
import torch
import torch.nn.functional as F
import time
import torch.optim
from training import _jensen_shannon_div
from utils.utils import AverageMeter

import numpy as np
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):

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
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images[0].size(0)
        labels = labels.to(device)

        # --- Raw-Adv-Aug Cheap-DICAR ---
        # New idea:
        #   1) generate adversarial example on raw image first
        #   2) apply the same random crop/flip params to raw clean and raw adv
        #   3) obtain two aligned clean/adv augmented views
        #
        # This avoids sharing a perturbation generated from view1 directly to view2.
        if os.environ.get("RAW_ADV_AUG_DICAR", "0") == "1":
            raw_images = images[0].to(device)

            def _paired_crop_flip(clean_x, adv_x, padding=4):
                B, C, H, W = clean_x.shape

                clean_pad = F.pad(clean_x, (padding, padding, padding, padding))
                adv_pad = F.pad(adv_x, (padding, padding, padding, padding))

                max_off = 2 * padding
                top = torch.randint(0, max_off + 1, (B,), device=clean_x.device)
                left = torch.randint(0, max_off + 1, (B,), device=clean_x.device)

                clean_out = torch.empty_like(clean_x)
                adv_out = torch.empty_like(adv_x)

                for i in range(B):
                    t = int(top[i].item())
                    l = int(left[i].item())
                    clean_out[i] = clean_pad[i, :, t:t+H, l:l+W]
                    adv_out[i] = adv_pad[i, :, t:t+H, l:l+W]

                flip_mask = torch.rand(B, device=clean_x.device) < 0.5
                if flip_mask.any():
                    clean_out[flip_mask] = torch.flip(clean_out[flip_mask], dims=[3])
                    adv_out[flip_mask] = torch.flip(adv_out[flip_mask], dims=[3])

                return clean_out, adv_out

            raw_adv = adversary(raw_images, labels).detach()

            images_aug1, images_adv1 = _paired_crop_flip(raw_images.detach(), raw_adv, padding=4)
            images_aug2, images_adv2 = _paired_crop_flip(raw_images.detach(), raw_adv, padding=4)

            images_pair = torch.cat([images_aug1, images_aug2], dim=0)
            images_adv = torch.cat([images_adv1, images_adv2], dim=0)

        else:
            images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
            images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B

            # --- Cheap -> RAAT Curriculum ---
            # Epoch 1..switch: original Cheap-DICAR
            # Epoch switch+1..end: full RAAT-style adversarial training on both views
            # --- Parameter-Fragile RAAT-DICAR ---
            # RAAT-style adversarial examples for both augmented views:
            #   x1_adv = Adv(x1), x2_adv = Adv(x2)
            #
            # Boundary screening is replaced later by parameter-fragility screening.
            if os.environ.get("PARAM_FRAGILE_RAAT_DICAR", "0") == "1":
                images_adv = adversary(images_pair, labels.repeat(2))
                images_adv1, images_adv2 = images_adv.chunk(2)

            # --- Cross-RAAT DICAR ---
            # Generate adversarial examples for both augmented views, as in RAAT:
            #   x1_adv = Adv(x1), x2_adv = Adv(x2)
            #
            # The ICT loss below will use cross interpolation:
            #   z1 = lambda*x1 + (1-lambda)*x2_adv
            #   z2 = lambda*x1_adv + (1-lambda)*x2
            elif os.environ.get("CROSS_RAAT_DICAR", "0") == "1":
                images_adv = adversary(images_pair, labels.repeat(2))
                images_adv1, images_adv2 = images_adv.chunk(2)

            # --- Batch-Alternating Cheap/RAAT ---
            # For each training batch n:
            #   if n % k == r: use RAAT
            #   otherwise: use Cheap-DICAR
            #
            # Default k=2, r=0:
            #   even batches use RAAT, odd batches use Cheap-DICAR.
            elif os.environ.get("ALTERNATE_CHEAP_RAAT", "0") == "1":
                alt_k = int(os.environ.get("ALTERNATE_CHEAP_RAAT_K", "2"))
                alt_raat_mod = int(os.environ.get("ALTERNATE_CHEAP_RAAT_RAAT_MOD", "0"))

                use_raat = (n % alt_k == alt_raat_mod)

                if use_raat:
                    # RAAT branch: attack both augmented views directly.
                    images_adv = adversary(images_pair, labels.repeat(2))
                    images_adv1, images_adv2 = images_adv.chunk(2)
                else:
                    # Cheap-DICAR branch: attack view1 only, reuse delta for view2.
                    images_adv1 = adversary(images_aug1, labels)
                    adv_perturb1 = images_adv1.clone().detach() - images_aug1.clone().detach()
                    images_adv2 = torch.clamp(
                        images_aug2.clone().detach() + adv_perturb1.detach(),
                        0.0, 1.0
                    )
                    images_adv = torch.cat([images_adv1, images_adv2], dim=0)

            elif os.environ.get("HYBRID_CHEAP_RAAT", "0") == "1":
                switch_epoch = int(os.environ.get("HYBRID_CHEAP_RAAT_SWITCH", "50"))

                if epoch <= switch_epoch:
                    # Cheap-DICAR phase: attack view1 only, reuse delta for view2.
                    images_adv1 = adversary(images_aug1, labels)
                    adv_perturb1 = images_adv1.clone().detach() - images_aug1.clone().detach()
                    images_adv2 = torch.clamp(
                        images_aug2.clone().detach() + adv_perturb1.detach(),
                        0.0, 1.0
                    )
                    images_adv = torch.cat([images_adv1, images_adv2], dim=0)

                else:
                    # RAAT phase: attack both augmented views directly.
                    images_adv = adversary(images_pair, labels.repeat(2))
                    images_adv1, images_adv2 = images_adv.chunk(2)

            # --- Shared-Delta DICAR ---
            # Learn one shared perturbation delta for both augmented views:
            #   delta = argmax CE(f(x1+delta), y) + CE(f(x2+delta), y)
            #
            # Then apply the same delta to both views.
            elif os.environ.get("SHARED_DELTA_DICAR", "0") == "1":
                if P.distance != "Linf":
                    raise NotImplementedError("SHARED_DELTA_DICAR currently supports Linf only.")

                x1 = images_aug1.detach()
                x2 = images_aug2.detach()
                y = labels

                eps = float(P.epsilon)
                alpha = float(P.alpha)
                n_iters = int(P.n_iters)

                # One shared delta for both views.
                delta = torch.empty_like(x1).uniform_(-eps, eps)

                # Keep x1+delta and x2+delta both inside [0, 1].
                lower = torch.maximum(-x1, -x2)
                upper = torch.minimum(1.0 - x1, 1.0 - x2)
                delta = torch.max(torch.min(delta, upper), lower).detach()

                for _ in range(n_iters):
                    delta.requires_grad_(True)

                    adv1 = torch.clamp(x1 + delta, 0.0, 1.0)
                    adv2 = torch.clamp(x2 + delta, 0.0, 1.0)

                    logits = model(torch.cat([adv1, adv2], dim=0))
                    loss_attack = F.cross_entropy(
                        logits,
                        torch.cat([y, y], dim=0),
                        reduction="mean"
                    )

                    grad = torch.autograd.grad(
                        loss_attack,
                        delta,
                        only_inputs=True,
                        retain_graph=False,
                        create_graph=False
                    )[0]

                    delta = delta.detach() + alpha * torch.sign(grad)
                    delta = torch.clamp(delta, -eps, eps)
                    delta = torch.max(torch.min(delta, upper), lower).detach()

                images_adv1 = torch.clamp(images_aug1.clone().detach() + delta, 0.0, 1.0)
                images_adv2 = torch.clamp(images_aug2.clone().detach() + delta, 0.0, 1.0)
                images_adv = torch.cat([images_adv1, images_adv2], dim=0)

            # --- Midpoint-Adv DICAR ---
            # Generate one adversarial perturbation on the midpoint of two augmented views,
            # then apply this midpoint perturbation to both endpoints.
            elif os.environ.get("MIDPOINT_ADV_DICAR", "0") == "1":
                x_mid = (0.5 * images_aug1 + 0.5 * images_aug2).detach()
                x_mid_adv = adversary(x_mid, labels)
                delta_mid = (x_mid_adv.clone().detach() - x_mid.clone().detach()).detach()

                images_adv1 = torch.clamp(images_aug1.clone().detach() + delta_mid, 0.0, 1.0)
                images_adv2 = torch.clamp(images_aug2.clone().detach() + delta_mid, 0.0, 1.0)

                images_adv = torch.cat([images_adv1, images_adv2], dim=0)

            else:
                # --- Cheap DICAR: reuse perturbation from view1 for view2 --- #
                # Generate adversarial perturbation only for the first augmented view.
                images_adv1 = adversary(images_aug1, labels)
                adv_perturb1 = images_adv1.clone().detach() - images_aug1.clone().detach()

                # Reuse the same detached perturbation for the second augmented view.
                images_adv2 = torch.clamp(images_aug2.clone().detach() + adv_perturb1.detach(), 0.0, 1.0)

                # Keep the rest of RAAT unchanged: boundary treatment, CE loss, and ICT/DICAR.
                images_adv = torch.cat([images_adv1, images_adv2], dim=0)

        # --- divide non-boundary, boundary and misclassified samples --- #
        model.eval()

        eval_output_benign = model(images_pair)
        eval_predicted_labels_benign = torch.argmax(eval_output_benign, 1).cpu().data.numpy()

        adv_perturb = images_adv.clone().detach() - images_pair.clone().detach()
        adv_input_reduced = images_pair.clone().detach() + P.BD_boundary_range * adv_perturb

        labels_np = deepcopy(labels.repeat(2)).cpu().numpy()
        all_index = np.array([i for i in range(len(labels_np))])
        misclassified_index = all_index[eval_predicted_labels_benign != labels_np]

        if os.environ.get("PARAM_FRAGILE_SCREEN", "0") == "1":
            # Batch-level AWP-style parameter perturbation:
            #   Delta W_l = rho * ||W_l|| / ||grad_l|| * grad_l
            rho = float(os.environ.get("PARAM_FRAGILE_RHO", "0.005"))
            eps_norm = 1e-12

            x_pf = images_pair.detach()
            y_pf = labels.repeat(2)

            model.zero_grad()
            logits_pf = model(x_pf)
            loss_pf = F.cross_entropy(logits_pf, y_pf)

            named_params = [
                (name, p) for name, p in model.named_parameters()
                if p.requires_grad and p.dim() > 1
            ]
            params = [p for _, p in named_params]

            grads = torch.autograd.grad(
                loss_pf,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

            perturb_list = []
            with torch.no_grad():
                for (name, p), g in zip(named_params, grads):
                    if g is None:
                        perturb_list.append((p, None))
                        continue

                    scale = rho * (p.detach().norm() / (g.detach().norm() + eps_norm))
                    e_w = scale * g.detach()
                    p.add_(e_w)
                    perturb_list.append((p, e_w))

                eval_output_param = model(images_pair.detach())
                eval_predicted_labels_param = torch.argmax(
                    eval_output_param, 1
                ).cpu().data.numpy()

                # Restore original parameters immediately.
                for p, e_w in perturb_list:
                    if e_w is not None:
                        p.sub_(e_w)

            boundary_index = all_index[
                (eval_predicted_labels_benign == labels_np)
                & (eval_predicted_labels_param != labels_np)
            ]

        else:
            eval_output_adv = model(images_adv)
            eval_predicted_labels_adv = torch.argmax(eval_output_adv, 1).cpu().data.numpy()

            eval_output_adv_reduced = model(adv_input_reduced)
            eval_predicted_labels_adv_reduced = torch.argmax(
                eval_output_adv_reduced, 1
            ).cpu().data.numpy()

            boundary_index = all_index[
                (eval_predicted_labels_benign == labels_np)
                & (eval_predicted_labels_adv != labels_np)
                & (eval_predicted_labels_adv_reduced != labels_np)
            ]

        # --- Optional true per-sample Param-Fragile screening ---
        # Training AWP is unchanged.
        # This only changes data selection:
        #   batch-level screening: one Delta W for the whole batch
        #   per-sample screening: each sample gets its own Delta W_i
        if (
            os.environ.get("PARAM_FRAGILE_SCREEN", "0") == "1"
            and os.environ.get("PARAM_FRAGILE_PER_SAMPLE", "0") == "1"
        ):
            rho_ps = float(os.environ.get("PARAM_FRAGILE_RHO", "0.002"))
            eps_norm_ps = 1e-12

            named_params_ps = [
                (name, param) for name, param in model.named_parameters()
                if param.requires_grad and param.dim() > 1
            ]
            params_ps = [param for _, param in named_params_ps]

            eval_predicted_labels_param_ps = eval_predicted_labels_benign.copy()

            # Only clean-correct samples need fragile screening.
            # Clean-wrong samples are already in misclassified_index.
            candidate_indices = [
                int(idx) for idx in all_index
                if eval_predicted_labels_benign[int(idx)] == labels_np[int(idx)]
            ]

            # For debugging only. 0 means full per-sample screening.
            ps_max = int(os.environ.get("PARAM_FRAGILE_PS_MAX", "0"))
            if ps_max > 0:
                candidate_indices = candidate_indices[:ps_max]

            for idx_ps in candidate_indices:
                x_one = images_pair[idx_ps:idx_ps + 1].detach()
                y_one = labels.repeat(2)[idx_ps:idx_ps + 1]

                model.zero_grad()
                logits_one = model(x_one)
                loss_one = F.cross_entropy(logits_one, y_one)

                grads_ps = torch.autograd.grad(
                    loss_one,
                    params_ps,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )

                perturb_list_ps = []

                with torch.no_grad():
                    for (name, param), grad in zip(named_params_ps, grads_ps):
                        if grad is None:
                            perturb_list_ps.append((param, None))
                            continue

                        scale = rho_ps * (
                            param.detach().norm()
                            / (grad.detach().norm() + eps_norm_ps)
                        )
                        e_w = scale * grad.detach()
                        param.add_(e_w)
                        perturb_list_ps.append((param, e_w))

                    pred_one = torch.argmax(model(x_one), dim=1).item()
                    eval_predicted_labels_param_ps[idx_ps] = int(pred_one)

                    for param, e_w in perturb_list_ps:
                        if e_w is not None:
                            param.sub_(e_w)

            boundary_index = all_index[
                (eval_predicted_labels_benign == labels_np)
                & (eval_predicted_labels_param_ps != labels_np)
            ]

            if n % 50 == 0:
                log_(
                    f"[ParamFragilePerSample] mis={len(misclassified_index)} "
                    f"fragile={len(boundary_index)} "
                    f"non={len(set(list(all_index)) - set(list(misclassified_index)) - set(list(boundary_index)))} "
                    f"rho={rho_ps} clean_keep={int(os.environ.get('PARAM_FRAGILE_CLEAN_KEEP', '0'))} "
                    f"ps_max={ps_max}"
                )

        non_boundary_index = np.sort(
            np.array(list(set(list(all_index)) - set(list(misclassified_index)) - set(list(boundary_index))))
        )

        if os.environ.get("PARAM_FRAGILE_SCREEN", "0") == "1" and n % 50 == 0:
            log_(f"[ParamFragile] mis={len(misclassified_index)} fragile={len(boundary_index)} non={len(non_boundary_index)} rho={float(os.environ.get('PARAM_FRAGILE_RHO', '0.005'))} clean_keep={int(os.environ.get('PARAM_FRAGILE_CLEAN_KEEP', '0'))}")

        model.train()

        # reconstruct supervised set for CE loss
        #
        # Default:
        #   non-fragile samples        -> images_adv
        #   boundary / fragile samples -> adv_input_reduced
        #   clean-misclassified        -> images_adv
        #
        # PF-CleanKeep:
        #   if a sample becomes wrong after parameter perturbation W+dW,
        #   it is already hard in parameter space.
        #   Therefore, keep its clean input instead of adding input perturbation.
        if (
            os.environ.get("PARAM_FRAGILE_SCREEN", "0") == "1"
            and os.environ.get("PARAM_FRAGILE_CLEAN_KEEP", "0") == "1"
        ):
            boundary_data_ce = images_pair[boundary_index]
        else:
            boundary_data_ce = adv_input_reduced[boundary_index]

        data_ce = torch.cat((
            images_adv[non_boundary_index],
            boundary_data_ce,
            images_adv[misclassified_index]
        ))

        labels_ce = torch.cat((
            labels.repeat(2)[non_boundary_index],
            labels.repeat(2)[boundary_index],
            labels.repeat(2)[misclassified_index]
        ))

        outputs_ce = model(data_ce)
        loss_ce = criterion(outputs_ce, labels_ce)
        loss = loss_ce

        ### ICT regularization ###
        ori_train_len = int(len(labels_np) / 2)
        ori_all_index = [idx for idx in range(ori_train_len)]
        misclassified_index_1 = [idx for idx in misclassified_index if idx < ori_train_len]
        misclassified_index_2 = [idx - ori_train_len for idx in misclassified_index if idx >= ori_train_len]
        ori_misclassified_index_set = set(misclassified_index_1) | set(misclassified_index_2)
        if os.environ.get("ALIGN_ALL_SAMPLES", "0") == "1":
            # No-selection ablation:
            # use all original samples for ICT/DICAR alignment,
            # including samples that would previously be filtered out.
            ICT_index = np.array(ori_all_index, dtype=np.int64)
        else:
            ICT_index = np.sort(
                np.array(list(set(ori_all_index) - ori_misclassified_index_set), dtype=np.int64)
            )

        if len(ICT_index) > 0:
            _lambda = np.random.beta(P.BD_alpha, P.BD_alpha)
            mixup_rate = max(_lambda, (1 - _lambda))
            images_adv1, images_adv2 = images_adv.chunk(2)
            # --- Hard-anchor Cheap-DICAR ---
            # Instead of always using the first view as the perturbation anchor,
            # choose the more adversarially informative endpoint.
            #
            # Default selection rule:
            #   anchor = argmax_{v in {x1, x2}} CE(f(v), y)
            #
            # Then reuse the already generated endpoint perturbation:
            #   delta_a = x_adv_anchor - x_anchor
            #
            # DICAR input:
            #   x_mid + delta_a, where x_mid = 0.5*x1 + 0.5*x2
            if os.environ.get("CROSS_RAAT_DICAR", "0") == "1":
                x1_clean = images_aug1[ICT_index].detach()
                x2_clean = images_aug2[ICT_index].detach()
                x1_adv = images_adv1[ICT_index].detach()
                x2_adv = images_adv2[ICT_index].detach()

                # Cross interpolation:
                #   z1(lambda) = lambda*x1_clean + (1-lambda)*x2_adv
                #   z2(lambda) = lambda*x1_adv   + (1-lambda)*x2_clean
                z1 = (
                    mixup_rate * x1_clean
                    + (1.0 - mixup_rate) * x2_adv
                ).detach()

                z2 = (
                    mixup_rate * x1_adv
                    + (1.0 - mixup_rate) * x2_clean
                ).detach()

                # Clean semantic interpolation target:
                #   target = lambda*p(x1_clean) + (1-lambda)*p(x2_clean)
                outputs_benign = model(images_pair)
                outputs_benign1, outputs_benign2 = outputs_benign.chunk(2)

                with torch.no_grad():
                    p1 = F.softmax(outputs_benign1[ICT_index] / P.T, dim=1)
                    p2 = F.softmax(outputs_benign2[ICT_index] / P.T, dim=1)
                    p_target = (
                        mixup_rate * p1
                        + (1.0 - mixup_rate) * p2
                    ).detach()

                logits_z = model(torch.cat([z1, z2], dim=0))
                logits_z1, logits_z2 = logits_z.chunk(2)

                def _soft_ce(logits, target_prob):
                    return -(
                        target_prob
                        * F.log_softmax(logits / P.T, dim=1)
                    ).sum(dim=1).mean()

                loss_con = 0.5 * P.lam * (
                    _soft_ce(logits_z1, p_target)
                    + _soft_ce(logits_z2, p_target)
                )
                loss += loss_con

            elif os.environ.get("HARD_ANCHOR_DICAR", "0") == "1":
                ha_mode = os.environ.get("HARD_ANCHOR_MODE", "ce")
                ha_freeze = os.environ.get("HARD_ANCHOR_FREEZE", "0") == "1"

                x1_clean = images_aug1[ICT_index].detach()
                x2_clean = images_aug2[ICT_index].detach()
                y_ict = labels[ICT_index]

                x_mid = (0.5 * x1_clean + 0.5 * x2_clean).detach()

                def _ha_js_prob_loss(logits, target_prob):
                    q = F.softmax(logits, dim=1)
                    t = target_prob.detach()
                    m = 0.5 * (q + t)

                    eps = 1e-12
                    log_q = torch.log(torch.clamp(q, min=eps))
                    log_t = torch.log(torch.clamp(t, min=eps))
                    log_m = torch.log(torch.clamp(m, min=eps))

                    js = (
                        0.5 * torch.sum(q * (log_q - log_m), dim=1)
                        + 0.5 * torch.sum(t * (log_t - log_m), dim=1)
                    )
                    return js.mean()

                with torch.no_grad():
                    clean_pair = torch.cat([x1_clean, x2_clean], dim=0)

                    # Current model is always used to build the DICAR soft target.
                    logits_clean_pair = model(clean_pair)
                    logits1, logits2 = logits_clean_pair.chunk(2)

                    p1 = F.softmax(logits1, dim=1)
                    p2 = F.softmax(logits2, dim=1)
                    p_mid = (0.5 * p1 + 0.5 * p2).detach()

                    # Anchor decision can be dynamic or frozen.
                    # frozen: use a snapshot model copied at the first call.
                    if ha_freeze:
                        import copy as _copy
                        _g = globals()
                        if "_HARD_ANCHOR_FROZEN_MODEL" not in _g:
                            _g["_HARD_ANCHOR_FROZEN_MODEL"] = _copy.deepcopy(model).eval()
                            for _p in _g["_HARD_ANCHOR_FROZEN_MODEL"].parameters():
                                _p.requires_grad_(False)

                        ha_model = _g["_HARD_ANCHOR_FROZEN_MODEL"]
                        logits_anchor_pair = ha_model(clean_pair)
                        logits1_anchor, logits2_anchor = logits_anchor_pair.chunk(2)
                    else:
                        logits1_anchor, logits2_anchor = logits1, logits2

                    if ha_mode in ["margin", "easy_margin"]:
                        z1_y = logits1_anchor.gather(1, y_ict.view(-1, 1)).squeeze(1)
                        z2_y = logits2_anchor.gather(1, y_ict.view(-1, 1)).squeeze(1)

                        mask1 = torch.ones_like(logits1_anchor, dtype=torch.bool)
                        mask2 = torch.ones_like(logits2_anchor, dtype=torch.bool)
                        mask1.scatter_(1, y_ict.view(-1, 1), False)
                        mask2.scatter_(1, y_ict.view(-1, 1), False)

                        z1_other = logits1_anchor.masked_fill(~mask1, -1e9).max(dim=1)[0]
                        z2_other = logits2_anchor.masked_fill(~mask2, -1e9).max(dim=1)[0]

                        margin1 = z1_y - z1_other
                        margin2 = z2_y - z2_other

                        if ha_mode == "easy_margin":
                            # larger margin = easier endpoint
                            choose_x1 = margin1 >= margin2
                        else:
                            # smaller margin = harder endpoint
                            choose_x1 = margin1 <= margin2
                    else:
                        ce1 = F.cross_entropy(logits1_anchor, y_ict, reduction="none")
                        ce2 = F.cross_entropy(logits2_anchor, y_ict, reduction="none")

                        if ha_mode in ["easy_ce", "easy"]:
                            # smaller CE = easier endpoint
                            choose_x1 = ce1 <= ce2
                        else:
                            # larger CE = harder endpoint
                            choose_x1 = ce1 >= ce2


                delta1 = (images_adv1[ICT_index].detach() - x1_clean).detach()
                delta2 = (images_adv2[ICT_index].detach() - x2_clean).detach()

                anchor_mask = choose_x1.view(-1, 1, 1, 1)
                delta_a = torch.where(anchor_mask, delta1, delta2).detach()

                ICT_data = torch.clamp(x_mid + delta_a, 0.0, 1.0)
                ICT_output_1 = model(ICT_data)

                loss_con = P.lam * _ha_js_prob_loss(ICT_output_1, p_mid)
                loss += loss_con

            # --- WS-RO-DICAR: Warm-Started Robust-Objective DICAR ---
            # Reuse one randomly selected endpoint perturbation from the
            # main adversarial training branch as the initialization.
            # Perform exactly one refinement step on the interpolation input
            # using CE + gamma * JS as the attack objective.
            #
            # No additional forward/backward is introduced compared with
            # the existing one-step PC/EA-DICAR branch.
            elif os.environ.get("WS_RO_DICAR", "0") == "1":
                ws_steps = int(os.environ.get("WS_RO_STEPS", "1"))
                if ws_steps != 1:
                    raise ValueError("WS-RO-DICAR is designed with WS_RO_STEPS=1.")

                ws_ratio = float(os.environ.get("WS_RO_RD", "1.0"))
                ws_alpha = float(os.environ.get("WS_RO_ALPHA", str(P.alpha)))
                ws_gamma = float(os.environ.get("WS_RO_GAMMA", "1.0"))
                ws_eps = float(P.epsilon) * ws_ratio

                x1_clean = images_aug1[ICT_index].detach()
                x2_clean = images_aug2[ICT_index].detach()
                y_ict = labels[ICT_index]

                x_lam = (
                    mixup_rate * x1_clean
                    + (1 - mixup_rate) * x2_clean
                ).detach()

                def _ws_js_prob_loss(logits, target_prob):
                    q = F.softmax(logits, dim=1)
                    t = target_prob.detach()
                    m = 0.5 * (q + t)

                    eps = 1e-12
                    log_q = torch.log(torch.clamp(q, min=eps))
                    log_t = torch.log(torch.clamp(t, min=eps))
                    log_m = torch.log(torch.clamp(m, min=eps))

                    js = (
                        0.5 * torch.sum(q * (log_q - log_m), dim=1)
                        + 0.5 * torch.sum(t * (log_t - log_m), dim=1)
                    )
                    return js.mean()

                with torch.no_grad():
                    clean_pair = torch.cat([x1_clean, x2_clean], dim=0)
                    p_clean_pair = F.softmax(model(clean_pair), dim=1)
                    p1, p2 = p_clean_pair.chunk(2)

                    p_lam = (
                        mixup_rate * p1
                        + (1 - mixup_rate) * p2
                    ).detach()

                # Endpoint perturbations already generated by the main branch
                delta1 = (
                    images_adv1[ICT_index].detach() - x1_clean
                ).detach()

                delta2 = (
                    images_adv2[ICT_index].detach() - x2_clean
                ).detach()

                # Random endpoint selection preserves symmetry
                anchor_mask = (
                    torch.rand(
                        x1_clean.size(0), 1, 1, 1,
                        device=x1_clean.device
                    ) < 0.5
                )

                delta = torch.where(anchor_mask, delta1, delta2).detach()

                # Transfer the endpoint perturbation to the interpolation input
                delta = torch.clamp(delta, -ws_eps, ws_eps)
                delta = torch.clamp(x_lam + delta, 0.0, 1.0) - x_lam
                delta = delta.detach()

                # Exactly one refinement step: CE encourages boundary pressure,
                # JS preserves the DICAR semantic-alignment objective.
                delta.requires_grad_()

                logits_attack = model(torch.clamp(x_lam + delta, 0.0, 1.0))

                loss_attack = (
                    F.cross_entropy(logits_attack, y_ict)
                    + ws_gamma * _ws_js_prob_loss(logits_attack, p_lam)
                )

                grad = torch.autograd.grad(
                    loss_attack, delta, only_inputs=True
                )[0]

                delta = delta.detach() + ws_alpha * torch.sign(grad.detach())
                delta = torch.clamp(delta, -ws_eps, ws_eps)
                delta = torch.clamp(x_lam + delta, 0.0, 1.0) - x_lam
                delta = delta.detach()

                ICT_data = torch.clamp(x_lam + delta, 0.0, 1.0)
                ICT_output_1 = model(ICT_data)

                loss_con = P.lam * _ws_js_prob_loss(ICT_output_1, p_lam)
                loss += loss_con

            # --- WE-DICAR: Worst-Endpoint DICAR ---
            # Reuse the two endpoint perturbations already generated by
            # the main adversarial training branch. No extra attack is generated.
            #
            #   delta_1 = images_adv1 - x1_clean
            #   delta_2 = images_adv2 - x2_clean
            #   candidate_j = x_lam + delta_j
            #   loss = mean(max(JS(candidate_1, p_lam), JS(candidate_2, p_lam)))
            elif os.environ.get("WE_DICAR", "0") == "1":
                x1_clean = images_aug1[ICT_index].detach()
                x2_clean = images_aug2[ICT_index].detach()

                x_lam = (
                    mixup_rate * x1_clean
                    + (1 - mixup_rate) * x2_clean
                ).detach()

                with torch.no_grad():
                    clean_pair = torch.cat([x1_clean, x2_clean], dim=0)
                    p_clean_pair = F.softmax(model(clean_pair), dim=1)
                    p1, p2 = p_clean_pair.chunk(2)

                    p_lam = (
                        mixup_rate * p1
                        + (1 - mixup_rate) * p2
                    ).detach()

                # Reuse endpoint perturbations from the main adversarial branch
                delta1 = (
                    images_adv1[ICT_index].detach() - x1_clean
                ).detach()

                delta2 = (
                    images_adv2[ICT_index].detach() - x2_clean
                ).detach()

                candidate1 = torch.clamp(x_lam + delta1, 0.0, 1.0)
                candidate2 = torch.clamp(x_lam + delta2, 0.0, 1.0)

                # Evaluate both candidates in one batched forward pass
                candidate_pair = torch.cat([candidate1, candidate2], dim=0)
                logits_pair = model(candidate_pair)
                logits1, logits2 = logits_pair.chunk(2)

                def _we_js_per_sample(logits, target_prob):
                    q = F.softmax(logits, dim=1)
                    t = target_prob.detach()
                    m = 0.5 * (q + t)

                    eps = 1e-12
                    log_q = torch.log(torch.clamp(q, min=eps))
                    log_t = torch.log(torch.clamp(t, min=eps))
                    log_m = torch.log(torch.clamp(m, min=eps))

                    js = (
                        0.5 * torch.sum(q * (log_q - log_m), dim=1)
                        + 0.5 * torch.sum(t * (log_t - log_m), dim=1)
                    )
                    return js

                loss_endpoint1 = _we_js_per_sample(logits1, p_lam)
                loss_endpoint2 = _we_js_per_sample(logits2, p_lam)

                # Per-sample worst endpoint selection
                loss_con = P.lam * torch.maximum(
                    loss_endpoint1, loss_endpoint2
                ).mean()

                loss += loss_con

            # --- Fast PC-DICAR / EA-DICAR branch ---
            # FAST_PC_DICAR=1 enables the cheap DICAR branch.
            # EA_DICAR=1 switches from midpoint-generated perturbation to
            # Endpoint-Anchored DICAR:
            #   x_mid = 0.5*x1 + 0.5*x2
            #   p_mid = sg(0.5*p(x1) + 0.5*p(x2))
            #   x_anchor is randomly sampled from {x1, x2} per sample
            #   delta = Adv(x_anchor)
            #   loss = JS(p(x_mid + delta), p_mid)
            elif os.environ.get("FAST_PC_DICAR", "0") == "1":
                pc_ratio = float(os.environ.get("PC_DICAR_RD", "1.0"))
                pc_steps = int(os.environ.get("PC_DICAR_STEPS", "1"))
                pc_alpha = float(os.environ.get("PC_DICAR_ALPHA", str(P.alpha)))
                pc_eps = float(P.epsilon) * pc_ratio
                ea_dicar = os.environ.get("EA_DICAR", "0") == "1"
                ro_pc_dicar = os.environ.get("RO_PC_DICAR", "0") == "1"
                ro_pc_gamma = float(os.environ.get("RO_PC_GAMMA", "1.0"))

                x1_clean = images_aug1[ICT_index].detach()
                x2_clean = images_aug2[ICT_index].detach()
                y_ict = labels[ICT_index]

                def _pc_js_prob_loss(logits, target_prob):
                    q = F.softmax(logits, dim=1)
                    t = target_prob.detach()
                    m = 0.5 * (q + t)

                    eps = 1e-12
                    log_q = torch.log(torch.clamp(q, min=eps))
                    log_t = torch.log(torch.clamp(t, min=eps))
                    log_m = torch.log(torch.clamp(m, min=eps))

                    js = 0.5 * torch.sum(q * (log_q - log_m), dim=1) + \
                         0.5 * torch.sum(t * (log_t - log_m), dim=1)
                    return js.mean()

                with torch.no_grad():
                    clean_pair_for_target = torch.cat([x1_clean, x2_clean], dim=0)
                    p_clean_pair = F.softmax(model(clean_pair_for_target), dim=1)
                    p1, p2 = p_clean_pair.chunk(2)

                    if ea_dicar:
                        # fixed midpoint target
                        p_target = (0.5 * p1 + 0.5 * p2).detach()
                    else:
                        # original PC-DICAR target
                        p_target = (mixup_rate * p1 + (1 - mixup_rate) * p2).detach()

                if ea_dicar:
                    # fixed midpoint input
                    x_mid = (0.5 * x1_clean + 0.5 * x2_clean).detach()

                    # per-sample random endpoint anchor, probability 0.5 each
                    anchor_mask = (torch.rand(
                        x1_clean.size(0), 1, 1, 1,
                        device=x1_clean.device
                    ) < 0.5)

                    x_anchor = torch.where(anchor_mask, x1_clean, x2_clean).detach()

                    # generate perturbation around endpoint anchor
                    x_attack_base = x_anchor
                else:
                    # original PC-DICAR: generate perturbation around interpolation path
                    x_lam = (mixup_rate * x1_clean + (1 - mixup_rate) * x2_clean).detach()
                    x_attack_base = x_lam

                # initialize one perturbation only
                delta = torch.zeros_like(x_attack_base).uniform_(-pc_eps, pc_eps).detach()
                delta = torch.clamp(x_attack_base + delta, 0.0, 1.0) - x_attack_base

                for _ in range(pc_steps):
                    delta.requires_grad_()

                    # attack is generated at anchor for EA-DICAR,
                    # or at interpolation point for PC-DICAR
                    logits_pc = model(torch.clamp(x_attack_base + delta, 0.0, 1.0))
                    js_attack = _pc_js_prob_loss(logits_pc, p_target)

                    if ro_pc_dicar:
                        loss_pc_attack = (
                            F.cross_entropy(logits_pc, y_ict)
                            + ro_pc_gamma * js_attack
                        )
                    else:
                        loss_pc_attack = js_attack

                    grad = torch.autograd.grad(loss_pc_attack, delta, only_inputs=True)[0]

                    delta = delta.detach() + pc_alpha * torch.sign(grad.detach())
                    delta = torch.clamp(delta, -pc_eps, pc_eps)
                    delta = torch.clamp(x_attack_base + delta, 0.0, 1.0) - x_attack_base
                    delta = delta.detach()

                if ea_dicar:
                    # apply endpoint-generated perturbation to midpoint
                    ICT_data = torch.clamp(x_mid + delta, 0.0, 1.0)
                else:
                    ICT_data = torch.clamp(x_attack_base + delta, 0.0, 1.0)

                ICT_output_1 = model(ICT_data)

                js_con = _pc_js_prob_loss(ICT_output_1, p_target)

                if os.environ.get("HL_DICAR", "0") == "1":
                    # Hard-label anchored DICAR:
                    # keep the soft DICAR target, but add a small CE anchor
                    # to prevent consistency regularization from weakening class margin.
                    hl_eta = float(os.environ.get("HL_ETA", "0.25"))
                    loss_con = P.lam * (
                        js_con + hl_eta * F.cross_entropy(ICT_output_1, y_ict)
                    )
                else:
                    loss_con = P.lam * js_con

                loss += loss_con

            else:
                ICT_data = (mixup_rate * images_adv1[ICT_index] + (1 - mixup_rate) * images_adv2[ICT_index])
                ICT_output_1 = model(ICT_data)

                outputs_benign = model(images_pair)
                outputs_benign1, outputs_benign2 = outputs_benign.chunk(2)
                ICT_output_2 = (mixup_rate * outputs_benign1[ICT_index] + (1 - mixup_rate) * outputs_benign2[ICT_index])

                loss_con = P.lam * _jensen_shannon_div(ICT_output_1, ICT_output_2, P.T)
                loss += loss_con

        # --- AWP training branch ---
        # This is real adversarial weight perturbation during training.
        # Difference from Param-Fragile:
        #   Param-Fragile uses parameter perturbation only to screen fragile samples.
        #   AWP_TRAIN perturbs weights, recomputes the training loss under W+dW,
        #   backprops that perturbed loss, then restores W before optimizer.step().
        if os.environ.get("AWP_TRAIN", "0") == "1" and epoch >= int(os.environ.get("AWP_START", "10")):
            awp_gamma = float(os.environ.get("AWP_GAMMA", "0.005"))
            awp_eps = 1e-12

            # For this first experiment, only support the default RAAT/ICT branch.
            # Do not mix AWP with experimental DICAR branches in the first run.
            _unsupported_awp_flags = [
                "CROSS_RAAT_DICAR", "HARD_ANCHOR_DICAR", "WS_RO_DICAR",
                "WE_DICAR", "FAST_PC_DICAR", "EA_DICAR", "RO_PC_DICAR",
                "HL_DICAR", "ALTERNATE_CHEAP_RAAT",
                "HYBRID_CHEAP_RAAT", "SHARED_DELTA_DICAR", "MIDPOINT_ADV_DICAR",
                "RAW_ADV_AUG_DICAR"
            ]
            for _flag in _unsupported_awp_flags:
                if os.environ.get(_flag, "0") == "1":
                    raise RuntimeError(
                        "AWP_TRAIN first run only supports default RAAT branch; "
                        f"please disable {_flag}."
                    )

            # 1) Build batch-level worst-case weight perturbation from current loss.
            named_params = [
                (name, param) for name, param in model.named_parameters()
                if param.requires_grad and param.dim() > 1
            ]
            params = [param for _, param in named_params]

            optimizer.zero_grad()

            # AWP_OBJECTIVE controls which loss is used to generate weight perturbation.
            # full       : CE + RAAT alignment loss, current default
            # align_only : only RAAT alignment / consistency loss
            # ce_only    : only CE loss
            awp_objective = os.environ.get("AWP_OBJECTIVE", "full").lower()
            if awp_objective in ("align_only", "alignment_only", "con_only"):
                loss_awp_source = loss_con if (len(ICT_index) > 0 and loss_con is not None) else loss
            elif awp_objective in ("ce_only", "ce"):
                loss_awp_source = loss_ce
            else:
                loss_awp_source = loss

            grads = torch.autograd.grad(
                loss_awp_source,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

            perturb_list = []
            with torch.no_grad():
                for (name, param), grad in zip(named_params, grads):
                    if grad is None:
                        perturb_list.append((param, None))
                        continue
                    scale = awp_gamma * (param.detach().norm() / (grad.detach().norm() + awp_eps))
                    e_w = scale * grad.detach()
                    param.add_(e_w)
                    perturb_list.append((param, e_w))

            # 2) Recompute the default RAAT loss under perturbed weights W+dW.
            outputs_ce_awp = model(data_ce)
            loss_ce_awp = criterion(outputs_ce_awp, labels_ce)
            loss_awp = loss_ce_awp
            loss_con_awp = None

            if len(ICT_index) > 0:
                # Default RAAT ICT branch only.
                ICT_data_awp = (
                    mixup_rate * images_adv1[ICT_index]
                    + (1 - mixup_rate) * images_adv2[ICT_index]
                )
                ICT_output_1_awp = model(ICT_data_awp)

                outputs_benign_awp = model(images_pair)
                outputs_benign1_awp, outputs_benign2_awp = outputs_benign_awp.chunk(2)
                ICT_output_2_awp = (
                    mixup_rate * outputs_benign1_awp[ICT_index]
                    + (1 - mixup_rate) * outputs_benign2_awp[ICT_index]
                )

                loss_con_awp = P.lam * _jensen_shannon_div(
                    ICT_output_1_awp, ICT_output_2_awp, P.T
                )
                loss_awp = loss_awp + loss_con_awp

            # 3) Backprop perturbed loss, restore original weights, then update.
            optimizer.zero_grad()
            loss_awp.backward()

            with torch.no_grad():
                for param, e_w in perturb_list:
                    if e_w is not None:
                        param.sub_(e_w)

            optimizer.step()

            if n % 50 == 0:
                log_(f"[AWPTrain] objective={awp_objective} gamma={awp_gamma} start={int(os.environ.get('AWP_START', '10'))} loss_awp={loss_awp.item():.6f}")

        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_ce.item(), batch_size)
        if len(ICT_index) > 0:
            losses['con'].update(loss_con.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossCon %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['con'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossCon %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['con'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_con', losses['con'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

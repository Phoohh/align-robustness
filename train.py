import torch
import math
import hashlib
from pathlib import Path
from experiment import atomic_json

from utils.utils import Logger
from utils.utils import save_checkpoint, save_checkpoint_epoch

from common.train import *
from evals import test_classifier_adv


kwargs = {}
if 'adv' in P.mode:
    from training.adv_train import setup
    kwargs['adversary'] = adversary
else:
    from training.train import setup

train, fname = setup(P.mode, P)

if resume and P.output_dir is not None and Path(P.output_dir).resolve() != Path(P.resume_path).resolve():
    raise ValueError('Resume in the original output directory to retain its best checkpoint')
logger = Logger(fname, ask=not resume, output_dir=P.resume_path if resume else P.output_dir)
logger.log(P)
logger.log(model)
if selection_free_v2:
    logger.log('CHECKPOINT_SELECTION=' + ('fixed_last_epoch; no validation/test evaluation' if final_refit
                                         else 'highest PGD accuracy on the held-out training validation split'))

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    is_best = False
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.augment_type == 'autoaug_sche' and epoch > (P.epochs/2):
        train_loader = P.train_second_loader

    train(P, epoch, model, criterion, optimizer, scheduler, train_loader, logger=logger, **kwargs)
    model.eval()

    should_evaluate = (not final_refit and
                       (epoch % P.error_step == 0 or (selection_free_v2 and epoch in (1, P.epochs))))
    if should_evaluate:
        error = test_classifier_adv(P, model, selection_loader, epoch,
                                    adversary=adversary_t, logger=logger, ret='adv')

        if not math.isfinite(float(error)):
            raise RuntimeError('Nonfinite PGD evaluation error')
        is_best = (not (Path(logger.logdir) / 'best.model').exists()) or (best > error)
        if is_best:
            best = error
            best_epoch = epoch

        logger.scalar_summary('eval/best_adv_error', best, epoch)
        eval_label = 'Adv_Val' if selection_free_v2 else 'Adv_Test'
        logger.log('[Epoch %3d] [%s %5.2f] [Best %5.2f]' % (epoch, eval_label, error, best))

    if final_refit and epoch == P.epochs:
        # The epoch count is locked before this run, never selected on test data.
        is_best = True
        best_epoch = epoch

    save_states = model.state_dict()
    save_checkpoint(epoch, best, save_states, optimizer.state_dict(), logger.logdir, is_best)
    if selection_free_v2:
        atomic_json(Path(logger.logdir) / 'checkpoint_selection.json', {
            'experiment_protocol': P.experiment_protocol,
            'protocol_stage': P.protocol_stage,
            'criterion': 'fixed_last_epoch' if final_refit else 'validation_pgd_accuracy',
            'data_source': 'none' if final_refit else 'training_holdout_validation',
            'selection_data': 'none' if final_refit else 'training_holdout',
            'best_epoch': best_epoch,
            'best_pgd_error': None if final_refit else float(best),
            'last_completed_epoch': epoch,
            'validation_split': P.validation_split,
            'selection_lock_sha256': P.selection_lock_sha256,
        })
    if epoch % P.save_step == 0:
        save_checkpoint_epoch(epoch, save_states, optimizer.state_dict(), logger.logdir)

# AA is launched only after this marker and a finite selected checkpoint are present.
if not (Path(logger.logdir) / 'best.model').is_file():
    raise RuntimeError('Training finished without a selected best.model checkpoint')
completion = {'epoch': P.epochs, 'best_pgd_error': None if final_refit else float(best)}
if selection_free_v2:
    checkpoint_hasher = hashlib.sha256()
    with (Path(logger.logdir) / 'best.model').open('rb') as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            checkpoint_hasher.update(chunk)
    completion.update(experiment_protocol=P.experiment_protocol, protocol_stage=P.protocol_stage,
                      checkpoint_selection='fixed_last_epoch' if final_refit else 'validation_pgd_accuracy',
                      selection_data='none' if final_refit else 'training_holdout',
                      best_epoch=best_epoch, validation_split=P.validation_split,
                      selection_lock_sha256=P.selection_lock_sha256,
                      checkpoint_sha256=checkpoint_hasher.hexdigest())
atomic_json(Path(logger.logdir) / 'TRAIN_COMPLETE.json', completion)
logger.writer.close()
logger.log_file.close()

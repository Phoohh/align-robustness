import torch
import math
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

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.augment_type == 'autoaug_sche' and epoch > (P.epochs/2):
        train_loader = P.train_second_loader

    train(P, epoch, model, criterion, optimizer, scheduler, train_loader, logger=logger, **kwargs)
    model.eval()

    if epoch % P.error_step == 0:
        error = test_classifier_adv(P, model, test_loader, epoch,
                                    adversary=adversary_t, logger=logger, ret='adv')

        if not math.isfinite(float(error)):
            raise RuntimeError('Nonfinite PGD evaluation error')
        is_best = (not (Path(logger.logdir) / 'best.model').exists()) or (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_adv_error', best, epoch)
        logger.log('[Epoch %3d] [Adv_Test %5.2f] [Best %5.2f]' % (epoch, error, best))

    save_states = model.state_dict()
    save_checkpoint(epoch, best, save_states, optimizer.state_dict(), logger.logdir, is_best)
    if epoch % P.save_step == 0:
        save_checkpoint_epoch(epoch, save_states, optimizer.state_dict(), logger.logdir)

# AA is launched only after this marker and a finite selected checkpoint are present.
if not (Path(logger.logdir) / 'best.model').is_file():
    raise RuntimeError('Training finished without a selected best.model checkpoint')
atomic_json(Path(logger.logdir) / 'TRAIN_COMPLETE.json', {'epoch': P.epochs, 'best_pgd_error': float(best)})
logger.writer.close()
logger.log_file.close()

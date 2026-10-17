import torch
import torch.nn as nn
import json
import re
from pathlib import Path
from torch.utils.data import DataLoader

from common.common import parse_args
from common.utils import get_optimizer, get_scheduler
from adv_lib.attack import attack_module
import models.classifier as C
from datasets import get_dataset, get_selection_free_datasets
from utils.utils import load_checkpoint, set_random_seed
from utils.resume_state import FILENAME, load_training_state, restore_exports

P = parse_args()
selection_free_v2 = P.experiment_protocol == 'selection_free_v2'
final_refit = selection_free_v2 and P.protocol_stage == 'final'
if P.max_epochs_this_run is not None and (not selection_free_v2 or P.max_epochs_this_run <= 0):
    raise ValueError('Epoch segmentation requires selection_free_v2 and a positive segment length')
if selection_free_v2:
    if P.study_percent is not None:
        raise ValueError('selection_free_v2 does not yet support study_percent')
    if P.error_step <= 0 or P.save_step <= 0:
        raise ValueError('Evaluation and checkpoint intervals must be positive')
    if final_refit:
        if not isinstance(P.selection_lock_sha256, str) or re.fullmatch('[0-9a-f]{64}', P.selection_lock_sha256) is None:
            raise ValueError('Final refits require a validated selection_lock_sha256 from the experiment driver')
    elif P.selection_lock_sha256 is not None:
        raise ValueError('Development runs must not claim a final selection lock')
    if P.resume_path is not None and P.output_dir is not None and Path(P.output_dir).resolve() != Path(P.resume_path).resolve():
        raise ValueError('Resume in the original output directory to retain split and checkpoint provenance')

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU

set_random_seed(P.seed)

### Initialize dataset ###
if selection_free_v2:
    train_set, validation_set, image_size, n_classes = get_selection_free_datasets(
        P, dataset=P.dataset, augment=P.augment_type)
else:
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, augment=P.augment_type)
P.image_size = image_size
P.n_classes = n_classes

### Define data loader ###
kwargs = {'pin_memory': True, 'num_workers': P.num_workers}
train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
if selection_free_v2:
    validation_loader = (DataLoader(validation_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
                         if validation_set is not None else None)
    selection_loader = validation_loader
    selection_role = 'fixed_last_epoch' if final_refit else 'training_holdout_validation'
else:
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    selection_loader = test_loader
    selection_role = 'legacy_test'

if P.augment_type == 'autoaug_sche':
    dataset_builder = get_selection_free_datasets if selection_free_v2 else get_dataset
    train_set_second, _, _, _ = dataset_builder(P, dataset=P.dataset, augment='autoaug')
    P.train_second_loader = DataLoader(train_set_second, shuffle=True, batch_size=P.batch_size, **kwargs)

### Initialize model ###
model = C.get_classifier(P, n_classes=P.n_classes).to(device)
optimizer, lr_decay_gamma = get_optimizer(P, model)
scheduler = get_scheduler(P, optimizer, lr_decay_gamma)

resume_rng_state = None
if P.resume_path is not None:
    resume = True
    exact = (load_training_state(P, P.resume_path)
             if selection_free_v2 and (Path(P.resume_path) / FILENAME).is_file() else None)
    if exact is not None:
        model_state, optim_state = exact['model'], exact['optimizer']
        config = {'epoch': exact['epoch'], 'best': exact['selection']['best_pgd_error']}
        restore_exports(exact, P.resume_path)
        resume_rng_state = exact['rng']
    else:
        if P.max_epochs_this_run is not None:
            raise ValueError('Segmented continuation requires an atomic checkpoint with RNG state')
        model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    if config['epoch'] > P.epochs:
        raise ValueError('Saved checkpoint exceeds the requested training schedule')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    # Config stores the last completed epoch. Continue with the next epoch.
    start_epoch = config['epoch'] + 1
    if exact is not None:
        scheduler.load_state_dict(exact['scheduler'])
    else:
        scheduler.step(config['epoch'])
    print(f'SCHEDULER_GLOBAL_EPOCH_RESTORED={config["epoch"]} LR={optimizer.param_groups[0]["lr"]:.8f}')
    best = config['best']
    best_epoch = None
    if selection_free_v2:
        selection_path = Path(P.resume_path) / 'checkpoint_selection.json'
        if not selection_path.is_file():
            raise ValueError('Cannot resume selection_free_v2 from a legacy or incomplete checkpoint')
        saved_selection = json.loads(selection_path.read_text())
        expected_criterion = 'fixed_last_epoch' if final_refit else 'validation_pgd_accuracy'
        if (saved_selection.get('experiment_protocol') != P.experiment_protocol
                or saved_selection.get('protocol_stage') != P.protocol_stage
                or saved_selection.get('criterion') != expected_criterion
                or saved_selection.get('last_completed_epoch') != config['epoch']
                or saved_selection.get('validation_split') != P.validation_split
                or saved_selection.get('selection_lock_sha256') != P.selection_lock_sha256
                or saved_selection.get('best_pgd_error') != best):
            raise ValueError('Checkpoint selection/split/lock provenance differs from this run')
        best_epoch = saved_selection.get('best_epoch')
        if not final_refit and (type(best_epoch) is not int or not 1 <= best_epoch <= config['epoch']):
            raise ValueError('Invalid validation-selected checkpoint epoch')
        if final_refit and best_epoch != (P.epochs if config['epoch'] == P.epochs else None):
            raise ValueError('Final refits may select only the predeclared last epoch')
    is_best = False
    error = 100.0
else:
    resume = False
    is_best = False
    start_epoch = 1
    best = 100.0
    best_epoch = None
    error = 100.0

if final_refit:
    best = None

criterion = nn.CrossEntropyLoss().to(device)
adversary, adversary_t = attack_module(P, model, criterion)

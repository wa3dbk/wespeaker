# Copyright (c) 2025 Waad Ben Kheder (benkheder@vocapia.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from pprint import pformat
import numpy as np
import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import *
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id


def setup_model_for_finetuning(
    logger,
    ckpt_path: str,
    configs: dict
):
    """
    Generic fine-tuning setup for any WeSpeaker model.

    Args:
        logger: Logger instance
        ckpt_path: Path to pretrained checkpoint
        configs: Configuration dictionary containing:
            - model: Model name
            - model_args: Model arguments
            - projection_args: Projection layer arguments
            - finetune_config: Fine-tuning configuration with:
                - freeze_strategy: "all" | "none" | "exclude" | "only"
                - freeze_layers: list of layer name patterns
                - freeze_projection: bool (default: False)
                - new_embed_dim: int or None (if set, replaces embedding layer)

    Returns:
        model: Configured model ready for fine-tuning
    """

    new_num_classes = configs['projection_args']['num_class']
    finetune_config = configs.get('finetune_config', {})

    # Default fine-tuning configuration
    freeze_strategy = finetune_config.get('freeze_strategy', 'exclude')
    freeze_layers = finetune_config.get('freeze_layers', ['projection', 'pool'])
    freeze_projection = finetune_config.get('freeze_projection', False)
    new_embed_dim = finetune_config.get('new_embed_dim', None)

    logger.info("<== Fine-tuning Configuration ==>")
    logger.info(f"Freeze strategy: {freeze_strategy}")
    logger.info(f"Freeze layers: {freeze_layers}")
    logger.info(f"Freeze projection: {freeze_projection}")
    if new_embed_dim is not None:
        logger.info(f"New embedding dimension: {new_embed_dim} (was {configs['model_args']['embed_dim']})")

    # 1. Instantiate model template
    model = get_speaker_model(configs['model'])(**configs['model_args'])

    # 2. Load the checkpoint
    logger.info(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = checkpoint.get("model_state_dict", checkpoint)

    # 3. Strip out layers that will be replaced
    # Always strip projection layer
    keys_to_remove = [k for k in sd.keys() if k.startswith("projection.")]

    # If changing embedding dimension, also strip embedding layers
    if new_embed_dim is not None:
        # Embedding layer patterns for different architectures
        embedding_patterns = [
            "seg_1.",      # ResNet embedding layers
            "seg_2.",      # ResNet second embedding layer (if two_emb_layer=True)
            "seg_bn_1.",   # ResNet embedding batch norm
            "xvector.dense.",  # CAMPPlus/ECAPA embedding layer
            "fc6.",        # TDNN embedding layer
        ]
        for key in list(sd.keys()):
            if any(pattern in key for pattern in embedding_patterns):
                keys_to_remove.append(key)

    for key in keys_to_remove:
        sd.pop(key, None)

    logger.info(f"Stripped {len(keys_to_remove)} layer(s) from checkpoint: projection and embedding layers")

    # 4. Load everything else (strict=False to allow missing layers)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded checkpoint - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # 5. Freeze layers based on strategy
    if freeze_strategy == 'all':
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze all model parameters")

    elif freeze_strategy == 'none':
        # Don't freeze anything
        for param in model.parameters():
            param.requires_grad = True
        logger.info("All model parameters unfrozen")

    elif freeze_strategy == 'exclude':
        # Freeze all, then unfreeze specified layers
        for param in model.parameters():
            param.requires_grad = False

        for name, module in model.named_modules():
            # Check if this layer matches any pattern in freeze_layers
            if any(pattern in name for pattern in freeze_layers):
                logger.info(f"Unfreezing layer: {name}")
                for p in module.parameters():
                    p.requires_grad = True

    elif freeze_strategy == 'only':
        # Unfreeze all, then freeze only specified layers
        for param in model.parameters():
            param.requires_grad = True

        for name, module in model.named_modules():
            # Check if this layer matches any pattern in freeze_layers
            if any(pattern in name for pattern in freeze_layers):
                logger.info(f"Freezing layer: {name}")
                for p in module.parameters():
                    p.requires_grad = False

    # 6. Replace embedding layer if new_embed_dim is specified
    if new_embed_dim is not None:
        logger.info(f"<== Replacing Embedding Layer ==>")
        model_name = configs['model']

        # Detect model architecture and rebuild embedding layer
        if model_name.startswith('ResNet') or model_name.startswith('SimAM_ResNet'):
            # ResNet models: rebuild seg_1 (and seg_2 if two_emb_layer=True)
            two_emb_layer = configs['model_args'].get('two_emb_layer', False)
            model.seg_1 = torch.nn.Linear(model.pool_out_dim, new_embed_dim)
            logger.info(f"Replaced ResNet seg_1: {model.pool_out_dim} -> {new_embed_dim}")

            if two_emb_layer:
                model.seg_bn_1 = torch.nn.BatchNorm1d(new_embed_dim, affine=False)
                model.seg_2 = torch.nn.Linear(new_embed_dim, new_embed_dim)
                logger.info(f"Replaced ResNet seg_2 and seg_bn_1 for two_emb_layer=True")
            else:
                model.seg_bn_1 = torch.nn.Identity()
                model.seg_2 = torch.nn.Identity()

        elif model_name.startswith('CAMPPlus'):
            # CAMPPlus: rebuild xvector.dense (DenseLayer)
            from wespeaker.models.campplus import DenseLayer
            model.xvector.dense = DenseLayer(model.pool_out_dim, new_embed_dim, config_str='batchnorm_')
            logger.info(f"Replaced CAMPPlus xvector.dense: {model.pool_out_dim} -> {new_embed_dim}")

        elif model_name.startswith('ECAPA_TDNN'):
            # ECAPA-TDNN: rebuild fc6 (Conv1d layer)
            model.fc6 = torch.nn.Conv1d(model.channels, new_embed_dim, kernel_size=1)
            logger.info(f"Replaced ECAPA_TDNN fc6: {model.channels} -> {new_embed_dim}")

        elif model_name.startswith('XVEC'):
            # TDNN/XVEC: rebuild fc6
            model.fc6 = torch.nn.Linear(model.fc5_out_dim, new_embed_dim)
            logger.info(f"Replaced XVEC fc6: {model.fc5_out_dim} -> {new_embed_dim}")

        else:
            logger.warning(f"Embedding layer replacement not implemented for {model_name}")
            logger.warning("Embedding dimension will remain unchanged")

        # Update embed_dim in configs for projection layer
        configs['model_args']['embed_dim'] = new_embed_dim
        configs['projection_args']['embed_dim'] = new_embed_dim

        # Always ensure embedding layers are trainable
        embedding_layer_names = ['seg_1', 'seg_2', 'seg_bn_1', 'xvector.dense', 'fc6']
        for name, param in model.named_parameters():
            if any(emb_name in name for emb_name in embedding_layer_names):
                param.requires_grad = True
        logger.info("Embedding layers set to trainable")

    # 7. Replace the projection head with new one for new number of classes
    in_features = configs['model_args']['embed_dim']
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    logger.info(f"Replaced projection layer with new one (out_features={new_num_classes})")

    # 8. Handle projection layer freezing
    if freeze_projection:
        logger.info("Freezing projection layer")
        for p in model.projection.parameters():
            p.requires_grad = False
    else:
        # Always unfreeze projection for fine-tuning (unless explicitly frozen)
        logger.info("Unfreezing projection layer")
        for p in model.projection.parameters():
            p.requires_grad = True

    # 9. Report trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 10. Log which layers are trainable
    logger.info("<== Trainable Layers ==>")
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    for layer in trainable_layers[:20]:  # Show first 20 to avoid clutter
        logger.info(f"  {layer}")
    if len(trainable_layers) > 20:
        logger.info(f"  ... and {len(trainable_layers) - 20} more layers")

    return model


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels with fine-tuning.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)

    # Disable speed perturbation for fine-tuning (optional, can be configured)
    if configs.get('finetune_config', {}).get('disable_speed_perturb', True):
        configs['dataset_args']['speed_perturb'] = False

    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    if isinstance(configs['gpus'], int):
        configs['gpus'] = [configs['gpus']]
    gpu = int(configs['gpus'][local_rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print("[warning] " + model_dir + " already exists !!!")
            if checkpoint is None:
                print("[error] checkpoint is null !")
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank":
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))

    # For model_init, only frontend and speaker model are needed
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')

    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args']['embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    logger.info("Number of classes: {}".format(len(spk2id_dict)))
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)

    if configs['data_type'] != 'feat' and configs['dataset_args']['speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info('No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False

    # If checkpoint is specified, setup model for fine-tuning
    if checkpoint is not None:
        model = setup_model_for_finetuning(logger, ckpt_path=checkpoint, configs=configs)
        # Try to extract epoch number from checkpoint filename
        # Supports patterns like: model_150.pt, avg_model.pt, final_model.pt, etc.
        epoch_match = re.findall(r"(?<=model_)\d+(?=\.pt)", checkpoint)
        if epoch_match:
            start_epoch = int(epoch_match[0]) + 1
            logger.info('Load checkpoint: {} (epoch {})'.format(checkpoint, epoch_match[0]))
        else:
            start_epoch = 1
            logger.info('Load checkpoint: {} (starting from epoch 1)'.format(checkpoint))
    else:
        # Training from scratch - add projection normally
        projection = get_projection(configs['projection_args'])
        model.add_module("projection", projection)
        start_epoch = 1

    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # Try to export the model by script
        if frontend_type == 'fbank':
            try:
                script_model = torch.jit.script(model)
                script_model.save(os.path.join(model_dir, 'init.zip'))
            except Exception as e:
                logger.warning(f"Failed to export scripted model: {e}")

    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  ddp_model,
                  criterion,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)

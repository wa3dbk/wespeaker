# Fine-tuning Guide for WeSpeaker Models

This guide explains how to fine-tune pretrained WeSpeaker models on new datasets using the flexible `finetune.py` script.

## Overview

Fine-tuning allows you to adapt pretrained speaker recognition models to new datasets with different speakers. The `wespeaker/bin/finetune.py` script provides a flexible framework for:

- **Universal model support**: Works with all WeSpeaker architectures (CAMPPlus, ECAPA-TDNN, ResNet, etc.)
- **Flexible layer freezing**: Control which layers to freeze/unfreeze with simple configuration
- **Multiple freezing strategies**: Support for various fine-tuning approaches
- **Pattern-based layer selection**: Use substring patterns to specify layers

## Quick Start

### 1. Prepare Your Data

Organize your fine-tuning data in the standard WeSpeaker format:
```bash
data/
├── train/
│   ├── wav.scp          # Audio file paths
│   └── utt2spk          # Utterance to speaker mapping
```

### 2. Choose a Pretrained Model

Download a pretrained model or use your own:
```bash
# Example: Download a pretrained model
mkdir -p pretraining_models
wget https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/avg_model.pt -O prerained_models/wespeaker-voxceleb-resnet34.pt
```

### 3. Create a Fine-tuning Configuration

Copy and modify one of the example configs:
```bash
cp examples/voxceleb/v2/conf/finetune_example.yaml conf/my_finetune.yaml
```

Edit the config to specify:
- `checkpoint`: Path to pretrained model (.pt file)
- `model`: Model architecture name (must match the pretrained model)
- `finetune_config`: Layer freezing strategy (see below)
- Training data paths

### 4. Run Fine-tuning

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0
bash run.sh --stage 4 \
  --config conf/my_finetune.yaml \
  --train_data data/train/wav.scp \
  --train_label data/train/utt2spk

# Multi-GPU (recommended)
export CUDA_VISIBLE_DEVICES=0,1
num_gpus=2
torchrun --nproc_per_node=$num_gpus \
  wespeaker/bin/finetune.py \
  --config conf/my_finetune.yaml \
  --train_data data/train/wav.scp \
  --train_label data/train/utt2spk
```

## Layer Freezing Strategies

The `finetune_config` section controls which layers are frozen during fine-tuning.

### Strategy Types

#### 1. `exclude` (Default)
Freeze **all layers EXCEPT** those matching the patterns in `freeze_layers`.

**Use case**: Fine-tune only the last layers (common approach)

```yaml
finetune_config:
  freeze_strategy: "exclude"
  freeze_layers:
    - "pool"              # Unfreeze pooling layer
    - "layer4"            # Unfreeze layer4 (ResNet)
    - "projection"        # Unfreeze projection (usually automatic)
```

#### 2. `only`
Freeze **ONLY** layers matching the patterns in `freeze_layers`.

**Use case**: Freeze early layers, train later layers

```yaml
finetune_config:
  freeze_strategy: "only"
  freeze_layers:
    - "conv1"             # Freeze first conv layer
    - "layer1"            # Freeze layer1
    - "layer2"            # Freeze layer2
```

#### 3. `none`
Don't freeze any layers (full fine-tuning).

**Use case**: When you have a large fine-tuning dataset

```yaml
finetune_config:
  freeze_strategy: "none"
  freeze_layers: []       # Ignored
```

#### 4. `all`
Freeze all layers.

**Use case**: Rare; useful for debugging or projection-only training

```yaml
finetune_config:
  freeze_strategy: "all"
  freeze_layers: []       # Can then selectively unfreeze
```

### Layer Name Patterns

Layer names are matched using **substring matching**. Here are common patterns for different architectures:

#### CAMPPlus
```yaml
freeze_layers:
  - "head"                # Feature extraction head
  - "xvector.tdnn"        # TDNN layer
  - "xvector.block1"      # First dense block
  - "xvector.block2"      # Second dense block
  - "xvector.block3"      # Third dense block
  - "xvector.transit1"    # First transition layer
  - "xvector.transit2"    # Second transition layer
  - "xvector.transit3"    # Third transition layer
  - "xvector.out_nonlinear"  # Output nonlinearity
  - "xvector.stats"       # Statistics pooling
  - "xvector.dense"       # Final dense layer
  - "pool"                # Pooling layer
```

#### ResNet (ResNet34, ResNet293, etc.)
```yaml
freeze_layers:
  - "conv1"               # First convolution
  - "bn1"                 # First batch norm
  - "layer1"              # First residual layer group
  - "layer2"              # Second residual layer group
  - "layer3"              # Third residual layer group
  - "layer4"              # Fourth residual layer group
  - "pool"                # Pooling layer
```

#### ECAPA-TDNN
```yaml
freeze_layers:
  - "layer1"              # First TDNN layer
  - "layer2"              # Second TDNN layer
  - "layer3"              # Third TDNN layer
  - "layer4"              # Fourth TDNN layer
  - "conv"                # Convolution layers
  - "tdnn"                # TDNN layers
  - "pool"                # Pooling layer
```

**Tip**: To see all layer names in your model, check the log output during initialization. The script logs all trainable layers.

## Configuration Examples

### Example 1: CAMPPlus - Conservative Fine-tuning

Fine-tune only the last layers and pooling (recommended for small datasets):

```yaml
# conf/finetune_campplus_conservative.yaml
exp_dir: exp/CAMPPlus-finetune
gpus: "[0,1]"
num_epochs: 30
checkpoint: pretrained_models/campplus_cn_common.pt

model: CAMPPlus
model_args:
  feat_dim: 80
  embed_dim: 512
  pooling_func: "TSTP"

finetune_config:
  freeze_strategy: "exclude"
  freeze_layers:
    - "pool"
    - "xvector.block3"
    - "xvector.transit3"
    - "xvector.out_nonlinear"
    - "xvector.stats"
    - "xvector.dense"
  freeze_projection: False

scheduler_args:
  initial_lr: 0.01        # Lower LR for fine-tuning
  final_lr: 0.00001
```

### Example 2: ResNet293 - Freeze Early Layers

Freeze early feature extraction, train later layers:

```yaml
# conf/finetune_resnet293.yaml
exp_dir: exp/ResNet293-finetune
gpus: "[0,1,2,3]"
num_epochs: 25
checkpoint: pretrained_models/wespeaker-voxceleb-resnet293-LM/avg_model.pt

model: ResNet293
model_args:
  feat_dim: 80
  embed_dim: 256
  pooling_func: "ASTP"

finetune_config:
  freeze_strategy: "only"
  freeze_layers:
    - "conv1"
    - "bn1"
    - "layer1"
    - "layer2"
  # layer3, layer4, pool, and projection will be trainable
  freeze_projection: False

scheduler_args:
  initial_lr: 0.01
  final_lr: 0.00001
```

### Example 3: ECAPA-TDNN - Full Fine-tuning

Full fine-tuning with very low learning rate (for large datasets):

```yaml
# conf/finetune_ecapa_full.yaml
exp_dir: exp/ECAPA-finetune-full
gpus: "[0,1]"
num_epochs: 20
checkpoint: pretrained_models/voxceleb_ECAPA_TDNN_GLOB_c1024_ASTP.pt

model: ECAPA_TDNN_GLOB_c1024
model_args:
  feat_dim: 80
  embed_dim: 192
  pooling_func: "ASTP"

finetune_config:
  freeze_strategy: "none"    # Train all layers
  freeze_layers: []
  freeze_projection: False

optimizer: Adam              # Adam often works better for full fine-tuning
optimizer_args:
  weight_decay: 0.00001

scheduler_args:
  initial_lr: 0.0001         # Very low LR for full fine-tuning
  final_lr: 0.00001
```

## Advanced Configuration Options

### Changing Embedding Dimension

**NEW FEATURE**: You can change the embedding dimension during fine-tuning to create smaller, more efficient models.

The embedding layer (the final layer before the projection that produces speaker embeddings) will be replaced with a new dimension and trained from scratch, while keeping the pretrained feature extractor frozen or partially frozen.

#### Basic Configuration

```yaml
model_args:
  embed_dim: 128  # New embedding size (pretrained might be 256)

finetune_config:
  new_embed_dim: 128  # Must match model_args.embed_dim
```

#### How It Works

When you specify `new_embed_dim`, the fine-tuning script performs the following steps:

1. **Loads pretrained checkpoint** and strips out:
   - Projection layer (always replaced)
   - Embedding layers (specific to architecture)

2. **Loads remaining weights** into the new model (feature extractor layers)

3. **Applies layer freezing** strategy (before replacing embedding layer)

4. **Replaces embedding layer** with new dimension:

   **ResNet Models** (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet221, ResNet293):
   ```
   Architecture: ... → layer4 → pool → seg_1 → [seg_bn_1 → seg_2] → projection
                                       ↑        ↑            ↑
                                    Replaced  (optional, if two_emb_layer=True)
   ```
   - **Replaced layers**:
     - `seg_1`: Linear(pool_out_dim → new_embed_dim)
     - `seg_bn_1`: BatchNorm1d(new_embed_dim) if `two_emb_layer=True`, else Identity
     - `seg_2`: Linear(new_embed_dim → new_embed_dim) if `two_emb_layer=True`, else Identity
   - **Input dimension**: Determined by pooling layer output (`pool_out_dim`)
   - **Output dimension**: `new_embed_dim`
   - **Note**: Most pretrained models use `two_emb_layer=False` (CNSRC 2022 optimization)

   **CAMPPlus**:
   ```
   Architecture: ... → block3 → transit3 → out_nonlinear → stats (pool) → dense → projection
                                                                           ↑
                                                                       Replaced
   ```
   - **Replaced layers**:
     - `xvector.dense`: DenseLayer(pool_out_dim → new_embed_dim)
       - DenseLayer = Conv1d(1x1) + BatchNorm (affine=False)
   - **Input dimension**: Determined by pooling layer output (`pool_out_dim`)
   - **Output dimension**: `new_embed_dim`

   **ECAPA-TDNN** (ECAPA_TDNN_GLOB_c512, ECAPA_TDNN_GLOB_c1024, etc.):
   ```
   Architecture: ... → layer4 → attention → pool → bn5 → fc6 → bn6 → projection
                                                          ↑
                                                      Replaced
   ```
   - **Replaced layers**:
     - `fc6`: Conv1d(channels → new_embed_dim, kernel_size=1)
   - **Input dimension**: Number of channels after layer4 (`channels`)
   - **Output dimension**: `new_embed_dim`

   **TDNN/XVEC**:
   ```
   Architecture: ... → fc5 → fc6 → projection
                              ↑
                          Replaced
   ```
   - **Replaced layers**:
     - `fc6`: Linear(fc5_out_dim → new_embed_dim)
   - **Input dimension**: fc5 output dimension (`fc5_out_dim`)
   - **Output dimension**: `new_embed_dim`

5. **Ensures embedding layers are trainable** (overrides freezing strategy for these layers)

6. **Replaces projection layer** with new dimension (in_features = `new_embed_dim`)

#### Benefits

- **Reduced model size**: e.g., 256→128 cuts embedding layer parameters in half
- **Faster inference**: Smaller embeddings mean faster similarity computations
- **Reduced overfitting**: Can improve performance on smaller datasets
- **Deployment friendly**: Ideal for resource-constrained devices
- **Flexible compression**: Test different embedding sizes (256→128→64)

#### Complete Example

```yaml
# conf/finetune_resnet34_emb128.yaml
exp_dir: exp/ResNet34-finetune-emb128
checkpoint: pretrained_models/resnet34_emb256/avg_model.pt  # Pretrained with 256-dim
num_epochs: 30

model: ResNet34
model_args:
  feat_dim: 80
  embed_dim: 128          # Target embedding dimension
  pooling_func: "ASTP"
  two_emb_layer: False    # Must match pretrained model

finetune_config:
  freeze_strategy: "only"
  freeze_layers:
    - "conv1"             # Freeze early feature extraction
    - "bn1"
    - "layer1"
    - "layer2"
  # layer3, layer4, pool, seg_1 will be trainable
  new_embed_dim: 128      # Replace seg_1 with 128-dim output
  freeze_projection: False

scheduler_args:
  initial_lr: 0.01        # Normal fine-tuning LR
  final_lr: 0.00001
```

#### Model Size Comparison

| Model | Original embed_dim | New embed_dim | Embedding Layer Params | Savings |
|-------|-------------------|---------------|------------------------|---------|
| ResNet34 (pool_out=512) | 256 | 128 | 512×256 → 512×128 | ~131K params |
| ResNet34 (pool_out=512) | 256 | 64 | 512×256 → 512×64 | ~229K params |
| CAMPPlus (pool_out=1536) | 512 | 256 | 1536×512 → 1536×256 | ~786K params |
| CAMPPlus (pool_out=1536) | 512 | 128 | 1536×512 → 1536×128 | ~983K params |

**Note**: Projection layer size also changes (embed_dim × num_speakers), providing additional savings.

#### Important Considerations

1. **Must match pretrained model architecture**: Ensure `two_emb_layer`, `pooling_func`, and other model args match the pretrained model (except `embed_dim`)

2. **Embedding layer is always trainable**: When `new_embed_dim` is specified, the embedding layer will be trained regardless of your freezing strategy

3. **Feature extractor stays frozen/partially frozen**: Only the embedding and projection layers adapt to the new dimension

4. **Recommended workflow**:
   - Start with small dimension reduction (256→192 or 256→128)
   - Monitor validation performance
   - If performance is acceptable, try more aggressive reduction (128→64)

5. **Training time**: Slightly longer than standard fine-tuning since embedding layer trains from scratch

#### Advanced: Testing Multiple Dimensions

You can experiment with different embedding dimensions to find the optimal size/performance tradeoff:

```bash
# Test different embedding dimensions
for embed_dim in 256 192 128 64; do
  exp_dir=exp/ResNet34-finetune-emb${embed_dim}
  torchrun --nproc_per_node=2 \
    wespeaker/bin/finetune.py \
    --config conf/finetune_base.yaml \
    --exp_dir ${exp_dir} \
    --model_args.embed_dim ${embed_dim} \
    --finetune_config.new_embed_dim ${embed_dim}
done

# Compare results
python tools/compare_embeddings.py \
  --dirs exp/ResNet34-finetune-emb*
```

#### Supported Models

| Model Family | Status | Replaced Layers |
|--------------|--------|-----------------|
| ResNet (all variants) | ✓ | seg_1, seg_2, seg_bn_1 |
| CAMPPlus | ✓ | xvector.dense |
| ECAPA-TDNN | ✓ | fc6 |
| TDNN/XVEC | ✓ | fc6 |
| ERes2Net | ✓ | Same as ResNet |
| Res2Net | ✓ | Same as ResNet |
| SimAM_ResNet | ✓ | Same as ResNet |
| XI_VEC | ✓ | Same as ECAPA-TDNN |
| RepVGG | ⚠️ | Not yet implemented |
| Gemini_DFResNet | ⚠️ | Not yet implemented |
| whisper_PMFA | ⚠️ | Not yet implemented |

For unsupported models, the script will log a warning and continue with the original embedding dimension.

### Projection Layer Settings

The projection layer is always replaced with a new one (for the new number of speakers). You can control whether it's frozen:

```yaml
finetune_config:
  freeze_projection: False   # Default: train the projection layer
  # Set to True only in rare cases (e.g., debugging)
```

### Speed Perturbation

Speed perturbation is typically disabled during fine-tuning:

```yaml
finetune_config:
  disable_speed_perturb: True  # Default: disabled for fine-tuning

# Or control it directly in dataset_args:
dataset_args:
  speed_perturb: False
```

### Learning Rate Guidelines

| Fine-tuning Type | Recommended Initial LR | Notes |
|------------------|------------------------|-------|
| Last layers only | 0.01 - 0.05 | Conservative, stable |
| Half model | 0.005 - 0.01 | Moderate |
| Full model | 0.0001 - 0.001 | Very low LR to preserve pretrained features |

### Optimizer Selection

```yaml
# SGD with momentum (default, stable)
optimizer: SGD
optimizer_args:
  momentum: 0.9
  nesterov: True
  weight_decay: 0.0001

# Adam (alternative, sometimes better for full fine-tuning)
optimizer: Adam
optimizer_args:
  weight_decay: 0.00001
```

## Best Practices

### 1. Start Conservative
Begin with freezing most layers, especially if you have a small dataset:
```yaml
freeze_strategy: "exclude"
freeze_layers: ["pool", "layer4", "projection"]  # Only train last layer + pool
```

### 2. Monitor Overfitting
Fine-tuning can overfit quickly on small datasets:
- Use fewer epochs (20-30 instead of 150)
- Use lower learning rate
- Monitor validation performance if available

### 3. Learning Rate Selection
- **Small dataset + few layers**: Start with `initial_lr: 0.01`
- **Large dataset + many layers**: Start with `initial_lr: 0.001`
- **Full fine-tuning**: Start with `initial_lr: 0.0001`

### 4. Batch Size Adjustment
You can use smaller batch sizes for fine-tuning:
```yaml
dataloader_args:
  batch_size: 64   # vs 128-256 for training from scratch
```

### 5. Data Augmentation
- Disable speed perturbation (usually too aggressive for fine-tuning)
- Keep reverb/noise augmentation if your dataset is small
```yaml
dataset_args:
  speed_perturb: False
  aug_prob: 0.6           # Keep reverb/noise aug
```

## Checkpoint Compatibility

The fine-tuning script supports various checkpoint formats:

| Checkpoint Name | Epoch Detection | Notes |
|-----------------|-----------------|-------|
| `model_150.pt` | ✓ Epoch 150 | Standard format |
| `avg_model.pt` | ✗ Starts epoch 1 | Averaged model |
| `final_model.pt` | ✗ Starts epoch 1 | Final checkpoint |
| `best_model.pt` | ✗ Starts epoch 1 | Best validation model |

All formats work correctly; epoch number is only used for logging.

## Troubleshooting

### Issue: "Missing keys" when loading checkpoint

**Cause**: Model architecture mismatch between config and checkpoint.

**Solution**: Ensure `model` in config matches the pretrained model architecture:
```yaml
# Check pretrained model's config.yaml to find correct model name
model: ResNet293  # Must match pretrained model
```

### Issue: Out of memory during training

**Solutions**:
1. Reduce batch size:
   ```yaml
   dataloader_args:
     batch_size: 32  # Reduce from 64
   ```

2. Reduce number of frames:
   ```yaml
   dataset_args:
     num_frms: 150  # Reduce from 200
   ```

3. Enable automatic mixed precision:
   ```yaml
   enable_amp: True
   ```

### Issue: Loss not decreasing

**Solutions**:
1. Check if too many layers are frozen:
   ```yaml
   # Make sure you're training enough parameters
   freeze_strategy: "exclude"
   freeze_layers: ["layer3", "layer4", "pool"]  # Unfreeze more layers
   ```

2. Increase learning rate:
   ```yaml
   scheduler_args:
     initial_lr: 0.05  # Increase if loss plateaus immediately
   ```

### Issue: Model overfitting quickly

**Solutions**:
1. Freeze more layers:
   ```yaml
   freeze_strategy: "exclude"
   freeze_layers: ["layer4", "pool"]  # Only train last layer
   ```

2. Reduce learning rate:
   ```yaml
   scheduler_args:
     initial_lr: 0.005  # Lower LR
   ```

3. Use fewer epochs:
   ```yaml
   num_epochs: 20  # Stop earlier
   ```

## Workflow Example

Complete fine-tuning workflow from start to finish:

```bash
#!/bin/bash
# Fine-tune ResNet34 on custom dataset

# 1. Download pretrained model
mkdir -p pretrained_models
wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.tar.gz
tar -xzf voxceleb_resnet34.tar.gz -C pretrained_models/

# 2. Prepare your data (assuming already in data/my_dataset/)
# data/my_dataset/train/wav.scp
# data/my_dataset/train/utt2spk

# 3. Create fine-tuning config
cat > conf/finetune_resnet34_custom.yaml << EOF
exp_dir: exp/ResNet34-custom-finetune
gpus: "[0,1]"
num_avg: 5
enable_amp: False
seed: 42
num_epochs: 25
save_epoch_interval: 5

checkpoint: pretrained_models/voxceleb_resnet34/avg_model.pt

dataloader_args:
  batch_size: 64
  num_workers: 8
  pin_memory: False
  drop_last: True

dataset_args:
  sample_num_per_epoch: 0
  shuffle: True
  shuffle_args:
    shuffle_size: 2500
  filter: True
  filter_args:
    min_num_frames: 100
    max_num_frames: 800
  resample_rate: 16000
  speed_perturb: False
  num_frms: 200
  aug_prob: 0.6
  fbank_args:
    num_mel_bins: 80
    frame_shift: 10
    frame_length: 25
    dither: 1.0

model: ResNet34
model_args:
  feat_dim: 80
  embed_dim: 256
  pooling_func: "ASTP"

projection_args:
  project_type: "arc_margin"
  scale: 32.0
  easy_margin: False

finetune_config:
  freeze_strategy: "only"
  freeze_layers:
    - "conv1"
    - "bn1"
    - "layer1"
    - "layer2"
  freeze_projection: False
  disable_speed_perturb: True

margin_scheduler: MarginScheduler
margin_update:
  initial_margin: 0.0
  final_margin: 0.2
  increase_start_epoch: 5
  fix_start_epoch: 15
  update_margin: True
  increase_type: "exp"

loss: CrossEntropyLoss
loss_args: {}

optimizer: SGD
optimizer_args:
  momentum: 0.9
  nesterov: True
  weight_decay: 0.0001

scheduler: ExponentialDecrease
scheduler_args:
  initial_lr: 0.01
  final_lr: 0.00001
  warm_up_epoch: 2
  warm_from_zero: False
EOF

# 4. Run fine-tuning
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 \
  wespeaker/bin/finetune.py \
  --config conf/finetune_resnet34_custom.yaml \
  --train_data data/my_dataset/train/wav.scp \
  --train_label data/my_dataset/train/utt2spk \
  --reverb_data data/rirs/lmdb \
  --noise_data data/musan/lmdb

# 5. Average checkpoints
python wespeaker/bin/average_model.py \
  --dst_model exp/ResNet34-custom-finetune/models/avg_model.pt \
  --src_path exp/ResNet34-custom-finetune/models \
  --num ${num_avg}

# 6. Test the fine-tuned model
python wespeaker/bin/extract.py \
  --exp_dir exp/ResNet34-custom-finetune \
  --model_path exp/ResNet34-custom-finetune/models/avg_model.pt \
  --wav_scp data/test/wav.scp \
  --device cuda
```

## Comparison with Standard Training

| Aspect | Training from Scratch | Fine-tuning |
|--------|----------------------|-------------|
| **Epochs** | 100-150 | 20-30 |
| **Learning Rate** | 0.1 (SGD) | 0.001-0.01 |
| **Batch Size** | 128-256 | 64-128 |
| **Speed Perturb** | Enabled | Disabled |
| **Layer Freezing** | N/A | Strategic |
| **Data Required** | Large (>100k utts) | Small-Medium (>10k utts) |
| **Training Time** | Days | Hours |

## Additional Resources

- **Example configs**: `examples/voxceleb/v2/conf/finetune_*.yaml`
- **Source code**: `wespeaker/bin/finetune.py`
- **WeSpeaker docs**: https://github.com/wenet-e2e/wespeaker
- **Pretrained models**: https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md

## Citation

If you use this fine-tuning framework in your research, please cite WeSpeaker:

```bibtex
@inproceedings{wang2023wespeaker,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlai and Qian, Yanmin},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

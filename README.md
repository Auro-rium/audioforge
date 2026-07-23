# AudioForge

AudioForge is a reproducible PyTorch framework for environmental sound
classification. It is built to compare models under one explicit data,
feature, training, evaluation, and artifact pipeline rather than hide the
experiment inside a notebook.

## Supported workflows

### FSD50K multilabel classification

The FSD50K path prepares manifests and a stable 200-class label map, loads and
resamples audio to fixed-length mono clips, creates normalized log-Mel features,
and optionally applies waveform and SpecAugment transforms. It supports:

- `scratch_cnn`: a transparent CNN baseline trained from random initialization;
- `ast`: transfer learning from the Hugging Face Audio Spectrogram Transformer,
  adapted via LoRA (`use_lora: true`, default) by training rank-`lora_r`
  adapters on the attention query/value projections plus the classifier head
  while the rest of the ~86M pretrained AST parameters stay frozen. Full
  fine-tuning and full-freeze linear probing remain available as `use_lora:
  false` + `freeze_backbone: false`/`true` respectively, mainly for comparison.

Both models train with a multilabel loss selected via the `loss_fn` training
config key: `bce` (default, `BCEWithLogitsLoss`) or `focal` (class-imbalance-aware,
tunable via `focal_gamma`). Training is implemented with
Hugging Face Accelerate and supports mixed precision, gradient accumulation,
multi-GPU execution, checkpoint resume, periodic validation, and best-model
selection by mAP.

Prepare data and run a smoke test from the repository root:

```bash
python scripts/prepare_fsd50k.py --root data/raw/fsd50k
python -m audioforge.training.train_fsd50k --config configs/fsd50k/smoke.yaml
```

## Inference and serving

FSD50K checkpoints emitted by training embed their model configuration. Event
inference requires the checkpoint and the matching label map:

```bash
python -m audioforge.inference.predict_event audio.wav \
  --checkpoint outputs/smoke_scratch/best/scratch_cnn_best.pt \
  --label-map data/manifests/fsd50k/label_map.json
```

The FastAPI service exposes `/health` and `/predict/event`. Configure the
artifacts through environment variables and start it with:

```bash
AUDIOFORGE_EVENT_CHECKPOINT=... \
AUDIOFORGE_LABEL_MAP=... \
python -m audioforge.serving.api
```

## Reproducibility and artifacts

Configuration files, manifests, label maps, distributed-runtime metadata,
checkpoints, per-evaluation metrics, and benchmark rows are stored explicitly.
Use the smoke and subset configurations for local validation, then use the
multi-GPU scripts for full benchmark runs. Dataset files are not committed to
the repository.

# AudioForge

AudioForge is a reproducible PyTorch framework for environmental sound
classification and machine-sound anomaly detection. It is built to compare
models under one explicit data, feature, training, evaluation, and artifact
pipeline rather than hide the experiment inside a notebook.

## Supported workflows

### FSD50K multilabel classification

The FSD50K path prepares manifests and a stable 200-class label map, loads and
resamples audio to fixed-length mono clips, creates normalized log-Mel features,
and optionally applies waveform and SpecAugment transforms. It supports:

- `scratch_cnn`: a transparent CNN baseline trained from random initialization;
- `ast`: transfer learning from the Hugging Face Audio Spectrogram Transformer.

Both models use multilabel `BCEWithLogitsLoss`. Training is implemented with
Hugging Face Accelerate and supports mixed precision, gradient accumulation,
multi-GPU execution, checkpoint resume, periodic validation, and best-model
selection by mAP.

Prepare data and run a smoke test from the repository root:

```bash
python scripts/prepare_fsd50k.py --root data/raw/fsd50k
python -m audioforge.training.train_fsd50k --config configs/fsd50k/smoke.yaml
```

### DCASE anomaly detection

The DCASE baseline path builds manifests with split, machine type, section,
domain, and normal/anomaly metadata. It provides a deterministic log-Mel
embedding baseline and three anomaly scorers:

- nearest-neighbour memory bank;
- regularized Mahalanobis distance;
- normalized score ensemble.

Example:

```bash
python scripts/prepare_dcase.py --root data/raw/dcase
python -m audioforge.training.train_dcase \
  --manifest data/manifests/dcase2024/all.csv \
  --output outputs/dcase/baseline
```

The resulting prediction CSV follows the DCASE metric module’s schema and can
be evaluated with `audioforge.evaluation.dcase_metrics`.

## Inference and serving

FSD50K checkpoints emitted by training embed their model configuration. Event
inference requires the checkpoint and the matching label map:

```bash
python -m audioforge.inference.predict_event audio.wav \
  --checkpoint outputs/smoke_scratch/best/scratch_cnn_best.pt \
  --label-map data/manifests/fsd50k/label_map.json
```

The FastAPI service exposes `/health`, `/predict/event`, and `/predict/anomaly`.
Configure the
artifacts through environment variables and start it with:

```bash
AUDIOFORGE_EVENT_CHECKPOINT=... \
AUDIOFORGE_LABEL_MAP=... \
AUDIOFORGE_ANOMALY_MODEL=... \
python -m audioforge.serving.api
```

DCASE artifacts can be queried with:

```bash
python -m audioforge.inference.predict_anomaly audio.wav \
  --model outputs/dcase/baseline.npz
```

## Reproducibility and artifacts

Configuration files, manifests, label maps, distributed-runtime metadata,
checkpoints, per-evaluation metrics, and benchmark rows are stored explicitly.
Use the smoke and subset configurations for local validation, then use the
multi-GPU scripts for full benchmark runs. Dataset files are not committed to
the repository.

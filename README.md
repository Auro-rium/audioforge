# AudioForge

AudioForge is a reproducible PyTorch framework for environmental sound
classification. It is built to compare models under one explicit data,
feature, training, evaluation, and artifact pipeline rather than hide the
experiment inside a notebook.

## Prerequisites

`torchaudio.load` (used throughout `audioforge.features.waveform`) requires
the `torchcodec` package, which in turn requires **FFmpeg installed at the
OS level** (not pip-installable) -- on Debian/Ubuntu:

```bash
sudo apt-get install -y ffmpeg
```

Without this, `torchaudio.load` fails with
`ModuleNotFoundError: No module named 'torchcodec'` (if torchcodec itself
is missing) or `RuntimeError: Could not load libtorchcodec` (if FFmpeg's
shared libraries aren't present) even though `uv sync`/`pip install`
completes without error -- this is an OS package, not a Python one, and
`uv`/`pip` have no way to install or check for it.

## Supported workflows

### FSD50K multilabel classification

The FSD50K path prepares manifests and a stable 200-class label map, loads and
resamples audio to fixed-length mono clips, creates normalized log-Mel features,
and optionally applies waveform and SpecAugment transforms. It supports:

- `scratch_cnn`: a transparent CNN baseline (2.4M parameters) trained from
  random initialization. Published checkpoint at
  https://huggingface.co/auro-rirum/audioforge-scratch-cnn-fsd50k with final
  mAP 0.3020 on FSD50K.
- `ast`: transfer learning from the Hugging Face Audio Spectrogram Transformer,
  adapted via LoRA (`use_lora: true`, default) by training rank-`lora_r`
  adapters on the attention query/value projections plus the classifier head
  while the rest of the ~86M pretrained AST parameters stay frozen. Full
  fine-tuning and full-freeze linear probing remain available as `use_lora:
  false` + `freeze_backbone: false`/`true` respectively, mainly for comparison.
  Published checkpoint at https://huggingface.co/auro-rirum/audioforge-ast-fsd50k
  with final mAP 0.5567 on FSD50K.

Both models train with a multilabel loss selected via the `loss_fn` training
config key: `bce` (default, `BCEWithLogitsLoss`) or `focal` (class-imbalance-aware,
tunable via `focal_gamma`). Training is implemented with Hugging Face
Accelerate and supports mixed precision, gradient accumulation, checkpoint
resume, periodic validation, and best-model selection by mAP. The training
scripts (`scripts/train_scratch_cnn.sh`, `scripts/train_ast_2gpu.sh`) default
to a single GPU (`AUDIOFORGE_NUM_PROCESSES=1`); pass a higher value to run
multi-GPU via `accelerate launch`.

Prepare data and run a smoke test from the repository root:

```bash
python scripts/prepare_fsd50k.py --root data/raw/fsd50k
bash scripts/smoke_train.sh       # scratch_cnn
bash scripts/smoke_train_ast.sh   # ast + LoRA
```

Then run the real training scripts:

```bash
AUDIOFORGE_NUM_PROCESSES=1 bash scripts/train_scratch_cnn.sh
AUDIOFORGE_NUM_PROCESSES=1 bash scripts/train_ast_2gpu.sh
```

This repository does not include an inference or serving layer — it covers
data prep, training, evaluation, and publishing trained checkpoints to the
Hugging Face Hub. Load a published checkpoint directly (see the model cards
linked below for a working code snippet) if you need to run inference.

## Publishing to the Hugging Face Hub

Both final models are now published on the Hugging Face Hub:

- **scratch_cnn**: https://huggingface.co/auro-rirum/audioforge-scratch-cnn-fsd50k
  (final mAP 0.3020)
- **ast** (LoRA): https://huggingface.co/auro-rirum/audioforge-ast-fsd50k
  (final mAP 0.5567)

To publish a checkpoint locally, use `scripts/export_hf.py`. Authenticate
first (`hf auth login` -- `huggingface-cli login` is deprecated in current
`huggingface_hub` releases -- or an `HF_TOKEN` env var on the machine you run
this from -- never pass a token as a CLI argument or paste it into a shared
terminal/chat):

```bash
# scratch_cnn: custom architecture, so this stages config.json +
# model.safetensors + a README with a loading snippet before pushing.
python scripts/export_hf.py \
  --checkpoint outputs/fsd50k/scratch_cnn_full/best/scratch_cnn_best.pt \
  --model-type scratch_cnn \
  --repo-id your-hf-username/audioforge-scratch-cnn-fsd50k

# ast: pushes through peft's/transformers' own push_to_hub, so LoRA
# checkpoints publish as a small adapter-only repo pointing back at the
# base AST model, not a full merged copy.
python scripts/export_hf.py \
  --checkpoint outputs/fsd50k/ast_2gpu/best/ast_best.pt \
  --model-type ast \
  --repo-id your-hf-username/audioforge-ast-fsd50k
```

Add `--dry-run` to build/validate the export locally (and, for `scratch_cnn`,
catch any checkpoint/config mismatch) without pushing anything.

## Reproducibility and artifacts

Configuration files, manifests, label maps, distributed-runtime metadata,
per-evaluation metrics, and benchmark rows are committed to `reports/` for
both real training runs (`reports/scratch_cnn_report.md`,
`reports/ast_report.md`, with their loss/eval-curve and per-class-AP plots).
Raw checkpoints (`.pt`, `.safetensors`) are gitignored — they live on the
Hugging Face Hub instead (see above), not in this repository. Dataset files
under `data/` are also not committed; run `scripts/download_fsd50k.sh` and
`scripts/prepare_fsd50k.py` to regenerate them.

Use `configs/fsd50k/smoke.yaml` / `smoke_ast.yaml` for a fast local sanity
check before committing to a full run, and `configs/fsd50k/random_subset.yaml`
for a quick run on a smaller slice of the data.

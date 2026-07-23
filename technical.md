# AudioForge — Technical Documentation

This document explains, in depth, what AudioForge is, how every piece of it
works, why it's built the way it is, and the full infrastructure/deployment
path around it. It exists as a single reference: architecture, math,
training mechanics, evaluation, serving, cloud infrastructure, and the
engineering decisions (and fixes) that got the codebase to its current
state.

---

## 1. What this project is

AudioForge is a reproducible PyTorch framework for **FSD50K multi-label
environmental sound event classification**: given a ~10s audio clip, predict
which of 200 possible sound tags apply (a clip can have zero, one, or many
correct tags simultaneously — "car", "honk", and "street noise" can all be
true of the same clip).

It ships two models under one shared data/feature/training/evaluation
pipeline, deliberately kept comparable:

- `scratch_cnn` — a small CNN trained from random initialization. The
  "what does a from-scratch baseline actually achieve" control.
- `ast` — transfer learning from a pretrained Audio Spectrogram Transformer,
  adapted via LoRA. The "how much does pretraining + parameter-efficient
  adaptation buy you" comparison.

The project previously also included a DCASE machine-anomaly-detection
path; it was removed entirely (see §11) to focus solely on FSD50K
classification.

---

## 2. Repository layout

```
audioforge/
  data/
    fsd50k.py       # raw FSD50K -> manifests + label_map.json
    manifests.py    # manifest read/write primitives (shared schema)
  features/
    waveform.py     # load -> mono -> resample -> crop/pad -> peak-normalize
    logmel.py       # waveform -> log-mel spectrogram, normalization modes
    augment.py      # waveform augment (gain/noise/shift) + SpecAugment
  models/
    scratch_cnn.py  # ScratchAudioCNN: 5-block CNN, from scratch
    ast.py          # ASTAudioClassifier: pretrained AST + optional LoRA
  training/
    trainer.py       # FSD50KTrainer: the Accelerate-based training loop
    train_fsd50k.py  # CLI entrypoint
    losses.py        # bce / focal loss factory
    distributed.py    # Accelerator setup, checkpoint save helpers
  evaluation/
    fsd50k_metrics.py    # mAP, per-class AP, precision/recall/F1
    benchmark_table.py   # render benchmark rows as markdown
  inference/
    predict_event.py  # EventPredictor: load a checkpoint, run inference
  serving/
    api.py           # FastAPI: /health, /predict/event
    gradio_app.py    # optional Gradio demo UI
  utils/
    checkpoint.py, device.py, logging.py, seed.py

configs/fsd50k/       # YAML configs: smoke, scratch_cnn, ast_2gpu, random_subset
scripts/               # data prep, download, training launchers, HF export
reports/               # benchmark markdown + figures/metrics (populated after real runs)
```

---

## 3. Data pipeline

### 3.1 Manifests

`scripts/prepare_fsd50k.py` reads the official FSD50K ground-truth CSVs
(`dev.csv`, `eval.csv`, `vocabulary.csv`) and cross-references every `fname`
against the actual audio files on disk. It produces:

- `train.csv`, `val.csv`, `test.csv` — flat CSVs with columns
  `path, split, labels, duration`, where `labels` is a comma-joined string
  (a clip can have multiple tags — this is a **multi-label**, not
  multi-class, problem).
- `label_map.json` — a stable `{label_to_id, id_to_label}` mapping over the
  200 FSD50K vocabulary labels, so label indices never shift between runs
  (critical: a checkpoint's output index 47 must always mean the same tag).
- `summary.json` / `README.manifest.md` — row counts, duration, missing-audio
  counts per split.

### 3.2 Downloading the raw corpus

`scripts/download_fsd50k.sh` downloads FSD50K directly from its Zenodo
distribution (record `4060432`) and reassembles it into the layout
`audioforge/data/fsd50k.py` expects. Two things matter here that aren't
obvious in isolation:

- **Zenodo splits `dev_audio` and `eval_audio` across multiple files**
  (`.z01`, `.z02`, ..., final `.zip`) because of a per-file size cap.
  Plain `unzip` on just the last part fails — the correct fix is
  `zip -s 0 <final.zip> --out <combined.zip>`, which merges all the parts
  sitting next to it into one valid archive, which can then be `unzip`'d
  normally.
- **The download URL scheme matters and changes over time.** The current
  correct pattern, verified live against the Zenodo API
  (`https://zenodo.org/api/records/4060432`), is
  `https://zenodo.org/api/records/<id>/files/<name>/content` — an older,
  commonly-referenced pattern (`zenodo.org/records/<id>/files/<name>?download=1`)
  is stale. Every one of the 9 files (dev_audio ×6 parts, eval_audio ×2
  parts, ground_truth.zip) was individually verified with a live `HEAD`
  request confirming `200 OK` and a `content-length` matching the API's
  reported byte size before being written into the script — this is not a
  guessed file list.

Download happens directly on the training instance, not the local laptop:
inbound transfer into EC2 is free (AWS only charges egress), so
downloading locally and re-uploading would cost transfer time twice for no
benefit, and offloads the extraction (disk/CPU-heavy: ~30GB compressed,
tens of thousands of small files) onto instance hardware instead of a
resource-constrained laptop.

### 3.3 Waveform preprocessing (`features/waveform.py`)

Pipeline per clip: `load_audio` (torchaudio) → `convert_to_mono` (average
channels if stereo) → `resample_waveform` (to a configured target rate,
16kHz by default) → `crop_or_pad_waveform` (fixed-length: random crop
during training for augmentation-like variety, center crop for val/test so
evaluation is deterministic) → `peak_normalize_waveform` (scale so the peak
amplitude hits a stable target, avoiding loudness being a spurious
confounding signal).

### 3.4 Feature extraction (`features/logmel.py`)

Converts a `[channels, time]` waveform into a `[channels, n_mels, frames]`
log-mel spectrogram via `torchaudio.transforms.MelSpectrogram`, then
`log(mel + log_offset)` for numerical stability (log of exactly zero is
`-inf`), then normalizes:

- `per_sample` mode: zero mean / unit std computed independently per
  spectrogram. Used for `scratch_cnn`.
- `ast` mode: fixed global mean/std matching what HF's `ASTFeatureExtractor`
  uses internally, so log-mel features are comparable to what AST was
  pretrained on.

For the `ast` model path, feature extraction is actually delegated entirely
to `transformers.AutoFeatureExtractor` (loaded from the same pretrained
checkpoint) rather than this module, since AST's exact patch-embedding
input format (128 mel bins × up to 1024 frames) needs to match what the
pretrained weights expect precisely.

### 3.5 Augmentation (`features/augment.py`, training only)

- Waveform-level: random gain (±30%), Gaussian noise injection, small
  circular time-shift.
- Spectrogram-level: SpecAugment-style frequency-band and time-band
  masking (randomly zero out a contiguous band of mel bins or time frames).

Both are explicitly disabled for validation/test — augmenting eval data
would silently corrupt the metrics being used to judge the model.

---

## 4. Models — from first principles

### 4.1 `ScratchAudioCNN` — why it's shaped the way it is

**The core idea: turn sound into a picture.** A raw waveform is a long 1D
list of numbers (160,000 of them for 10s at 16kHz) — an awkward shape for a
CNN, which is built for 2D grids. A log-mel spectrogram *is* a 2D image:
one axis is time, the other is frequency, and "brightness" is energy. A
dog bark isn't a single value anymore, it's a *shape* — a blob with a
characteristic frequency pattern sitting somewhere in this time-frequency
image. That reframes "is there a dog bark in this clip" as "is there a
dog-bark-shaped blob somewhere in this image," which is exactly the
question CNNs are built to answer.

**Why convolution, not a giant fully-connected layer.** Flattening a
`128 × 313` spectrogram (128 mel bins × ~313 time frames for a 10s clip at
16kHz/hop_length=512) into one vector and fully-connecting it to even a
modest hidden layer costs ~10M+ weights before anything useful happens, and
worse, it has no shared notion of "what a pattern looks like" — it would
separately have to learn "dog bark at position X" and "dog bark at
position Y" as unrelated facts. A convolution is a small filter that
**slides across the image reusing the same weights at every position**: if
it learns to recognize part of a bark, it recognizes that pattern wherever
it occurs, for free, because the weights are shared, not duplicated per
position. That's translation invariance, and it's the reason CNNs are both
cheaper and smarter than fully-connected nets for grid-like data.

**The `ConvBlock` design, line by line:**

```python
Conv2d(in, out, kernel_size=3, padding=1, bias=not use_batch_norm)
BatchNorm2d(out)          # if enabled
Activation (GELU)
Conv2d(out, out, kernel_size=3, padding=1, bias=not use_batch_norm)
BatchNorm2d(out)
Activation (GELU)
MaxPool2d(kernel_size=2, stride=2)   # optional, per block
```

- `kernel_size=3, padding=1` is a "same" convolution — output stays the
  same spatial size as input; pooling, not the conv itself, does the
  shrinking.
- **Two stacked 3×3 convs, not one bigger conv**: the classic VGG trick.
  Two 3×3 convs stacked have the same receptive field (5×5 — each output
  pixel "sees" a 5×5 patch of input) as one 5×5 conv, but with fewer
  parameters (2×9=18 vs 25 per channel pair) *and* an extra nonlinearity
  squeezed in between, giving more representational power for the same
  field of view.
- **`BatchNorm2d` after every conv, and why `bias=not use_batch_norm`**:
  BatchNorm re-centers/re-scales each channel's activations toward zero
  mean/unit variance every forward pass, using two learnable numbers per
  channel (`γ` scale, `β` shift). This matters for training stability —
  without it, activations can drift to huge or tiny magnitudes across 5
  stacked blocks, causing gradients to explode or vanish. Since BatchNorm's
  `β` already does the "shift," giving the *preceding* conv its own bias
  would be redundant — two things fighting to do the same job — so the
  conv's bias is disabled whenever BatchNorm follows it.
- **`GELU`, not plain `ReLU`**: nonlinearity is what lets stacking many
  conv layers represent nonlinear functions at all (a stack of purely
  linear operations collapses to being equivalent to one linear operation,
  no matter how many layers). GELU is a smoothed version of ReLU's hard
  zero-cutoff, which tends to help gradient flow slightly, and is the
  modern default across CNNs and transformers alike.
- **`MaxPool2d(2, 2)` on 4 of 5 blocks**: looks at every non-overlapping
  2×2 patch, keeps the strongest value, halving both spatial dimensions.
  This (a) halves compute for every subsequent layer, and (b) makes the
  network care less about *exact* pixel position and more about "is this
  pattern present nearby" — exactly right for tagging, where you don't
  care whether the bark occurred at frame 100 or frame 103.

**The full tower, with real tensor shapes** (batch size `B`, `base_channels=32`):

```
input:                       [B,   1, 128, 313]
Block1 (1→32→32,  pool):     [B,  32,  64, 156]
Block2 (32→64→64, pool):     [B,  64,  32,  78]
Block3 (64→128→128, pool):   [B, 128,  16,  39]
Block4 (128→256→256, pool):  [B, 256,   8,  19]
Block5 (256→256, NO pool):   [B, 256,   8,  19]
```

Channel count doubles every time spatial resolution halves (32→64→128→256)
— a deliberate, standard tradeoff: as the grid shrinks, each remaining
"pixel" needs to encode richer, more abstract information (early layers
detect simple edges/textures, late layers detect whole complex patterns),
so channel count compensates for lost spatial resolution, roughly
balancing compute per block. Block 5 skips pooling — one more round of
refinement at the smallest resolution before spatial information is
discarded entirely.

**`AdaptiveAvgPool2d((1,1))`** collapses `[B,256,8,19]` to `[B,256,1,1]` by
averaging every channel to one number. Two reasons for averaging instead of
flattening the whole grid: (1) it forces the network to learn "is this
pattern present anywhere" rather than "is this pattern at this specific
pixel" — built-in translation invariance at the whole-clip level, which is
exactly the right bias for tagging; (2) it makes the network **input-size
agnostic** — whatever grid size arrives (which changes if `clip_seconds`
changes), it always collapses to exactly `1×1`, so the classifier head
never needs to know or care how many time frames there were.

**Classifier head**: `Flatten → Dropout → Linear(256,128) → GELU → Dropout
→ Linear(128,200)`. A single `Linear(256,200)` can only draw straight
decision boundaries in 256-dimensional space; the extra hidden layer lets
features combine nonlinearly ("feature 12 AND feature 40, but only if
feature 3 is low") before the final 200 independent predictions. Dropout
guards against overfitting at this squeezed 256-number bottleneck, where
it's easy for the network to over-rely on a few specific features.

**Weight initialization**: Kaiming-normal for conv layers (a formula
derived from how variance propagates through ReLU-family nonlinearities,
picked so activation magnitude stays stable across 10 stacked layers at the
very start of training — get this wrong and gradients are useless from
step one), Xavier-uniform for the linear layers, BatchNorm starting at
`weight=1, bias=0` (so BatchNorm starts as the identity function — the
network starts as close to "well-behaved defaults" as possible).

**Output**: raw logits, no sigmoid applied inside the model. Training uses
`BCEWithLogitsLoss`, which fuses sigmoid + the loss computation into one
numerically stable step (via a log-sum-exp identity, avoiding ever
computing `log(sigmoid(x))` directly, which can blow up for very negative
`x`). This is deliberately **sigmoid, not softmax**: softmax forces 200
probabilities to sum to 1 (right for "exactly one of these classes is
true"), which is wrong here — a clip can legitimately be "car" AND "street
noise" AND "honking" simultaneously. Sigmoid treats each of the 200 outputs
as an independent yes/no question, which is what multi-label tagging
actually needs.

**Parameter count**: ~2.4M total at `base_channels=32` (almost all in the
conv blocks; the classifier head is ~59K, a rounding error). Small enough
to train fast from nothing, transparent enough that every design choice
above is a deliberate, explainable decision rather than "whatever the
pretrained checkpoint happened to learn" — an honest floor to measure
whether AST's much larger compute budget is actually earning its keep.

### 4.2 `ASTAudioClassifier` — transfer learning + LoRA

**What AST is**: the Audio Spectrogram Transformer treats a log-mel
spectrogram as an image and applies a Vision Transformer (ViT) architecture
to it — the spectrogram is cut into overlapping 16×16 patches (stride 10),
each patch linearly projected into a 768-dim embedding, plus a learned
positional embedding, then run through 12 transformer encoder layers
(hidden size 768, 12 attention heads, intermediate FFN size 3072). For a
10s clip (128 mel bins × 1024 time frames, matching what the pretrained
checkpoint expects), this produces **~1214 patch tokens** (12 freq-patches
× 101 time-patches + 2 special tokens). `MIT/ast-finetuned-audioset-10-10-0.4593`
was pretrained on AudioSet (~2M clips, ~527 classes); AudioForge takes that
checkpoint and repurposes it for FSD50K's 200 classes by swapping in a new
classification head sized for 200 labels
(`ignore_mismatched_sizes=True` handles the shape mismatch).

#### Transfer learning vs. fine-tuning vs. instruction tuning — precisely

These are three different concepts, and only two apply here:

- **Transfer learning** (what AudioForge does): reusing knowledge learned on
  one task/dataset (AudioSet tagging) for a different task (FSD50K
  tagging). This is the *what*.
- **Fine-tuning** (the mechanism): continuing gradient descent on the
  pretrained weights using new-task data. This is the *how* transfer
  learning happens — and AudioForge supports it in three modes (see below).
- **Instruction tuning** (does **not** apply): an LLM-specific technique —
  fine-tuning a pretrained language model on (instruction, response) pairs
  so it follows natural-language commands (base GPT → ChatGPT-style
  assistant). There's no LLM anywhere in this codebase and no
  instruction-following objective; both AudioForge tasks are classic
  supervised (multi-label BCE) or unsupervised (now-removed anomaly
  scoring) objectives. This concept just doesn't map onto anything here.

#### The three AST training modes

Controlled by `ASTConfig`/`FSD50KTrainConfig` fields `use_lora` and
`freeze_backbone` (mutually exclusive — `use_lora=True` already freezes the
backbone, combining both raises a `ValueError`):

1. **`use_lora=True` (default, recommended)** — LoRA adapters on attention
   query/value projections + a fully-trained classifier head; ~86.5M of
   ~87M backbone parameters stay frozen.
2. **`freeze_backbone=True`** — full-freeze linear probe: only the new
   classifier head trains, the entire backbone stays frozen, no adapters.
3. **Both `False`** — full fine-tuning: every one of ~87M AST parameters
   gets gradients, at a much smaller learning rate (2e-5 in the full-FT
   case vs 2e-4 for LoRA) to avoid destroying pretrained representations.

#### Why LoRA, concretely (not just "it's popular")

**The math**: for a frozen pretrained weight matrix `W` (shape
`[out_features, in_features]`), LoRA doesn't touch `W` at all. Instead it
adds a *low-rank* update alongside it: two small trainable matrices `A`
(shape `[r, in_features]`) and `B` (shape `[out_features, r]`), where `r`
(the "rank," here `8`) is far smaller than `in_features`/`out_features`
(768 each for AST). The effective forward pass becomes
`output = W·x + (B·A)·x · (alpha/r)` — `W` is frozen and untouched, only
`A` and `B` receive gradients. The intuition: adapting a pretrained model
to a new task usually only requires a low-dimensional *correction* to its
existing representations, not a full re-derivation of every weight from
scratch — so a rank-8 update captures most of the useful adaptation at a
tiny fraction of the parameter count of the full matrix.

**Applied here**: `target_modules=["query", "value"]` — the attention
mechanism's query and value projection matrices in each of the 12 encoder
layers (the standard LoRA-on-ViT/BERT-family default; leaving `key`
untouched is a common choice that captures most of the benefit at lower
adapter parameter count). `modules_to_save=["classifier"]` tells `peft` to
keep the new classification head *fully* trainable (not LoRA-adapted, not
frozen) alongside the adapters, since it's a brand-new randomly-initialized
head with no pretrained values to preserve.

**Parameter count, concretely**: per adapted module, LoRA adds
`2 × r × hidden = 2 × 8 × 768 = 12,288` params. With query+value adapted
across 12 layers, that's 24 adapted modules → `24 × 12,288 ≈ 295K` LoRA
params. The classifier head (768→200 linear + LayerNorm) adds ≈155K. Total
trainable ≈ **450K out of ~87M — about 0.5%** of the model.

**Why this beats full fine-tuning here, specifically**: AudioSet
(pretraining data) and FSD50K (fine-tuning target) overlap heavily in
domain — both are general environmental/everyday sound tagging. When the
downstream task is this close to the pretraining distribution, there's
little for full fine-tuning to "relearn"; the marginal benefit of updating
all 87M parameters over updating a well-chosen low-rank correction is
small, while the cost (compute, memory, risk of overfitting/forgetting on
a much smaller fine-tuning set) is real. This isn't a universal LoRA
argument — it's the right call **for this specific pairing** of
pretraining and target task.

**A nuance worth being honest about**: LoRA's VRAM savings scale with how
big the frozen model is. At AST's ~87M-parameter scale, freezing weights
saves real memory (~1GB — the gradients + AdamW optimizer states that
would otherwise be needed for the full 87M params) but it's a modest
saving, not the dramatic 10x+ reduction associated with LoRA on 7B+
parameter LLMs, because activation memory (which scales with batch size ×
sequence length × depth, not with how many params are frozen) dominates
total memory regardless of whether LoRA is used — see §9.1 for the exact
numbers.

---

## 5. Training pipeline (`training/trainer.py`)

### 5.1 Loss functions (`training/losses.py`)

- `bce` (default): plain `nn.BCEWithLogitsLoss` — independent
  sigmoid+binary-cross-entropy per label.
- `focal`: `FocalBCEWithLogitsLoss`, which down-weights already-confident
  predictions via `(1 - p_t)^gamma × BCE`, where `p_t` is the model's
  probability for the *correct* class (high `p_t` → heavily down-weighted
  loss contribution). Useful for FSD50K's long-tailed label distribution,
  where a handful of common tags could otherwise dominate the gradient.

Selected via the `loss_fn` config key (`bce`/`focal`), tunable via
`focal_gamma`, built by a small `build_loss_fn()` factory rather than
hardcoded in the trainer.

### 5.2 The training loop, mechanically

Built on Hugging Face **Accelerate**, so multi-GPU / mixed-precision
support comes from configuration, not hand-rolled DDP code:

- `AdamW` optimizer, filtered to only parameters with `requires_grad=True`
  (this is what makes LoRA's frozen backbone automatically "just work" with
  the optimizer — frozen params are never even passed to it).
- Linear warmup + linear decay learning-rate schedule
  (`get_linear_schedule_with_warmup`), warmup steps = `warmup_ratio ×
  total_update_steps`.
- Gradient accumulation (`accelerator.accumulate(model)`) and gradient
  clipping (`clip_grad_norm_`) for training stability at small per-step
  batch sizes.
- Every `eval_every_steps`, runs a full validation pass; every
  `save_every_steps`, saves a full Accelerate checkpoint
  (`accelerator.save_state`, includes optimizer/scheduler/RNG state, so
  training can resume exactly, not just reload weights).
- Best-model tracking by validation mAP: whenever a new best is found,
  saves an unwrapped, portable model checkpoint (just
  `model_state_dict` + embedded training config + metrics — this is the
  file `inference/predict_event.py` and `scripts/export_hf.py` actually
  consume, distinct from the full Accelerate resume-state checkpoints).

### 5.3 Config (`FSD50KTrainConfig`)

A flat dataclass (not the pydantic `AudioForgeConfig` that used to exist —
that system was built but never actually wired into training/inference/
serving anywhere, so it was removed; see §11). YAML configs under
`configs/fsd50k/` use nested sections (`data:`, `model:`, `features:`,
`training:`, `runtime:`, `augmentation:`) purely for human readability —
`FSD50KTrainConfig.from_dict()` flattens them and filters to known
dataclass fields.

### 5.4 Structured training history (AST only)

AST runs write `training_history.jsonl` in the output directory: one JSON
line per logged training step (`kind: "train_step"`, with `epoch, step,
loss, lr, wall_clock`) and one per evaluation (`kind: "eval"`, with `epoch,
step, loss, mAP, micro_f1, macro_f1, wall_clock`) — both on a shared step
axis, so a loss curve and validation-metric curve can be plotted together
directly from one file, rather than regex-scraping `train_stdout.log` (which
is what `scripts/make_fsd50k_benchmark_row.py` still does for the last
logged loss value — this file is the better source going forward).
`scratch_cnn` deliberately does **not** get this: it's a quick baseline,
and its existing `eval_step_N.json` + `best_metrics.json` are already
enough to tell whether it worked — no need for the extra bookkeeping.

---

## 6. Evaluation (`evaluation/fsd50k_metrics.py`)

- **Per-class Average Precision (AP)**: area under the precision-recall
  curve for one label, threshold-independent. **Explicitly returns `None`
  (not `0`) for any class with zero positive examples in the evaluation
  set**, since AP is mathematically undefined there — faking it as 0 would
  quietly corrupt the mAP average, especially damaging on FSD50K's
  long-tailed label distribution where many classes are rare in any given
  eval slice.
- **mAP**: mean of per-class AP over classes where AP is defined.
- **Precision / Recall / F1**, both **macro** (unweighted mean across
  classes — treats rare and common classes equally) and **micro** (pooled
  across all label predictions — dominated by common classes, closer to
  "overall accuracy").
- All computed at a configurable probability threshold (default 0.5) for
  the precision/recall/F1 family; AP itself doesn't need a threshold.

---

## 7. Inference & Serving

`inference/predict_event.py`'s `EventPredictor` loads a checkpoint and
reconstructs the exact model + preprocessing it was trained with, because
every checkpoint **embeds its own training config** (`extra.config` in the
checkpoint payload) — no separate deployment config file needed beyond the
label map. It correctly distinguishes LoRA checkpoints (reduced state dict
containing only adapter + classifier weights, loaded with `strict=False`
since the rest of the backbone is already correctly loaded from the
pretrained checkpoint via `from_pretrained`) from full-fine-tune/frozen
checkpoints (complete state dict, loaded `strict=True`).

`serving/api.py` is a FastAPI app (`/health`, `/predict/event`) that loads
the model exactly once at app startup (not per-request) and serves
requests against that single loaded instance. `serving/gradio_app.py` is
an optional demo UI wrapping the same predictor.

---

## 8. Publishing to the Hugging Face Hub (`scripts/export_hf.py`)

Two different export paths, because the two models have fundamentally
different relationships to the Hub's tooling:

- **`scratch_cnn`** is a custom architecture with no native `transformers`
  class, so there's no automatic Hub integration to lean on. The script
  builds a plain Hub-compatible repo by hand: `config.json` (the
  architecture hyperparameters), `model.safetensors` (weights, via
  `safetensors.torch.save_file` — safer and faster to load than pickle-based
  `.pt` files), and an auto-generated `README.md` model card (with the
  checkpoint's actual validation mAP/F1 baked in, plus a copy-pasteable
  loading snippet). Before uploading anything, it round-trips the
  checkpoint through the real `create_scratch_cnn(**config)` constructor
  locally, so a config/checkpoint mismatch is caught immediately rather
  than surfacing for whoever downloads it later. Upload itself goes through
  `HfApi().upload_folder()`.
- **`ast`** is pushed through `peft.PeftModel.push_to_hub()` (or plain
  `transformers.PreTrainedModel.push_to_hub()` if trained without LoRA) —
  both are mature, well-documented Hub integrations that already handle
  adapter export and model-card generation correctly, so no custom
  repo-building code was needed. A LoRA push publishes a small
  adapter-only repo (just the ~450K trained parameters) that references the
  base AST model on the Hub, rather than a full merged multi-hundred-MB
  copy — which is both the idiomatic way to share a LoRA adapter and keeps
  the provenance ("this is an adapter on top of X") explicit rather than
  hidden by merging.

Both support `--dry-run` (build/validate the local export without
uploading) and neither hardcodes a destination repo — `--repo-id` is
required, entirely up to whoever runs it.

**Auth**: via `huggingface-cli login` or an `HF_TOKEN` environment
variable on the machine actually running the script — never as a CLI
argument or pasted into a shared terminal, both end up recoverable in
shell history or logs.

---

## 9. AWS infrastructure

### 9.1 VRAM sizing — the actual calculation, not a guess

Three things consume GPU memory during training: **weights** (must exist
for the forward pass regardless of what's frozen), **gradients + optimizer
state** (only for *trainable* parameters — this is what LoRA reduces),
and **activations** (intermediate tensors kept for backprop — scales with
batch size × sequence length × depth, and is usually the dominant cost for
transformers, not raw parameter count).

**`scratch_cnn`**: ~2.4M params, all trainable. Weights+gradients+AdamW
state in fp32 ≈ 38MB total. Activations for a `[B,1,128,313]` input
shrinking to `8×19` by the last block are at most a few hundred MB even at
batch=32. Not the constraint, by a wide margin.

**`ast` + LoRA**, the one that actually matters:

- Weight memory: whole ~87M-param model must be loaded for the forward
  pass regardless of what's frozen → `87M × 4 bytes (fp32) ≈ 348MB`.
- Grad + AdamW state: only for the ~450K trainable params →
  `450K × 4B × 3 (grad + 2 Adam moments) ≈ 5.4MB`. (Full fine-tuning of all
  87M params instead would cost `87M × 4B × 3 ≈ 1.04GB` here instead — the
  actual, quantified LoRA saving at this model size: real, but not the
  dramatic 10x story associated with multi-billion-parameter LLMs.)
- Activations, per sample, fp16 under mixed precision, eager attention,
  sequence length ≈1214 tokens (128 mel bins × 1024 time frames →
  12 freq-patches × 101 time-patches + 2 special tokens):
  - Attention score matrices: `12 heads × 1214² × 2 bytes × 12 layers ≈ 405MB`
  - Hidden-state/FFN intermediate tensors (~8 tensors/layer, one at 4×
    width): `≈ 246MB`
  - **≈ 650MB/sample**

At the configured per-GPU batch size of 4
(`configs/fsd50k/ast_2gpu.yaml`): `~2.6GB` activations + `0.35GB` weights +
`~5MB` grad/opt + `~1–1.5GB` fixed CUDA/cuDNN/allocator overhead ≈
**~4–5GB per GPU**. Full fine-tuning instead of LoRA would only push that to
~5–6GB. (This is a conservative upper bound: setting
`attn_implementation="sdpa"` on `from_pretrained` would cut the attention
term further via PyTorch's fused kernels — a lever not yet used here.)

**Conclusion**: this workload needs single-digit-GB VRAM per GPU, full
stop. Any current-generation data-center GPU has enormous headroom over
what's actually required.

### 9.2 Instance choice

Given VRAM isn't the constraint, instance choice comes down to $/hour and
one AWS-specific wrinkle: GPU instance families jump straight from 1 GPU to
4 GPUs (g5.xlarge/2xlarge/4xlarge/8xlarge/16xlarge are all single-A10G;
g5.12xlarge/24xlarge jump to 4×A10G — no "exactly 2 GPU" shape exists).
Since this workload doesn't need multi-GPU parallelism at all given how
small its footprint is, the chosen instance is a **single-GPU
`g5.xlarge`/`g5.2xlarge` (1× A10G, 24GB)** — comfortable headroom, and
enough room to raise batch size well past the current 4 if desired.

### 9.3 Spot vs. on-demand economics

Live Spot pricing (pulled directly via `aws ec2 describe-spot-price-history`,
not estimated) for `g5.xlarge` in `us-east-1` varies significantly by
Availability Zone for identical hardware: as low as **~$0.46–0.52/hr**
(`us-east-1f`, `us-east-1d`) up to **~$0.93/hr** (`us-east-1a`), against an
on-demand rate around **$1.00/hr** flat regardless of AZ. AZ choice
matters roughly as much as the Spot/on-demand decision itself — at the
cheap end, Spot is saving over 50%, not just the usual "up to" marketing
number.

The trainer already checkpoints periodically and supports
`--resume-from`, so a Spot interruption is a resume, not a lost run — Spot
was chosen for the actual training cost.

### 9.4 Service quotas — why this blocked launch entirely at first

New/low-GPU-usage AWS accounts default to a **vCPU quota of 0** for the
G/VT instance family — `g5.xlarge` fails to launch with a quota error
regardless of billing/credit status until this is raised. This is a
different mechanism from billing credits entirely (checked separately in
Billing → Credits, which can also independently restrict GPU instance
types on promotional/Activate credit grants).

Both **on-demand** (`L-DB2E81BA`, "Running On-Demand G and VT instances")
and **Spot** (`L-3819A6DF`, "All G and VT Spot Instance Requests") quotas
were requested at `8` vCPUs (covers `g5.xlarge`=4 or `g5.2xlarge`=8) —
deliberately both, even though Spot is the intended pricing model, because
they're two entirely separate quota buckets: if Spot capacity is
unavailable in a given AZ at launch time (which happens, especially for
GPU instances) and only the Spot quota were approved, the fallback to
on-demand would itself be blocked pending a second quota request and wait.
Quota requests themselves are free regardless of value — only running
instances (and their attached EBS storage, which bills per GB-month even
while an instance is *stopped*, not just while *running*) actually cost
money.

Both requests landed in `CASE_OPENED` (manual review) rather than
auto-approving, common for GPU quota bumps on accounts without prior GPU
usage history.

### 9.5 Credentials — a live incident, handled

Mid-setup, a long-lived IAM access key (`AKIA...`-prefixed) was pasted
directly into chat. That is treated as a compromised credential
regardless of subsequent handling, since it's now sitting in a chat
transcript in plaintext — the standing recommendation is to rotate/
deactivate any key the moment it's shared through a channel like that, and
prefer `aws configure`/SSO login (entered directly into a terminal, never
pasted into chat) for anything going forward. Separately, of the four AWS
CLI profiles configured on this machine (`default`, `logsage`,
`incidentops-deploy`, `compute-negotiator`), only one had a valid session
at the time, and it belonged to a different, unrelated project's AWS
account — a reminder to always verify `aws sts get-caller-identity`
before assuming a working credential is for the right account.

---

## 10. Engineering history — what was fixed and why

The codebase went through a full audit-and-repair pass before any of the
above infrastructure work started. Key fixes:

- **Smoke test was completely broken.** `configs/fsd50k/smoke.yaml` was an
  empty file (silently falling back to full-dataset defaults instead of a
  16-sample smoke config), while the *actual* smoke-test parameters were
  sitting, misplaced, inside `scripts/smoke_train.sh` — a file with a
  `.sh` extension containing pure YAML and no shebang, which would fail
  immediately if executed as bash (`data:` is not a valid command). Fixed
  by moving the YAML to its correct path and rewriting the script as an
  actual bash runner.
- **`compute_per_class_average_precision` was silently broken**: it
  computed precision/recall/F1 against a hardcoded all-zero prediction
  array regardless of the model's actual scores. It was unused anywhere in
  the codebase (confirmed by grep), so harmless in practice, but a landmine
  for future use — fixed to threshold real scores.
- **`training/losses.py` was fully implemented but never called** — the
  trainer hardcoded `nn.BCEWithLogitsLoss()` directly. Wired in via a
  `loss_fn` config key and a `build_loss_fn()` factory.
- **An entire orphaned config system existed**: `audioforge/config.py`
  defined a pydantic `AudioForgeConfig`/`load_config` system, exported from
  `audioforge/__init__.py` as if it were "the" config — but nothing in
  training, inference, or serving actually imported or used it; the real
  config was (and remains) the separate flat `FSD50KTrainConfig` dataclass
  in `trainer.py`. Removed entirely rather than risk two disconnected
  config systems drifting further apart.
- **The DCASE anomaly-detection path was removed entirely**, per explicit
  direction to focus solely on FSD50K classification. It also had its own
  honesty problems worth noting for the record: DCASE config files named
  `beats_knn.yaml`/`beats_memory_bank.yaml` implied a pretrained BEATs
  audio encoder was in use, but the actual implementation was a classical,
  non-learned log-mel-statistics + k-NN/Mahalanobis-distance baseline —
  and all four DCASE configs produced *identical* output regardless of
  name, since the training script never actually branched on scoring
  method before removal made the question moot.
- **Empty, unreferenced stub files removed**: `data/splits.py`,
  `models/heads.py`, `scripts/export_hf.py` (the last one has since been
  properly (re)written with real functionality — see §8).
- **DCASE inference served up a fresh k-NN refit on every single API
  request** in `serving/api.py`, rebuilding the nearest-neighbor index from
  disk each call — fixed (before removal made it moot) by caching a loaded
  predictor once at app startup, the same pattern `EventPredictor` already
  used correctly.

---

## 11. Current status

As of this document: the codebase is fixed, LoRA-enabled, and has real
data-download/deployment tooling ready, but **no model has actually been
trained yet** — no real dataset is present locally, no checkpoints exist,
`reports/` is empty. The AWS GPU instance is not yet launched (blocked on
the service quota requests in §9.4 clearing). Next real steps, in order:
quota approval → launch a Spot `g5.xlarge`/`g5.2xlarge` → run
`scripts/download_fsd50k.sh` → `scripts/prepare_fsd50k.py` →
`scripts/smoke_train.sh` (validate the fixes actually work on real
hardware before committing to a full run) → full `scratch_cnn` and
`ast`(+LoRA) training runs → `scripts/export_hf.py` to publish both to the
Hugging Face Hub.

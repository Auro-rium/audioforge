data:
  train_manifest: data/manifests/fsd50k/train.csv
  val_manifest: data/manifests/fsd50k/val.csv
  label_map_path: data/manifests/fsd50k/label_map.json
  max_train_samples: 16
  max_val_samples: 8

model:
  model_name: scratch_cnn
  num_labels: 200
  base_channels: 16
  dropout: 0.1

features:
  sample_rate: 16000
  clip_seconds: 10.0
  n_fft: 1024
  hop_length: 512
  n_mels: 128
  normalize_mode: per_sample

training:
  epochs: 1
  batch_size: 2
  eval_batch_size: 2
  num_workers: 0
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_ratio: 0.0
  gradient_accumulation_steps: 1
  mixed_precision: "no"
  max_grad_norm: 1.0
  threshold: 0.5
  eval_every_steps: 999999
  save_every_steps: 999999
  log_every_steps: 1

runtime:
  output_dir: outputs/smoke_scratch
  checkpoint_dir: outputs/smoke_scratch/checkpoints
  seed: 42
  deterministic: false
  log_level: INFO

augmentation:
  waveform_augment: false
  spec_augment: false

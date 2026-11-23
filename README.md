# Phase 1: report-conditioned voxel-flow MRI generation

This folder trains a model that generates a 3D T1-weighted brain MRI from a
radiology report. It contains the complete pipeline, from raw data to a demo app:

download → preprocess → encode reports → select scans → build cache →
train 200 epochs → evaluate and compute FID → Streamlit app

The code is revision 2.3 of the voxel-flow model (commit `9071985` of this
repository). This revision trained the first 200-epoch voxel run. The original
epoch-200 checkpoint loads into this code with all weights matched.
No data, caches, or checkpoints are included. Each step creates its own outputs.

## Contents

- [Terms](#terms)
- [Folder layout](#folder-layout)
- [Step 0. Setup](#step-0-setup)
- [Step 1. Download and preprocess MRI](#step-1-download-and-preprocess-mri)
- [Step 2. Join reports and encode them](#step-2-join-reports-and-encode-them)
- [Step 3. Select the 5,000-scan cohort](#step-3-select-the-5000-scan-cohort)
- [Step 4. Build the voxel cache](#step-4-build-the-voxel-cache)
- [Step 5. Train](#step-5-train)
- [Step 6. Evaluate](#step-6-evaluate)
- [Step 7. Generate one MRI](#step-7-generate-one-mri)
- [Step 8. Compute FID](#step-8-compute-fid)
- [Step 9. Run the Streamlit app](#step-9-run-the-streamlit-app)
- [Step outputs at a glance](#step-outputs-at-a-glance)
- [Differences from the original run](#differences-from-the-original-run)

## Terms

| Term | Meaning |
| --- | --- |
| Voxel flow | A flow-matching generative model. It moves random noise to an MRI in small steps, at the full 240 × 240 × 168 voxel grid. |
| MR-RATE | The source dataset of brain MRI studies and reports, hosted on Hugging Face. Access needs approval. |
| Study / series | One study is one MRI examination with one report. One study has several series (sequences). Each T1 series is one training scan. |
| BioMedCLIP | A frozen medical text encoder. It converts each report into token features that condition the model. |
| EMA | Exponential moving average of the model weights. Evaluation and the app use EMA weights. |
| DDP | PyTorch DistributedDataParallel: one training process per GPU. |
| BF16 | 16-bit floating-point training precision. |
| FID | Fréchet distance between features of real and generated scans. Lower is better. Features come from the 3D MedicalNet ResNet-50. |
| RAS | Axis order Right, Anterior, Superior. All saved volumes use it. |
| MNI template | The standard MNI ICBM 2009c brain template. Preprocessing registers each scan to it. |
| `pending_review` | Processed scans that no person has accepted yet. The original experiment used these scans. |

## Folder layout

```
phase_1/
  README.md, pyproject.toml, uv.lock   Instructions and the pinned environment
  code/
    configs/voxel_flow.yaml            All data paths and training settings
    configs/report_encoder.yaml        Text-encoder revisions and folder
    preprocessing/preprocess.py        Step 1
    scripts/download_assets.py         Step 1
    scripts/prepare_reports.py         Step 2
    scripts/select_cohort.py           Step 3
    scripts/prepare_voxel_cache.py     Step 4
    scripts/voxel_flow.py              Steps 5-7 (train, evaluate, sample)
    scripts/evaluate_voxel_fid.py      Step 8
    scripts/app.py, brain_viewer.html  Step 9
    src/voxel_flow/                    Model, flow, data, training, evaluation
    util/                              MRI checks, report cache, Haar transform, FID
    third_party/                       MedicalNet ResNet definition (from WDM-3D, MIT license)
    tests/
  data/        Created in step 1: MRI, masks, reports, labels, template
  checkpoints/ Created in step 1: downloaded text-encoder weights (not model checkpoints)
  runs/        Created in steps 2 and 5-9: report cache, model checkpoints, results
  cache/       Created in steps 3-5: cohort list, voxel cache, compiler cache
```

**Rules:**
- Run every command from `phase_1/`.
- Keep data, caches, and runs outside `code/`. The code raises an error for paths inside `code/`.
- To change a path or a training setting, edit `code/configs/voxel_flow.yaml`.

## Step 0. Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. On Hugging Face, request and accept access to the MR-RATE dataset.
3. Install the environment and log in:

```bash
cd phase_1
uv sync --frozen            # Installs all packages, including PyTorch 2.14.1+
uv run hf auth login
uv run pytest -q            # CPU tests only; no downloads
```

Training needs GPUs. The original run used 2 × RTX A6000 (48 GB).

## Step 1. Download and preprocess MRI

```bash
uv run python code/preprocessing/preprocess.py plan --root data/mr_rate_t1
uv run python code/scripts/download_assets.py --root data/mr_rate_t1

# Pilot: process 10 training studies first.
uv run python code/preprocessing/preprocess.py process --root data/mr_rate_t1 \
  --split train --max-studies 10

# Full run: process every planned study.
uv run python code/preprocessing/preprocess.py process --root data/mr_rate_t1
```

- `plan` downloads the dataset tables and pins the dataset revision in `data/mr_rate_t1/policy.json`.
- `download_assets.py` downloads the pathology labels (same revision), the MNI
  template, and the BioMedCLIP encoder (into `checkpoints/`).
- `process` registers each T1 scan to the template and pads it to
  240 × 240 × 168 RAS. It saves the MRI (divided by its brain maximum) and a
  binary brain mask (`*_ras_mask.nii.gz`).
- After the pilot, open a few saved MRI and mask files in a NIfTI viewer. Check
  that the brain is complete and the mask fits it. Then run the full command.
- Splits are by patient. All series of one patient stay in one split.
- Hugging Face timeouts on a slow network: set `HF_HUB_ETAG_TIMEOUT=60` and
  `HF_HUB_DOWNLOAD_TIMEOUT=300`, and add `--download-workers 2`. Rerun the same
  command to continue.

**Outputs:** `data/mr_rate_t1/standardized_candidates.csv` (all processed scans,
`pending_review`), `data/mr_rate_t1/reports.csv`, and `data/mr_rate_t1/plan.csv`
(exclusion reasons).

## Step 2. Join reports and encode them

```bash
uv run python code/scripts/prepare_reports.py \
  --manifest data/mr_rate_t1/standardized_candidates.csv \
  --output-dir runs/reports
```

- Adds the 37 official pathology labels and the study report to each series.
  Series without labels, and scans that preprocessing marked `held`, are excluded.
- Contrast rule: series without a suffix are noncontrast; suffix `-2` is
  postcontrast; suffixes above 2 are excluded. This rule was verified for this
  dataset only.
- Encodes every report with BioMedCLIP. Long reports are split into chunks of at
  most 256 tokens. No text is cut off.

**Outputs in `runs/reports/`:**
- `report_inputs.csv`: retained series, with labels, report, and contrast. Step 3 reads it.
- `excluded.csv`: excluded series and the reason for each.
- `report_token_cache/`: report token features. Training reads it.

## Step 3. Select the 5,000-scan cohort

```bash
uv run python code/scripts/select_cohort.py
```

- Reads `runs/reports/report_inputs.csv`.
- Samples 4,500 training scans (seed 42) and 500 validation scans (seed 43)
  without replacement, inside the existing splits. No test scans are selected.

**Output:** `cache/voxel_flow_inputs/manifest.csv`.

## Step 4. Build the voxel cache

```bash
uv run python code/scripts/prepare_voxel_cache.py plan     # Disk-space estimate only
uv run python code/scripts/prepare_voxel_cache.py build
uv run python code/scripts/prepare_voxel_cache.py verify --manifest cache/voxel_cache/manifest.csv
```

- Reads `cache/voxel_flow_inputs/manifest.csv` and the MRI and mask files.
- Each target is the MRI divided by the median of positive voxels inside the
  brain mask. Only nonzero voxels are stored, with an exact round-trip check.

**Output:** `cache/voxel_cache/` with `manifest.csv` and `targets/*.npz`.

## Step 5. Train

Two GPUs (the original setting):

```bash
CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_COMPILE_THREADS=2 \
uv run torchrun --standalone --nproc-per-node=2 code/scripts/voxel_flow.py train \
  --config code/configs/voxel_flow.yaml
```

One GPU (same effective batch of 32; accumulation 8 is set automatically):

```bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=2 \
uv run python code/scripts/voxel_flow.py train \
  --config code/configs/voxel_flow.yaml --batch-size 4
```

Resume an interrupted run: add `--resume runs/voxel_flow_v2_3_5000_seed42/training/last.pt`
to the same command.

Quick check on a small cohort before the full run: add
`--smoke-test --output-dir runs/voxel_flow_smoke/training`.

- Settings: 200 epochs, 32 scans per update, 141 updates per epoch, AdamW with
  learning rate 1e-4, EMA decay 0.995, BF16, gradient clip 10.
- After every epoch, training also runs a small automatic evaluation
  (2 validation patients, 16 steps) and saves a preview image.

**Outputs in `runs/voxel_flow_v2_3_5000_seed42/`** (folder set by `run_dir` in the config):
- `training/`: `config.yaml`, `manifest.csv`, `train.jsonl` (loss log),
  `last.pt`, `best_loss.pt` (lowest validation loss),
  `epoch_010.pt`, `epoch_025.pt`, `epoch_050.pt`, `epoch_100.pt`, `epoch_150.pt`,
  `epoch_200.pt`, and `preview_epoch_NNN.png`.
- `evaluation_epoch_NNN/`: the automatic evaluation for each epoch.
  The app in step 9 uses `evaluation_epoch_200/`.

## Step 6. Evaluate

A larger evaluation than the automatic one, on held-out validation patients:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python code/scripts/voxel_flow.py evaluate \
  --config code/configs/voxel_flow.yaml \
  --checkpoint runs/voxel_flow_v2_3_5000_seed42/training/epoch_200.pt \
  --weights ema --split validation --num-samples 32 \
  --output-dir runs/voxel_flow_v2_3_5000_seed42/evaluation_epoch_200_full
```

- Outputs: generated and real NIfTI files, `metrics.json`, per-case CSV files,
  and `preview.png`.
- Use `--split test` only for the final result. Never tune settings on test data.
- These metrics measure image statistics. They do not prove correct anatomy or
  agreement with the report.

## Step 7. Generate one MRI

```bash
CUDA_VISIBLE_DEVICES=0 uv run python code/scripts/voxel_flow.py sample \
  --config code/configs/voxel_flow.yaml \
  --checkpoint runs/voxel_flow_v2_3_5000_seed42/training/epoch_200.pt \
  --weights ema --study-uid STUDY_UID \
  --geometry data/mr_rate_t1/IMAGE_PATH --contrast noncontrast \
  --output-dir runs/sample
```

Take `STUDY_UID` and `IMAGE_PATH` from one row of `cache/voxel_cache/manifest.csv`
(columns `study_uid` and `image_path`). Use a `validation` row so that the report
was not used in training. `--geometry` gives only the voxel grid of the output;
the image itself is not used.

## Step 8. Compute FID

Generates one MRI for each validation scan (500 in the full cohort; seed 42 + row number),
then compares them with the real scans:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc-per-node=2 \
  code/scripts/evaluate_voxel_fid.py --output runs/fid_epoch_200
uv run python code/scripts/evaluate_voxel_fid.py --output runs/fid_epoch_200 --aggregate
```

- Default checkpoint: `epoch_200.pt` in the config's `run_dir`. Use `--checkpoint` to change it.
- Default preprocessing: per-volume clipping at the 0.1% and 99.9% intensity
  quantiles, min-max scaling, padding to 256³, then average pooling to 128³.
  Add `--native` to the first command to keep the native 240 × 240 × 168 grid.
- **Result:** `runs/fid_epoch_200/metrics.json`, key `medicalnet_frechet_distance`.

## Step 9. Run the Streamlit app

Needs step 5 to be complete (`training/epoch_200.pt` and `evaluation_epoch_200/`).
To use another checkpoint, add `--checkpoint PATH` after `--evaluation-dir`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run streamlit run code/scripts/app.py -- \
  --evaluation-dir runs/voxel_flow_v2_3_5000_seed42/evaluation_epoch_200
```

**Use the trained 200-epoch checkpoint without training.** On this server, the
original checkpoint is in `checkpoints/voxel_flow_v2_3_5000_seed42/`:
`training/epoch_200.pt`, its `config.yaml` and `manifest.csv`, and the saved
`evaluation_epoch_200/`. Its config reads the original report-token cache (15 GB)
and voxel cache (32 GB) by absolute path in the parent workspace; they are not
copied. `checkpoints/` is not in git.

```bash
CUDA_VISIBLE_DEVICES=0 uv run streamlit run code/scripts/app.py -- \
  --evaluation-dir checkpoints/voxel_flow_v2_3_5000_seed42/evaluation_epoch_200
```

- **Generate & explore**: pick a validation report or write your own, set the
  seed and steps, and generate an MRI. View it in 3D, compare it with the real
  scan, and download the NIfTI file and its settings.
- **Reports**: the exact report used for the displayed MRI.
- **Saved evaluation**, **Metrics**, **Settings**: browse the saved evaluation.
- Custom reports are encoded on the CPU. The 3D viewer (NiiVue) loads from a
  CDN, so the browser needs internet access.

## Step outputs at a glance

| Step | Reads | Writes |
| --- | --- | --- |
| 1 | MR-RATE on Hugging Face | `data/mr_rate_t1/standardized_candidates.csv`, `reports.csv`, MRI and masks; `checkpoints/` |
| 2 | `standardized_candidates.csv`, `reports.csv`, labels | `runs/reports/report_inputs.csv`, `runs/reports/report_token_cache/` |
| 3 | `runs/reports/report_inputs.csv` | `cache/voxel_flow_inputs/manifest.csv` |
| 4 | `cache/voxel_flow_inputs/manifest.csv`, MRI and masks | `cache/voxel_cache/` |
| 5 | `cache/voxel_cache/`, `runs/reports/report_token_cache/` | `runs/voxel_flow_v2_3_5000_seed42/` |
| 6-8 | `training/epoch_200.pt` | the `--output-dir` or `--output` you give |
| 9 | `training/epoch_200.pt`, `evaluation_epoch_200/` | nothing (downloads only) |

## Differences from the original run

- **Cohort IDs may differ.** The original 5,000 scans were sampled from an
  intermediate Stage-1 file. Here they come from `runs/reports/report_inputs.csv`.
  The method and seeds are the same, but the selected scans can differ if the
  candidate lists differ.
- **One check removed.** The original report step also compared its output with
  separately saved report embeddings. Those files are not created here.
- **Scan status.** The scans are `pending_review`, as in the original run
  (`exploratory_pending_review: true`). For person-accepted scans, use
  `data/mr_rate_t1/samples.csv` in step 2 and set
  `exploratory_pending_review: false` in `code/configs/voxel_flow.yaml`.
- **Not included:** diagnostic and sweep scripts, v3 experiments, the older
  Stage-1/Stage-2 models, and mask repair for old outputs.

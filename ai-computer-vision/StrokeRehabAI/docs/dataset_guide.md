# Dataset Guide

## Unified sample format

Every dataset — public or custom — is converted into the same
per-sample `.npz` schema, so the rest of the pipeline
(`datasets/rehab_dataset.py`, `training/`) is dataset-agnostic:

```python
{
    "landmarks": np.ndarray,   # (T, 33, 3) float64, MediaPipe joint order
    "visibility": np.ndarray,  # (T, 33) float64, per-joint MediaPipe visibility
    "label": int,              # unified quality class (see datasets/label_processor.py)
    "label_dict": str,         # full UnifiedLabel as a string (exercise, regression target, etc.)
    "metadata": dict,          # {sample_id, subject_id, exercise_type, source_dataset, ...}
    "features": dict,          # full clinical feature suite, see feature_extraction/feature_pipeline.py
}
```

Conversion pipeline (`datasets/dataset_converter.py` ->
`datasets/sample_converter.py:convert_video_to_sample`):

```
raw video --> VideoProcessor            (frame extraction, resolution/FPS
                                          normalization, corrupted/duplicate-
                                          frame detection; datasets/video_processor.py)
          --> LandmarkCache              (cached MediaPipe BlazePose extraction,
                                          gap-filled for missed detections;
                                          datasets/landmark_cache.py)
          --> FeatureExtractionPipeline  (joint angles, kinematics, clinical
                                          descriptors, rep count; feature_extraction/)
          --> LabelProcessor             (unifies exercise/quality/regression/
                                          clinician-score/binary labels;
                                          datasets/label_processor.py)
          --> one unified .npz
```

Every conversion run also produces a JSON preprocessing report
(`datasets/preprocessing_report.py`) at
`outputs/evaluation_reports/<dataset>_preprocessing_report.json`,
recording video/subject/exercise/frame counts, label distribution, and
every missing file, missing annotation, corrupted file, and duplicate
encountered.

## Supported datasets

### This project's own recordings ("custom") — fully supported

```
raw_dir/<subject_id>/<exercise_name>/<trial_name>.mp4
raw_dir/labels.csv   # columns: subject_id, exercise_name, trial_name, clinician_score
```

```python
from pathlib import Path
from datasets.dataset_converter import DatasetConverter

DatasetConverter(output_dir="data/processed").convert_custom(
    Path("data/raw/my_recordings"), labels_csv=Path("data/raw/my_recordings/labels.csv")
)
```

This is the only converter path exercised against real (synthetic, in
CI) video end-to-end — see `tests/test_dataset_converter.py`.

### UI-PRMD
- **URL:** https://www.webpages.uidaho.edu/ui-prmd/
- **Expected local path:** `data/raw/ui_prmd`
- **License note:** Requires manual download per the dataset's usage terms.
- **Converter status:** best-effort. `DatasetConverter.convert_ui_prmd`
  discovers video files matching the documented `m<exercise>_s<subject>_e<episode>.<ext>`
  naming convention and routes them through the standard pipeline. UI-PRMD's
  primary data is Kinect/Vicon joint-angle/position *files* rather than
  video for every recording; if you'd rather use that native skeletal
  data directly (bypassing MediaPipe entirely), that requires a
  dedicated reader — not yet implemented (see the `TODO` in
  `datasets/dataset_converter.py`).

### KIMORE
- **URL:** https://vrai.dii.univpm.it/content/kimore-dataset
- **Expected local path:** `data/raw/kimore`
- **License note:** Requires registration; not auto-downloadable.
- **Converter status:** best-effort. `convert_kimore` expects
  `data/raw/kimore/<subject_id>/.../*.mp4` plus a CSV/JSON annotation
  file somewhere under that tree with a subject-id column and a
  clinical-score column (several common column-name variants are tried
  automatically — see `DatasetConverter._load_flexible_annotations`).
  Clinician scores are normalized assuming KIMORE's documented 0-100
  scale.

### IntelliRehabDS
- **URL:** https://zenodo.org/records/4610859
- **Expected local path:** `data/raw/intellirehab_ds`
- **License note:** Available on Zenodo under its stated license.
- **Converter status:** best-effort, same subject/exercise-folder +
  flexible-annotation discovery as KIMORE, assuming a 0-1 correctness
  scale.

### Stroke Rehabilitation Exercise Dataset
- **URL:** https://data.niaid.nih.gov/resources?id=mendeley_ygpdzx52g2
- **Expected local path:** `data/raw/stroke_rehab_mendeley`
- **License note:** Hosted via the NIAID data portal (originally
  Mendeley) — verify license terms before redistribution.
- **Converter status:** best-effort, same pattern, assuming a 0-10
  clinician rating scale.

> **Why "best-effort"?** None of these four datasets are bundled with
> this project, so their converters could not be validated against
> real raw files — only against their public documentation. Each
> converter is defensively coded (missing files/annotations are
> recorded in the preprocessing report, not silently ignored) and
> clearly marks its layout assumptions with `TODO` comments in
> `datasets/dataset_converter.py`. Once you have real raw data,
> inspect a handful of files/annotations and adjust the column-name
> lists / filename patterns as needed — the underlying
> video->landmarks->features->label pipeline does not change.

## Workflow

```powershell
# 1. See instructions + expected path for a dataset
python -m datasets.download_dataset --dataset UI-PRMD

# 2. Manually download from the printed URL and extract into the printed path

# 3. Confirm it's present
python -m datasets.dataset_checker --dataset UI-PRMD

# 4. Convert (Python, not yet a CLI — see snippets above)

# 5. Check final corpus statistics
python -m datasets.dataset_statistics
```

## Splitting

`training/dataset_loader.py` splits at the **file (subject) level**
before building any windowed dataset — this matters because
augmentation (temporal crop/drop/stretch) changes how many fixed-length
windows a sample produces, so window-level splitting after augmentation
would silently misalign indices across differently-augmented dataset
instances. `configs/datasets.yaml -> split_by: "subject"` (default)
additionally guarantees no patient's data crosses a split boundary.

## Data representation

`configs/datasets.yaml -> data_representation` controls the tensor
layout dataloaders produce; in practice this is auto-selected from
`configs/model.yaml -> model.architecture` via
`models/model_factory.py:data_representation_for`:

- `"graph"` (ST-GCN, default): `(in_channels, sequence_length, num_joints)`
- `"flat"` (LSTM baseline): `(sequence_length, num_joints * in_channels)`

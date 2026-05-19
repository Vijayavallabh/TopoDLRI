# TOPODLRI Project

## Prerequisites
- Linux or macOS
- Python 3.12
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
- Clone this  repo:
```bash
git clone https://github.com/Vijayavallabh/TopoDLRI
cd TopoDLRI
```

- Create and use a Python 3.12 virtual environment with `uv`, then install dependencies from `requirements.txt`:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Dataset: Latest Fundus Photography (TES/DRS)

Pairs **Topcon Maestro2** (domain `TES`, side `A`) with **iCare Eidon** (domain `DRS`, side `B`).

### Source Layout

```text
<src_dir>/
  topcon_maestro2/           # TES — Topcon Maestro2 (domain A)
    Complete/          <PID>/  *.dcm
    Anomalous/         <PID>/  *.dcm
  icare_eidon/               # DRS — iCare Eidon (domain B)
    Complete/          <PID>/  *.dcm
    Anomalous/         <PID>/  *.dcm
  flagged_participants.xlsx   # (optional)
```

### Pipeline: `prepare_dataset.py`

Reads raw DICOMs from `icare_eidon/` and `topcon_maestro2/`, pairs by participant ID + laterality across devices, and writes `train_A`/`train_B`/`test_A`/`test_B` directories.

```bash
python prepare_dataset.py --src_dir datasets/latest --dst_dir datasets/eye
python prepare_dataset.py --src_dir datasets/latest --dst_dir datasets/eye --exclude_flagged
python prepare_dataset.py --src_dir datasets/latest --dst_dir datasets/eye --test_split 0.2
python prepare_dataset.py --src_dir datasets/latest --dst_dir datasets/eye --format dcm
```

Output filenames: `{PID}_TES_{OD|OS}.png` (A / Topcon Maestro2) and `{PID}_DRS_{OD|OS}.png` (B / iCare Eidon).

### Key Options

- `--format png|dcm`: output format — `png` converts DICOM to PNG (default), `dcm` copies raw DICOMs (faster)
- `--test_split <float>`: fraction of participants held out for test (default: `0.15`)
- `--exclude_flagged`: drops PIDs listed in `flagged_participants.xlsx` (default: `False`)
- `--flagged_xlsx <path>`: path to flagged participants file (defaults to `src_dir/flagged_participants.xlsx`)

### Preprocessing: `crop_images.py`

Removes black borders from Topcon Maestro2 (domain A) images by cropping 300 px from the left and right edges. Operates on `train_A` and `test_A`.

```bash
python crop_images.py
```

### Watermark Removal: `remove_watermarks.py`

Removes iCare Eidon watermarks from the bottom-left and bottom-right corners of domain B images (`train_B`, `test_B`). Pixels in those regions with all RGB channels > 20 are set to black.

```bash
python remove_watermarks.py
```

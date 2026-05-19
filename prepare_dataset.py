#!/usr/bin/env python3
"""
prepare_dataset.py


Workflow:
  1. Process the Complete/ folder  — convert every DICOM to PNG (or copy DCM)
     with TES/DRS domain tokens, keeping only cross-device paired eyes.
  2. Process the Anomalous/ folder — same conversion, but also filter out
     participants that have no matching laterality across devices.
  3. Merge the (cleaned) anomalous samples into the complete samples.
  4. Split the merged pool into train / test sets.

Output layout:
    dst_dir/
      train_A/  — Topcon Maestro2 (TES token)
      train_B/  — iCare Eidon    (DRS token)
      test_A/
      test_B/

Usage:
    python prepare_dataset.py --src_dir datasets/latest --dst_dir datasets/eye
    python prepare_dataset.py --format dcm          # faster, copies .dcm as-is
    python prepare_dataset.py --exclude_flagged     # drop anomalous participants
"""

import argparse
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import pydicom
from PIL import Image
from pydicom.pixels.processing import convert_color_space


DOMAIN_A_TOKEN = "TES"
DOMAIN_B_TOKEN = "DRS"
LATERALITY_MAP = {"l": "OS", "r": "OD"}

# Match a standalone laterality token in a filename stem split by '_'.
_LAT_RE = re.compile(r'^[lr]$')


# ── pair verification (mirrors aligned_dataset._pair_identity_from_stem) ─────


def _stem(path: Path) -> str:
    return path.stem


def _pair_identity_from_stem(stem: str) -> str:
    """Canonical key that strips domain markers (TES/DRS) so A/B sides match."""
    base_stem, _, aug_suffix = stem.partition("__")
    parts = base_stem.split("_")
    domain_tokens = {"TES", "DRS", "A", "B", "IMG", "LABEL"}
    filtered = [p for p in parts if p not in domain_tokens]
    if not filtered:
        filtered = parts
    key = "_".join(filtered)
    if aug_suffix:
        return f"{key}__{aug_suffix}"
    return key


def _pair_identity(path: Path) -> str:
    return _pair_identity_from_stem(_stem(path))


def _build_unique_map(paths: list[Path], key_fn) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for p in paths:
        key = key_fn(p)
        if key in mapping:
            raise ValueError(f'Duplicate pair key "{key}" for files: {mapping[key]} and {p}')
        mapping[key] = p
    return mapping


def _align_b_to_a(a_paths: list[Path], b_paths: list[Path]) -> list[Path]:
    if len(a_paths) != len(b_paths):
        raise ValueError(
            f"Pairing mismatch: {len(a_paths)} files in A vs {len(b_paths)} files in B"
        )
    a_stems = [_stem(p) for p in a_paths]
    b_stem_map = _build_unique_map(b_paths, _stem)
    if all(s in b_stem_map for s in a_stems):
        return [b_stem_map[s] for s in a_stems]
    a_keys = [_pair_identity(p) for p in a_paths]
    b_key_map = _build_unique_map(b_paths, _pair_identity)
    missing = [k for k in a_keys if k not in b_key_map]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"Pairing mismatch: could not find B files for A keys: {sample}")
    return [b_key_map[k] for k in a_keys]


def verify_output_pairs(dst: Path, phase: str, ext: str):
    """Verify every file in {phase}_A has a matching pair in {phase}_B."""
    a_dir = dst / f"{phase}_A"
    b_dir = dst / f"{phase}_B"
    a_paths = sorted(a_dir.glob(f"*.{ext}"))
    b_paths = sorted(b_dir.glob(f"*.{ext}"))
    if not a_paths and not b_paths:
        return
    aligned_b = _align_b_to_a(a_paths, b_paths)
    bad = [(a, b) for a, b in zip(a_paths, aligned_b) if _pair_identity(a) != _pair_identity(b)]
    if bad:
        sample = "\n".join(f"  {a.name} <-> {b.name}" for a, b in bad[:5])
        raise ValueError(f"Output pair verification FAILED:\n{sample}")
    print(f"  ✓ {phase}: {len(a_paths)} pairs verified")


# ── helpers ──────────────────────────────────────────────────────────────────


def scan_device_dir(device_dir: Path) -> dict:
    """Scan a device's folder (Complete/ or Anomalous/).

    Returns {pid: {'l': [DicomPath, ...], 'r': [DicomPath, ...]}}.
    """
    result: dict[int, dict[str, list[Path]]] = {}
    if not device_dir.exists():
        return result
    for pdir in sorted(device_dir.iterdir()):
        if not pdir.is_dir():
            continue
        try:
            pid = int(pdir.name)
        except ValueError:
            continue
        by_lat: dict[str, list[Path]] = defaultdict(list)
        for f in sorted(pdir.glob("*.dcm")):
            for token in f.stem.split("_"):
                if _LAT_RE.match(token):
                    by_lat[token].append(f)
                    break
        if by_lat:
            result[pid] = dict(by_lat)
    return result


def pick_best(file_list: list[Path]) -> Path:
    """When duplicates exist, keep the first file (sorted by name)."""
    return file_list[0]


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale any numeric array fully into uint8 [0, 255]."""
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    mn, mx = arr_f.min(), arr_f.max()
    if mx > mn:
        arr_f = (arr_f - mn) / (mx - mn) * 255.0
    else:
        arr_f = np.zeros_like(arr_f, dtype=np.float32)
    return arr_f.clip(0, 255).astype(np.uint8)


# JPEG transfer syntaxes where pydicom's pillow handler decodes to RGB
_JPEG_TS_UIDs = {
    "1.2.840.10008.1.2.4.50",   # JPEG Baseline (Process 1)
    "1.2.840.10008.1.2.4.51",   # JPEG Extended (Process 2 & 4)
    "1.2.840.10008.1.2.4.57",   # JPEG Lossless, Non-Hierarchical
    "1.2.840.10008.1.2.4.70",   # JPEG Lossless, SV1
}


def _is_jpeg_encoded(ds) -> bool:
    """Check if the DICOM uses a JPEG transfer syntax."""
    uid = getattr(ds.file_meta, "TransferSyntaxUID", None)
    return uid is not None and uid in _JPEG_TS_UIDs


def dcm_to_png_array(dcm_path: Path) -> np.ndarray:
    """Read DICOM and return correct RGB uint8 array.

    For JPEG-encoded DICOMs, pydicom's pillow handler already decodes to RGB,
    so no further colour-space conversion is needed.
    """
    ds = pydicom.dcmread(str(dcm_path))
    arr: np.ndarray = ds.pixel_array

    # ── handle planar configuration (C, H, W) → (H, W, C) ──
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # ── convert colour space ──
    # Only convert YBR→RGB for *uncompressed* DICOMs.  JPEG-encoded ones are
    # already decoded to RGB by pydicom's pillow_handler (the DICOM header
    # still says YBR_FULL_422, but pixel_array contains RGB).
    pi = getattr(ds, "PhotometricInterpretation", None)
    if pi and pi.upper().startswith("YBR") and not _is_jpeg_encoded(ds):
        arr = convert_color_space(arr, pi, "RGB")

    # ── rescale to full 8-bit range ──
    arr = _rescale_to_uint8(arr)

    # ── ensure 3-channel RGB ──
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    else:
        c = arr.shape[2]
        if c == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        elif c >= 4:
            arr = arr[:, :, :3]

    return arr


def _convert_worker(args: tuple) -> str:
    """Worker for parallel pool: convert one A/B pair."""
    pid, lat, a_src, b_src, dst, fmt, phase = args
    eye_tag = LATERALITY_MAP[lat]
    a_dst = dst / f"{phase}_A" / f"{pid}_{DOMAIN_A_TOKEN}_{eye_tag}.{fmt}"
    b_dst = dst / f"{phase}_B" / f"{pid}_{DOMAIN_B_TOKEN}_{eye_tag}.{fmt}"
    a_dst.parent.mkdir(parents=True, exist_ok=True)
    b_dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "dcm":
        shutil.copy2(a_src, a_dst)
        shutil.copy2(b_src, b_dst)
    else:
        Image.fromarray(dcm_to_png_array(a_src)).save(a_dst)
        Image.fromarray(dcm_to_png_array(b_src)).save(b_dst)
    return f"PID {pid} {eye_tag}"


def process_pairs(
    icare: dict, topcon: dict, dst: Path, fmt: str, phase: str,
    label: str, drop_unpaired: bool = True,
) -> dict:
    """Convert cross-device pairs in parallel.

    Only processes participants present on BOTH devices with at least one
    common laterality.  When *drop_unpaired* is True (default), participants
    that cannot be paired are skipped.
    """
    tasks = []
    for pid in sorted(set(icare.keys()) | set(topcon.keys())):
        icare_lats = icare.get(pid, {})
        topcon_lats = topcon.get(pid, {})
        common = set(icare_lats.keys()) & set(topcon_lats.keys())
        if drop_unpaired and not common:
            continue
        for lat in sorted(common):
            tasks.append((
                pid, lat,
                pick_best(topcon_lats[lat]),
                pick_best(icare_lats[lat]),
                dst, fmt, phase,
            ))

    n = len(tasks)
    if n == 0:
        return {"participants": 0, "pairs": 0}

    print(f"  {label}: {n} paired images ({n} A + {n} B)")
    done, seen_pids = 0, set()
    with ProcessPoolExecutor(max_workers=4) as pool:
        fut = {pool.submit(_convert_worker, t): t for t in tasks}
        for f in as_completed(fut):
            msg = f.result()
            done += 1
            seen_pids.add(fut[f][0])
            if done % 10 == 0 or done == n:
                print(f"    [{done}/{n}] {msg}")

    return {"participants": len(seen_pids), "pairs": n}


def print_breakdown(label: str, icare: dict, topcon: dict):
    """Print per-participant image counts."""
    all_pids = sorted(set(icare.keys()) | set(topcon.keys()))
    paired = 0
    for pid in all_pids:
        ic = ", ".join(f"{k}={len(v)}" for k, v in sorted(icare.get(pid, {}).items()))
        tc = ", ".join(f"{k}={len(v)}" for k, v in sorted(topcon.get(pid, {}).items()))
        common = sorted(set(icare.get(pid, {})) & set(topcon.get(pid, {})))
        ps = "/".join(common) if common else "NO PAIR"
        if common:
            paired += 1
        print(f"  PID {pid:>4}: iCare [{ic}]  Topcon [{tc}]  => {ps}")
    print(f"  → {label}: {len(all_pids)} participants, {paired} can pair, "
          f"{len(all_pids) - paired} skipped")


# ── main ─────────────────────────────────────────────────────────────────────


def load_flagged_pids(path: Path) -> set[int]:
    """Read the flagged participants xlsx and return the set of flagged PIDs."""
    if not path.exists():
        print(f"  [info] No flagged participants file: {path}")
        return set()
    try:
        df = pd.read_excel(path, sheet_name="Flagged Participants")
        pids = set(df.iloc[:, 0].dropna().astype(int))
        print(f"  Flagged participants loaded: {len(pids)} PIDs from {path.name}")
        return pids
    except Exception as e:
        print(f"  [warn] Could not read flagged participants {path}: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset from raw DICOM folders"
    )
    parser.add_argument(
        "--src_dir", type=Path, default=Path("datasets/latest"),
        help="Source directory (contains icare_eidon/ and topcon_maestro2/)",
    )
    parser.add_argument(
        "--dst_dir", type=Path, default=Path("datasets/eye"),
        help="Output directory (train_A/B, test_A/B will be created)",
    )
    parser.add_argument("--flagged_xlsx", type=Path, default=None)
    parser.add_argument(
        "--test_split", type=float, default=0.15,
        help="Fraction of participants held out for test (default: 0.15)",
    )
    parser.add_argument(
        "--format", choices=["png", "dcm"], default="png",
        help="Output format (dcm = copy as-is, faster; png = convert)",
    )
    parser.add_argument(
        "--exclude_flagged", action="store_true", default=False,
        help="Exclude participants listed in the flagged xlsx from the merged pool "
             "(default: False, flagged participants are included if pairable)",
    )
    args = parser.parse_args()

    if args.flagged_xlsx is None:
        args.flagged_xlsx = args.src_dir / "flagged_participants.xlsx"

    if not args.src_dir.exists():
        print(f"Error: src_dir not found: {args.src_dir}")
        sys.exit(1)

    if pydicom is None and args.format == "png":
        print("Error: pydicom + Pillow needed for PNG.  Use --format dcm or "
              "pip install pydicom Pillow")
        sys.exit(1)

    ext = args.format
    print(f"Source:      {args.src_dir}")
    print(f"Destination: {args.dst_dir}")
    print(f"Format:      {ext}")
    print(f"Test split:  {args.test_split}")
    print(f"Exclude flagged: {args.exclude_flagged}")

    # ─────────────────────────────────────────────────────────────────────
    #  1. SCAN all four subdirectories
    # ─────────────────────────────────────────────────────────────────────
    icare_comp = scan_device_dir(args.src_dir / "icare_eidon" / "Complete")
    topcon_comp = scan_device_dir(args.src_dir / "topcon_maestro2" / "Complete")
    icare_anom = scan_device_dir(args.src_dir / "icare_eidon" / "Anomalous")
    topcon_anom = scan_device_dir(args.src_dir / "topcon_maestro2" / "Anomalous")

    print(f"\niCare Complete:  {len(icare_comp)} participants")
    print(f"Topcon Complete: {len(topcon_comp)} participants")
    print(f"iCare Anomalous: {len(icare_anom)} participants")
    print(f"Topcon Anomalous:{len(topcon_anom)} participants")

    # ─────────────────────────────────────────────────────────────────────
    #  2. PRINT BREAKDOWN for anomalous (informational)
    # ─────────────────────────────────────────────────────────────────────
    print("\n── Anomalous participant breakdown ──")
    print_breakdown("Anomalous", icare_anom, topcon_anom)

    # ─────────────────────────────────────────────────────────────────────
    #  3. BUILD MERGED PARTICIPANT POOL
    # ─────────────────────────────────────────────────────────────────────
    #    Complete: all participants present on BOTH devices (already clean)
    #    Anomalous: only those with at least one common laterality
    # ─────────────────────────────────────────────────────────────────────
    comp_pool = set(icare_comp.keys()) & set(topcon_comp.keys())
    anom_pool = {
        pid for pid in (set(icare_anom.keys()) | set(topcon_anom.keys()))
        if set(icare_anom.get(pid, {})) & set(topcon_anom.get(pid, {}))
    }
    merged_pool = sorted(comp_pool | anom_pool)
    print(f"\n── Participant pool ──")
    print(f"  Complete:  {len(comp_pool)}")
    print(f"  Anomalous: {len(anom_pool)} (filtered to pairable only)")
    print(f"  Merged:    {len(merged_pool)}")

    # ─────────────────────────────────────────────────────────────────────
    #  4. FILTER FLAGGED PARTICIPANTS
    # ─────────────────────────────────────────────────────────────────────
    flagged_pids = set()
    if args.exclude_flagged:
        flagged_pids = load_flagged_pids(args.flagged_xlsx)
        if flagged_pids:
            before = len(merged_pool)
            merged_pool = [p for p in merged_pool if p not in flagged_pids]
            print(f"  Excluded {before - len(merged_pool)} flagged participants")

    # ─────────────────────────────────────────────────────────────────────
    #  5. SPLIT INTO TRAIN / TEST
    # ─────────────────────────────────────────────────────────────────────
    split = max(1, int(len(merged_pool) * (1 - args.test_split)))
    train_pids = set(merged_pool[:split])
    test_pids = set(merged_pool[split:])
    print(f"  Train: {len(train_pids)}  Test: {len(test_pids)}")

    # ─────────────────────────────────────────────────────────────────────
    #  6. PROCESS — convert DICOMs and write to train_*/test_*
    # ─────────────────────────────────────────────────────────────────────
    for d in ["train_A", "train_B", "test_A", "test_B"]:
        (args.dst_dir / d).mkdir(parents=True, exist_ok=True)

    print("\n── Processing Complete participants ──")
    for phase_name, pid_set in [("train", train_pids), ("test", test_pids)]:
        ic = {p: icare_comp[p] for p in pid_set if p in icare_comp}
        tc = {p: topcon_comp[p] for p in pid_set if p in topcon_comp}
        s = process_pairs(ic, tc, args.dst_dir, ext, phase_name,
                          f"Complete ({phase_name})", drop_unpaired=False)
        print(f"     participants={s['participants']} pairs={s['pairs']}")

    print("\n── Processing Anomalous participants ──")
    for phase_name, pid_set in [("train", train_pids), ("test", test_pids)]:
        ic = {p: icare_anom[p] for p in pid_set if p in icare_anom}
        tc = {p: topcon_anom[p] for p in pid_set if p in topcon_anom}
        s = process_pairs(ic, tc, args.dst_dir, ext, phase_name,
                          f"Anomalous ({phase_name})", drop_unpaired=True)
        print(f"     participants={s['participants']} pairs={s['pairs']}")

    # ─────────────────────────────────────────────────────────────────────
    #  7. VERIFY PAIRS
    # ─────────────────────────────────────────────────────────────────────
    print("\n── Verifying output pairs ──")
    for phase in ["train", "test"]:
        verify_output_pairs(args.dst_dir, phase, ext)

    # ─────────────────────────────────────────────────────────────────────
    #  8. SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    expected_dirs = ["train_A", "train_B", "test_A", "test_B"]
    actual_dirs = sorted(d.name for d in args.dst_dir.iterdir() if d.is_dir())
    print(f"  Output dirs: {actual_dirs}")
    if set(actual_dirs) != set(expected_dirs):
        print(f"  [warn] expected only {expected_dirs}, got {actual_dirs}")
    for phase in ["train", "test"]:
        nA = len(list((args.dst_dir / f"{phase}_A").glob(f"*.{ext}")))
        nB = len(list((args.dst_dir / f"{phase}_B").glob(f"*.{ext}")))
        print(f"  {phase}_A: {nA}   {phase}_B: {nB}  (paired: {min(nA, nB)})")
    total = sum(
        len(list((args.dst_dir / p).glob(f"*.{ext}")))
        for p in expected_dirs
    )
    print(f"  Total images: {total}  (paired sets: {total // 2})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
copy_data_dhcp.py
Convert dHCP rel3 BIDS data (nii.gz) to MRtrix .mif format for downstream processing.

Sources:
  DWI + mask : rel3_dhcp_dmri_shard_pipeline/<sub>/<ses>/dwi/
  T2w, segmentations, bias-corrected T2w : rel3_dhcp_anat_pipeline/<sub>/<ses>/anat/

Output per session ($OUTPUT_DIR/<sub>/<ses>/):
  postmc-dwi.mif       – DWI with gradient table embedded (from nii.gz + bvec/bval)
  reconmask.mif        – brain mask (diffusion space)
  T2w.mif              – T2-weighted image (optional, skipped if source missing)
  T2w_restore.mif      – bias-corrected T2w (for improved registration)
  drawem9.mif          – 9-tissue Draw-EM segmentation (for ACT 5tt generation)
  drawem87.mif         – 87-label Draw-EM parcellation (for connectome generation)
  T2w_brain_mask.mif   – brain mask in T2w/anatomical space

Usage:
  ./copy_data_dhcp.py --dwi-source /path/to/dmri_pipeline --anat-source /path/to/anat_pipeline \\
      --output-dir /path/to/output --scan-info /path/to/scan_info.csv
  ./copy_data_dhcp.py --dwi-source ... --anat-source ... --output-dir ... --scan-info ... --dry-run
  ./copy_data_dhcp.py --dwi-source ... --anat-source ... --output-dir ... --scan-info ... sub-EXAMPLE
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path

EXPECTED_VOLUMES = 300
MANUAL_EXCLUSIONS = {
    "sub-CC00769XX19": "anat and DWI acquired in different sessions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dHCP rel3 BIDS data to MRtrix .mif format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dwi-source", type=Path, required=True,
        help="Path to dHCP rel3 dMRI SHARD pipeline derivatives.",
    )
    parser.add_argument(
        "--anat-source", type=Path, required=True,
        help="Path to dHCP rel3 anat pipeline derivatives.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for converted .mif files.",
    )
    parser.add_argument(
        "--scan-info", type=Path, required=True,
        help="Path to neonatal scan info CSV (for radiology exclusions).",
    )
    parser.add_argument(
        "--nthreads", type=int, default=os.cpu_count() or 1,
        help="Number of MRtrix threads (default: all cores).",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without converting.",
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        metavar="subject_id",
        help="Subject IDs to process (e.g. sub-EXAMPLE). Defaults to all subjects.",
    )
    return parser.parse_args()


def load_radiology_exclusions(csv_path: Path, excluded_scores: set[int] = {4, 5}) -> set[str]:
    """Return subject IDs with radiology_score in excluded_scores."""
    excluded = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_str = row.get("radiology_score", "").strip()
            try:
                score = int(score_str)
            except ValueError:
                continue
            if score in excluded_scores:
                excluded.add(row["src_subject_id"].strip())
    return excluded


def get_nvols(nii_path: Path) -> int | None:
    """Return the number of volumes (4th dimension) in a NIfTI file via mrinfo."""
    result = subprocess.run(
        ["mrinfo", "-size", str(nii_path)],
        capture_output=True, text=True,
    )
    parts = result.stdout.split()
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            pass
    return None


def convert_file(
    label: str,
    src: Path,
    dst: Path,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
    nthreads: int = 1,
) -> str:
    """
    Run mrconvert src -> dst with optional extra args.

    Returns one of: "skipped", "converted", "warned"
    """
    if extra_args is None:
        extra_args = []

    if dst.exists():
        print(f"    [skip] {label} already exists")
        return "skipped"
    if not src.exists():
        print(f"    [WARN] {label} source not found: {src}")
        return "warned"

    cmd = ["mrconvert"] + extra_args + [str(src), str(dst), "-nthreads", str(nthreads), "-quiet"]

    if dry_run:
        print(f"    [dry]  {' '.join(cmd)}")
        return "skipped"

    print(f"    [conv] {label} ...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True)
        return "converted"
    except subprocess.CalledProcessError as e:
        print(f"    [ERROR] mrconvert failed for {label}: {e}")
        dst.unlink(missing_ok=True)
        return "warned"


def main() -> None:
    args = parse_args()

    dwi_source = args.dwi_source
    anat_source = args.anat_source
    output_dir = args.output_dir
    scan_info = args.scan_info
    nthreads = args.nthreads

    if args.dry_run:
        print("=== DRY RUN MODE - No files will be converted ===\n")

    # ---------- Build subject list ----------
    if args.subjects:
        subjects = sorted(args.subjects)
    else:
        subjects = sorted(p.name for p in dwi_source.glob("sub-*") if p.is_dir())

    radiology_excluded = load_radiology_exclusions(scan_info)
    total_subjects = len(subjects)
    total_sessions = sum(
        1 for s in subjects
        for _ in (dwi_source / s).glob("ses-*/") if (dwi_source / s).is_dir()
    )
    n_radiology = sum(1 for s in subjects if s in radiology_excluded)
    n_manual = sum(1 for s in subjects if s in MANUAL_EXCLUSIONS)
    subjects = [s for s in subjects if s not in radiology_excluded and s not in MANUAL_EXCLUSIONS]
    volume_skipped = 0
    converted = 0
    processed = 0
    issues: list[tuple[str, str]] = []  # (subject/session, description)

    for subject in subjects:
        sub_dwi_dir = dwi_source / subject

        if not sub_dwi_dir.is_dir():
            msg = "DWI subject directory not found"
            print(f"[WARN] {subject}: {msg}")
            issues.append((subject, msg))
            continue

        for ses_dir in sorted(sub_dwi_dir.glob("ses-*/")):
            session = ses_dir.name
            out_path = output_dir / subject / session

            # --- Volume check ---
            dwi_nii = ses_dir / "dwi" / f"{subject}_{session}_desc-preproc_dwi.nii.gz"
            if dwi_nii.exists():
                nvols = get_nvols(dwi_nii)
                if nvols is not None and nvols != EXPECTED_VOLUMES:
                    print(f"{subject}/{session}  [DROPPED: {nvols} volumes != {EXPECTED_VOLUMES} — skipping]")
                    volume_skipped += 1
                    continue

            processed += 1
            print(f"{subject}/{session}")

            # --- DWI (nii.gz + bvec/bval → mif) ---
            dwi_bvec = ses_dir / "dwi" / f"{subject}_{session}_desc-preproc_dwi.bvec"
            dwi_bval = ses_dir / "dwi" / f"{subject}_{session}_desc-preproc_dwi.bval"
            dwi_out  = out_path / "postmc-dwi.mif"

            ses_key = f"{subject}/{session}"

            if dwi_bvec.exists() and dwi_bval.exists():
                result = convert_file(
                    "postmc-dwi.mif", dwi_nii, dwi_out,
                    extra_args=["-fslgrad", str(dwi_bvec), str(dwi_bval)],
                    dry_run=args.dry_run,
                    nthreads=nthreads,
                )
                if result == "converted":
                    converted += 1
                elif result == "warned":
                    issues.append((ses_key, "DWI mrconvert failed or source missing"))
            else:
                print(f"    [WARN] bvec/bval not found")
                issues.append((ses_key, "bvec/bval not found"))

            # --- Brain mask ---
            mask_nii = ses_dir / "dwi" / f"{subject}_{session}_desc-brain_mask.nii.gz"
            mask_out = out_path / "reconmask.mif"
            result = convert_file("reconmask.mif", mask_nii, mask_out, dry_run=args.dry_run, nthreads=nthreads)
            if result == "converted":
                converted += 1
            elif result == "warned":
                issues.append((ses_key, "reconmask source not found or conversion failed"))

            # --- T2w (optional) ---
            t2_nii = anat_source / subject / session / "anat" / f"{subject}_{session}_T2w.nii.gz"
            t2_out = out_path / "T2w.mif"
            if t2_nii.exists():
                result = convert_file("T2w.mif", t2_nii, t2_out, dry_run=args.dry_run, nthreads=nthreads)
                if result == "converted":
                    converted += 1
                elif result == "warned":
                    issues.append((ses_key, "T2w conversion failed"))
            else:
                print(f"    [WARN] T2w not found: {t2_nii}")
                issues.append((ses_key, f"T2w not found (anat sessions: {[p.name for p in (anat_source / subject).glob('ses-*') if p.is_dir()]})"))

            # --- Bias-corrected T2w (for improved registration) ---
            t2_restore_nii = anat_source / subject / session / "anat" / f"{subject}_{session}_desc-restore_T2w.nii.gz"
            t2_restore_out = out_path / "T2w_restore.mif"
            result = convert_file("T2w_restore.mif", t2_restore_nii, t2_restore_out, dry_run=args.dry_run, nthreads=nthreads)
            if result == "converted":
                converted += 1
            elif result == "warned":
                issues.append((ses_key, "T2w_restore source not found or conversion failed"))

            # --- Draw-EM 9-tissue segmentation (for ACT / 5tt generation) ---
            drawem9_nii = anat_source / subject / session / "anat" / f"{subject}_{session}_desc-drawem9_dseg.nii.gz"
            drawem9_out = out_path / "drawem9.mif"
            result = convert_file("drawem9.mif", drawem9_nii, drawem9_out, dry_run=args.dry_run, nthreads=nthreads)
            if result == "converted":
                converted += 1
            elif result == "warned":
                issues.append((ses_key, "drawem9 source not found or conversion failed"))

            # --- Draw-EM 87-label parcellation (for connectome generation) ---
            drawem87_nii = anat_source / subject / session / "anat" / f"{subject}_{session}_desc-drawem87_dseg.nii.gz"
            drawem87_out = out_path / "drawem87.mif"
            result = convert_file("drawem87.mif", drawem87_nii, drawem87_out, dry_run=args.dry_run, nthreads=nthreads)
            if result == "converted":
                converted += 1
            elif result == "warned":
                issues.append((ses_key, "drawem87 source not found or conversion failed"))

            # --- Brain mask in T2w/anatomical space ---
            t2_mask_nii = anat_source / subject / session / "anat" / f"{subject}_{session}_desc-brain_mask.nii.gz"
            t2_mask_out = out_path / "T2w_brain_mask.mif"
            result = convert_file("T2w_brain_mask.mif", t2_mask_nii, t2_mask_out, dry_run=args.dry_run, nthreads=nthreads)
            if result == "converted":
                converted += 1
            elif result == "warned":
                issues.append((ses_key, "T2w brain mask source not found or conversion failed"))

            print()

    # ---------- Summary ----------
    print("=" * 40)
    print("Done.")
    print(f"  Subjects found             : {total_subjects}")
    print(f"  Sessions found             : {total_sessions}")
    print(f"  Excluded due to:")
    print(f"    - Radiology score 4/5    : {n_radiology} subjects")
    print(f"    - Incomplete volumes     : {volume_skipped} sessions")
    for sub, reason in MANUAL_EXCLUSIONS.items():
        print(f"    - {sub}: {reason}")
    print(f"  Subjects remaining         : {len(subjects)}")
    print(f"  Sessions remaining         : {processed}")
    print(f"  Files converted            : {converted}")
    print(f"  Issues                     : {len(issues)}")
    if issues:
        print("\nIssue details:")
        for ses_key, desc in issues:
            print(f"  {ses_key}: {desc}")
    print("=" * 40)


if __name__ == "__main__":
    main()

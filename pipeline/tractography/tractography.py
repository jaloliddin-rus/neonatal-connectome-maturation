#!/usr/bin/env python3
"""tractography.py

Run whole-brain tractography with multiple parameter configurations
for neonatal dHCP subjects using MSMT-CSD WM FODs.

Usage:
    python tractography.py --dhcp-dir /path/to/dhcp --scratch-dir /path/to/scratch
    python tractography.py --dhcp-dir /path/to/dhcp --scratch-dir /path/to/scratch sub-EXAMPLE/ses-00000
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KNOWN_ODF_METHODS = {
    "max_pietsch_Ao",
    "max_pietsch_Ay",
    "max_pietsch_combined",
    "max_pietsch_dual_sift2w",
    "baseline_msmt",
}

DEFAULT_ODF_METHODS = [
    "max_pietsch_combined",
    "max_pietsch_dual_sift2w",
    "baseline_msmt",
]

TRACTOGRAPHY_PARAMS = {
    "cutoff": 0.05,
    "angle": 45,
    "step": 0.5,
    "minlength": 10,
    "maxlength": 250,
    "n_streamlines": 10_000_000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def have(cmd: str) -> bool:
    """Return True if *cmd* is on PATH."""
    return shutil.which(cmd) is not None


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run *cmd*, raising on failure and forwarding stdout/stderr."""
    print(f"   $ {shlex.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def is_valid_tck(tck: Path) -> bool:
    """Return True if *tck* exists, is non-empty, and tckinfo succeeds."""
    if not tck.is_file() or tck.stat().st_size == 0:
        return False
    try:
        subprocess.run(
            ["tckinfo", str(tck)],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def read_file_value(path: Path) -> str:
    """Read and return the stripped contents of a single-value file, or 'NA'."""
    try:
        return path.read_text().strip() or "NA"
    except OSError:
        return "NA"


def sum_weights_file(path: Path) -> str:
    """Sum all numeric values in *path* and return as a string, or 'NA'.

    Non-numeric and comment lines (e.g. MRtrix3 header metadata) are skipped.
    """
    try:
        total = 0.0
        count = 0
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for token in stripped.split():
                    try:
                        total += float(token)
                        count += 1
                    except ValueError:
                        continue
        return f"{total:.6f}" if count > 0 else "NA"
    except OSError:
        return "NA"


def detect_nthreads() -> int:
    """Detect number of hardware threads available."""
    return min(os.cpu_count() or 1, 8)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubjectConfig:
    subj_dir: Path
    mask: Path
    force: bool
    nthreads: int
    odf_methods: list[str]
    act: Path | None = None  # 5tt image for ACT (None = no ACT)

    # Derived
    subject_id: str = ""
    session_id: str = ""
    tract_base: Path = field(default_factory=lambda: Path())

    def __post_init__(self):
        self.subject_id = self.subj_dir.parent.name
        self.session_id = self.subj_dir.name
        self.tract_base = self.subj_dir / "tractography"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def prepare_wmfod(subj_dir: Path, odf_method: str, force: bool) -> Path | None:
    """Determine (and if necessary create) the WM FOD for a given ODF method.

    For combined methods, creates the combined FOD if necessary.
    Returns None if the required FOD files are missing for combined methods.
    For other methods, returns the expected path without checking existence
    (the caller is responsible for verifying the file exists).
    """
    odf_est = subj_dir / "odf_estimation"

    if odf_method == "max_pietsch_Ao":
        return odf_est / "max_pietsch" / "wmfod_Ao_norm.mif"

    if odf_method == "max_pietsch_Ay":
        return odf_est / "max_pietsch" / "wmfod_Ay_norm.mif"

    if odf_method in ("max_pietsch_combined", "max_pietsch_dual_sift2w"):
        ao_fod = odf_est / "max_pietsch" / "wmfod_Ao_norm.mif"
        ay_fod = odf_est / "max_pietsch" / "wmfod_Ay_norm.mif"
        if not ao_fod.is_file() or not ay_fod.is_file():
            print("WARNING: Max Pietsch Ao or Ay FOD missing for combined method")
            print(f"         Ao: {ao_fod}")
            print(f"         Ay: {ay_fod}")
            print("         Skipping this method.")
            return None

        combined = odf_est / "max_pietsch" / "wmfod_combined_Ao_Ay.mif"
        if not force and combined.is_file():
            print(f"  Combined Ao+Ay FOD already exists - {combined}")
        else:
            print("  Creating combined Ao+Ay FOD (stored in odf_estimation/max_pietsch/)...")
            try:
                run(["mrcalc", str(ay_fod), str(ao_fod), "-add", str(combined), "-force"])
            except subprocess.CalledProcessError:
                print("   ERROR: mrcalc failed to create combined FOD")
                combined.unlink(missing_ok=True)
                return None
        return combined

    # Default: assume wmfod_norm.mif naming (msmt methods)
    return odf_est / odf_method / "wmfod_norm.mif"


def run_tckgen(
    wmfod: Path,
    tck: Path,
    mask: Path,
    params: dict,
    nthreads: int,
    force: bool,
    act: Path | None = None,
) -> bool:
    """Run tckgen.  Returns True on success, False on failure."""
    if not force and is_valid_tck(tck):
        print("   Tractogram already exists and is valid - skipping tckgen")
        return True

    if not force and tck.is_file():
        print("   Tractogram exists but is invalid - regenerating")

    print("   Running tckgen...")
    cmd = [
        "tckgen", str(wmfod), str(tck),
        "-algorithm", "iFOD2",
        "-seed_dynamic", str(wmfod),
        "-step", str(params["step"]),
        "-angle", str(params["angle"]),
        "-cutoff", str(params["cutoff"]),
        "-minlength", str(params["minlength"]),
        "-maxlength", str(params["maxlength"]),
        "-select", str(params["n_streamlines"]),
        "-nthreads", str(nthreads),
        "-force",
    ]

    if act is not None:
        cmd += ["-act", str(act), "-backtrack"]
    else:
        cmd += ["-mask", str(mask)]

    try:
        run(cmd)
    except subprocess.CalledProcessError:
        print("   ERROR: tckgen failed - cleaning up partial output")
        tck.unlink(missing_ok=True)
        return False

    return True


def run_sift2(
    tck: Path,
    fod: Path,
    weights_file: Path,
    mu_file: Path,
    mask: Path,
    nthreads: int,
    force: bool,
    label: str = "SIFT2",
    act: Path | None = None,
) -> bool:
    """Run tcksift2.  Returns True on success, False on failure."""
    if not is_valid_tck(tck):
        print(f"   {label}: skipping - tractogram missing or invalid")
        return False

    if not force and weights_file.is_file():
        print(f"   {label}: weights already exist - skipping")
        return True

    print(f"   Running {label}...")
    cmd = [
        "tcksift2", str(tck), str(fod), str(weights_file),
        "-out_mu", str(mu_file),
        "-nthreads", str(nthreads),
        "-force",
    ]
    if act is not None:
        cmd += ["-act", str(act)]
    else:
        cmd += ["-proc_mask", str(mask)]

    try:
        run(cmd)
    except subprocess.CalledProcessError:
        print(f"   ERROR: {label} failed - cleaning up partial output")
        weights_file.unlink(missing_ok=True)
        mu_file.unlink(missing_ok=True)
        return False

    return True


def process_odf_method(cfg: SubjectConfig, odf_method: str) -> bool:
    """Process a single ODF method for one subject.  Returns True on success."""
    print()
    print("=" * 42)
    print(f"Processing ODF method: {odf_method}")
    print("=" * 42)

    # Resolve WM FOD
    wmfod = prepare_wmfod(cfg.subj_dir, odf_method, cfg.force)
    if wmfod is None or not wmfod.is_file():
        if wmfod is not None:
            print(f"WARNING: WM FOD not found for method {odf_method}: {wmfod}")
            print("         Skipping this method.")
        return False

    print(f"WMFOD : {wmfod}")

    outdir = cfg.tract_base / odf_method
    outdir.mkdir(parents=True, exist_ok=True)

    # dual_sift2w reuses the tractogram generated by max_pietsch_combined
    # (same FOD, same parameters - no need to run tckgen twice)
    if odf_method == "max_pietsch_dual_sift2w":
        tck = cfg.tract_base / "max_pietsch_combined" / "tractogram.tck"
        print(f"  Reusing tractogram from max_pietsch_combined: {tck}")
    else:
        tck = outdir / "tractogram.tck"

    sift2_weights = outdir / "sift2_weights.txt"
    sift2_mu_file = outdir / "sift2_mu.txt"

    if odf_method != "max_pietsch_dual_sift2w":
        print()
        print("==> Running tractography")
        print(
            f"==> cutoff={TRACTOGRAPHY_PARAMS['cutoff']}, angle={TRACTOGRAPHY_PARAMS['angle']}, "
            f"step={TRACTOGRAPHY_PARAMS['step']}, minlength={TRACTOGRAPHY_PARAMS['minlength']}, "
            f"maxlength={TRACTOGRAPHY_PARAMS['maxlength']}, select={TRACTOGRAPHY_PARAMS['n_streamlines']}"
        )

    # -- tckgen --
    if odf_method == "max_pietsch_dual_sift2w":
        # Tractogram is shared with max_pietsch_combined - no tckgen needed
        if not is_valid_tck(tck):
            print(f"   ERROR: Shared tractogram not found: {tck}")
            print("          Run max_pietsch_combined first.")
            return False
        print("   Reusing existing tractogram (shared with max_pietsch_combined)")
    else:
        if not run_tckgen(wmfod, tck, cfg.mask, TRACTOGRAPHY_PARAMS, cfg.nthreads, cfg.force, act=cfg.act):
            return False

    # -- Standard SIFT2 (skipped for dual SIFT2 method) --
    if odf_method != "max_pietsch_dual_sift2w":
        if not run_sift2(
            tck, wmfod, sift2_weights, sift2_mu_file,
            cfg.mask, cfg.nthreads, cfg.force, act=cfg.act,
        ):
            return False

    # -- Dual SIFT2 for max_pietsch_dual_sift2w --
    if odf_method == "max_pietsch_dual_sift2w":
        print()
        print("   === Running DUAL SIFT2 (maturation decomposition) ===")

        odf_est = cfg.subj_dir / "odf_estimation" / "max_pietsch"
        ay_fod = odf_est / "wmfod_Ay_norm.mif"
        ao_fod = odf_est / "wmfod_Ao_norm.mif"

        sift2_weights_y = outdir / "sift2_weights_y.txt"
        sift2_mu_y = outdir / "sift2_mu_y.txt"
        sift2_weights_o = outdir / "sift2_weights_o.txt"
        sift2_mu_o = outdir / "sift2_mu_o.txt"

        # --- Ay (young/immature) ---
        if not run_sift2(
            tck, ay_fod, sift2_weights_y, sift2_mu_y,
            cfg.mask, cfg.nthreads, cfg.force, label="SIFT2_y", act=cfg.act,
        ):
            return False

        print(f"   SIFT2_y mu: {read_file_value(sift2_mu_y)}")
        print(f"   SIFT2_y sum_weights: {sum_weights_file(sift2_weights_y)}")

        # --- Ao (old/mature) ---
        if not run_sift2(
            tck, ao_fod, sift2_weights_o, sift2_mu_o,
            cfg.mask, cfg.nthreads, cfg.force, label="SIFT2_o", act=cfg.act,
        ):
            return False

        print(f"   SIFT2_o mu: {read_file_value(sift2_mu_o)}")
        print(f"   SIFT2_o sum_weights: {sum_weights_file(sift2_weights_o)}")

        print("   === DUAL SIFT2 complete ===")

    return True


def process_subject(cfg: SubjectConfig) -> bool:
    """Process all ODF methods for a single subject.

    Returns True on success (even partial), False on total failure.
    """
    if not cfg.subj_dir.is_dir():
        print(f"ERROR: Subject directory not found: {cfg.subj_dir}")
        return False

    if cfg.force:
        print("[tract] FORCE mode enabled - will reprocess existing outputs")

    if not cfg.mask.is_file():
        print(f"ERROR: Brain mask not found: {cfg.mask}")
        return False

    cfg.tract_base.mkdir(parents=True, exist_ok=True)

    print()
    print(f"{cfg.subject_id}/{cfg.session_id}")
    print(f"MASK  : {cfg.mask}")
    print(f"ACT   : {cfg.act if cfg.act else 'disabled'}")
    print(f"ODF METHODS: {' '.join(cfg.odf_methods)}")
    print(f"NTHREADS: {cfg.nthreads}")

    # Process each ODF method
    methods_failed = 0
    for odf_method in cfg.odf_methods:
        if not process_odf_method(cfg, odf_method):
            methods_failed += 1

    if methods_failed == len(cfg.odf_methods):
        print(f"ERROR: All {methods_failed} ODF method(s) failed.")
        return False
    if methods_failed > 0:
        print(f"WARNING: {methods_failed}/{len(cfg.odf_methods)} ODF method(s) failed.")

    print("Done.")
    return True


# ---------------------------------------------------------------------------
# CLI / Batch
# ---------------------------------------------------------------------------

def discover_subjects(dhcp_dir: Path) -> list[str]:
    """Return sorted list of 'sub-*/ses-*' relative paths under *dhcp_dir*."""
    subjects = []
    for sub_dir in sorted(dhcp_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for ses_dir in sorted(sub_dir.iterdir()):
            if ses_dir.is_dir() and ses_dir.name.startswith("ses-"):
                subjects.append(f"{sub_dir.name}/{ses_dir.name}")
    return subjects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run whole-brain tractography with multiple parameter "
        "configurations for neonatal dHCP subjects using MSMT-CSD WM FODs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
        Examples:
        %(prog)s --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp
        %(prog)s --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp sub-EXAMPLE/ses-00000
        """,
    )
    parser.add_argument(
        "subjects", nargs="*", metavar="subject/session",
        help="One or more subject/session paths (default: discover all in --dhcp-dir)",
    )
    parser.add_argument(
        "--dhcp-dir", type=Path, required=True,
        help="Processed dHCP data directory containing sub-XX/ses-YY/ structure",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, required=True,
        help="Scratch directory containing DWI inputs (reconmask.mif per session)",
    )
    parser.add_argument(
        "--nthreads", type=int, default=None, metavar="N",
        help="Number of threads (default: auto-detect)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if outputs exist",
    )
    parser.add_argument(
        "--odf-methods", nargs="+", default=None, metavar="METHOD",
        help=f"ODF methods to process (default: {' '.join(DEFAULT_ODF_METHODS)})",
    )
    parser.add_argument(
        "--act-variant", default="none", metavar="VARIANT",
        help="5tt variant for ACT (default: none = mask-only tracking). "
             "Set to e.g. 'sgm_amyg_hipp' to enable ACT.",
    )
    args = parser.parse_args(argv)

    dhcp_dir = args.dhcp_dir
    scratch_dir = args.scratch_dir

    force = args.force
    nthreads = args.nthreads if args.nthreads is not None else detect_nthreads()
    odf_methods = args.odf_methods if args.odf_methods is not None else list(DEFAULT_ODF_METHODS)

    # Validate ODF method names
    unknown = [m for m in odf_methods if m not in KNOWN_ODF_METHODS]
    if unknown:
        print(f"ERROR: Unknown ODF method(s): {' '.join(unknown)}")
        print(f"       Known methods: {' '.join(sorted(KNOWN_ODF_METHODS))}")
        return 1

    # Validate method ordering: dual_sift2w must come after combined (shared tractogram)
    if ("max_pietsch_dual_sift2w" in odf_methods
            and "max_pietsch_combined" in odf_methods
            and odf_methods.index("max_pietsch_dual_sift2w") < odf_methods.index("max_pietsch_combined")):
        print("ERROR: max_pietsch_combined must appear before max_pietsch_dual_sift2w in ODF_METHODS")
        return 1

    # Check for required MRtrix3 commands
    missing = [cmd for cmd in ["tckgen", "tcksift2", "tckinfo", "mrcalc"] if not have(cmd)]
    if missing:
        print(f"ERROR: Missing required MRtrix3 commands: {' '.join(missing)}")
        print("       Please install MRtrix3 or ensure it's in your PATH")
        return 1

    # Resolve subjects
    if args.subjects:
        subjects_to_process = args.subjects
    else:
        if not dhcp_dir.is_dir():
            print(f"ERROR: DHCP directory not found: {dhcp_dir}")
            return 1
        subjects_to_process = discover_subjects(dhcp_dir)
        if not subjects_to_process:
            print(f"ERROR: No subjects found in {dhcp_dir}")
            return 1

    act_variant = args.act_variant
    use_act = act_variant.lower() != "none"

    print(f"Tractography - Processing {len(subjects_to_process)} subject(s), {len(odf_methods)} ODF method(s)")
    if use_act:
        print(f"ACT enabled (5tt variant: {act_variant})")
    else:
        print("ACT disabled (mask-only tracking)")

    failed = 0
    for i, rel in enumerate(subjects_to_process, 1):
        ses = dhcp_dir / rel
        if not ses.is_dir():
            print(f"WARNING: Directory not found: {ses}, skipping.")
            failed += 1
            continue

        print()
        print("=" * 40)
        print(f"Processing [{i}/{len(subjects_to_process)}]: {rel}")
        print("=" * 40)

        # Derive scratch-based mask path
        scratch_subj = scratch_dir / rel
        mask_path = scratch_subj / "reconmask.mif"

        # Resolve 5tt image for ACT
        act_path = None
        if use_act:
            act_path = ses / f"5tt_{act_variant}.mif"
            if not act_path.is_file():
                print(f"WARNING: 5tt image not found: {act_path}")
                print("         Run generate_5tt.py first, or use --act-variant none")
                failed += 1
                continue

        cfg = SubjectConfig(
            subj_dir=ses,
            mask=mask_path,
            force=force,
            nthreads=nthreads,
            odf_methods=odf_methods,
            act=act_path,
        )

        if not process_subject(cfg):
            print(f"ERROR: Failed to process {rel}")
            failed += 1

    total = len(subjects_to_process)
    print()
    print("=" * 40)
    print("Batch processing complete.")
    print(f"Processed: {total - failed}/{total} subjects")
    if failed > 0:
        print(f"Failed: {failed} subjects")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

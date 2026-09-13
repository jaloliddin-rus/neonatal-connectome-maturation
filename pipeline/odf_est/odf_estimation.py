#!/usr/bin/env python3
"""
odf_estimation.py
Run ODF estimation for all methods on subjects.

Usage:
  ./odf_estimation.py --dhcp-dir /path/to/dhcp --scratch-dir /path/to/scratch
  ./odf_estimation.py --dhcp-dir ... --scratch-dir ... sub-EXAMPLE/ses-00000
  ./odf_estimation.py --dhcp-dir ... --scratch-dir ... --methods max_pietsch
  ./odf_estimation.py --dhcp-dir ... --scratch-dir ... --rmse sub-EXAMPLE/ses-00000
  ./odf_estimation.py --dhcp-dir ... --scratch-dir ... --force sub-EXAMPLE/ses-00000
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

LMAX_MS3    = "8,0,0"  # WM, GM, CSF
ALL_METHODS = ["max_pietsch", "baseline_msmt"]


# ---------- Config ----------
@dataclass
class Config:
    methods:     set[str]
    force:       bool
    nthreads:    int
    do_rmse:     bool
    dhcp_dir:    Path
    scratch_dir: Path
    tmp_dir:     Path
    rf_ay:       Path
    rf_ao:       Path
    rf_iso:      Path


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ODF estimation for all methods on subjects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subjects", nargs="*", metavar="sub/ses",
        help="Subject/session paths relative to dhcp-dir (default: all)",
    )
    parser.add_argument(
        "--dhcp-dir", type=Path, required=True,
        help="Path to processed dHCP data directory (output dirs: sub-XX/ses-YY/odf_estimation/)",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, required=True,
        help="Path to scratch directory containing DWI and mask .mif files",
    )
    parser.add_argument(
        "--tmp-dir", type=Path, default=None,
        help="Temporary directory for intermediate files (default: system temp)",
    )
    parser.add_argument(
        "--rf-dir", type=Path, default=None,
        help="Directory containing Pietsch atlas response functions "
             "(rf_tissue_32.9, rf_tissue_44.1, rf_csf_all). "
             "Default: data/response_functions/ relative to repo root.",
    )
    parser.add_argument(
        "--methods", default="",
        help=f"Comma-separated methods to run. Available: {', '.join(ALL_METHODS)} (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        default=False,
        help="Reprocess all files (override skip logic)",
    )
    parser.add_argument(
        "--nthreads", type=int,
        default=os.cpu_count() or 1,
        help="Number of MRtrix threads (default: all cores)",
    )
    parser.add_argument("--rmse", action="store_true",
                        help="Compute predicted signal and RMSE maps per method")
    return parser.parse_args()


# ---------- Helpers ----------
def run(cmd: list) -> None:
    """Run an MRtrix command, echo it, raise on failure."""
    cmd = [str(c) for c in cmd]
    print(f"   $ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def valid_mif(p: Path) -> bool:
    """Check file exists and has a parseable MRtrix header."""
    if not p.exists() or p.stat().st_size == 0:
        return False
    return subprocess.run(
        ["mrinfo", str(p), "-size", "-quiet"],
        capture_output=True,
    ).returncode == 0


def dhollander_resp(dwi: Path, mask: Path, out_dir: Path, nthreads: int, force: bool, tmp_dir: Path) -> None:
    """Estimate Dhollander response functions; skip if cached."""
    rf_wm  = out_dir / "rf_wm.txt"
    rf_gm  = out_dir / "rf_gm.txt"
    rf_csf = out_dir / "rf_csf.txt"

    if not force and nonempty(rf_wm) and nonempty(rf_gm) and nonempty(rf_csf):
        print(f"   Using cached response functions in {out_dir.name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    run([
        "dwi2response", "dhollander", dwi,
        rf_wm, rf_gm, rf_csf,
        "-voxels", out_dir / "rf_voxels.mif",
        "-mask", mask, "-nthreads", nthreads, "-scratch", tmp_dir, "-force",
    ])

    if not (nonempty(rf_wm) and nonempty(rf_gm) and nonempty(rf_csf)):
        raise RuntimeError(f"Failed to generate response functions in {out_dir}")


def compute_rmse(dwi: Path, pred: Path, mask: Path, out: Path, nthreads: int, tmp: Path) -> None:
    """Compute voxelwise RMSE map via sequential MRtrix calls (no shell pipe needed)."""
    print("   Computing RMSE map...")
    diff2 = tmp / "rmse_diff2.mif"
    mean  = tmp / "rmse_mean.mif"
    sqrt_ = tmp / "rmse_sqrt.mif"
    run(["mrcalc",  dwi, pred, "-sub", "2", "-pow", diff2, "-nthreads", nthreads, "-force"])
    run(["mrmath",  diff2, "mean", "-axis", "3", mean, "-nthreads", nthreads, "-force"])
    run(["mrcalc",  mean, "-sqrt", sqrt_,  "-nthreads", nthreads, "-force"])
    run(["mrcalc",  sqrt_, mask, "-mult", out, "-nthreads", nthreads, "-force"])


def run_msmt3(dwi: Path, out_dir: Path, lmax: str, resp_dir: Path,
              mask: Path, nthreads: int, cfg: Config, tmp: Path | None) -> None:
    """MSMT-CSD 3-tissue estimation + mtnormalise."""
    wmfod_norm  = out_dir / "wmfod_norm.mif"
    gmfod_norm  = out_dir / "gmfod_norm.mif"
    csffod_norm = out_dir / "csffod_norm.mif"

    if (not cfg.force
            and valid_mif(wmfod_norm) and valid_mif(gmfod_norm) and valid_mif(csffod_norm)
            and (not cfg.do_rmse or valid_mif(out_dir / "rmse.mif"))):
        print("   Already processed — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fod_cmd = [
        "dwi2fod", "msmt_csd", dwi,
        resp_dir / "rf_wm.txt",  out_dir / "wmfod.mif",
        resp_dir / "rf_gm.txt",  out_dir / "gmfod.mif",
        resp_dir / "rf_csf.txt", out_dir / "csffod.mif",
        "-mask", mask, "-nthreads", nthreads, "-lmax", lmax, "-force",
    ]
    if cfg.do_rmse:
        fod_cmd += ["-predicted_signal", out_dir / "pred.mif"]
    run(fod_cmd)
    run([
        "mtnormalise",
        out_dir / "wmfod.mif",  wmfod_norm,
        out_dir / "gmfod.mif",  gmfod_norm,
        out_dir / "csffod.mif", csffod_norm,
        "-mask", mask, "-nthreads", nthreads, "-force",
    ])
    if cfg.do_rmse:
        if tmp is None:
            raise RuntimeError("tmp dir required when do_rmse is True")
        compute_rmse(dwi, out_dir / "pred.mif", mask, out_dir / "rmse.mif", nthreads, tmp)

    # Clean up unnormalized intermediates once normalised outputs confirmed
    if nonempty(wmfod_norm) and nonempty(gmfod_norm) and nonempty(csffod_norm):
        for f in ["wmfod.mif", "gmfod.mif", "csffod.mif"]:
            (out_dir / f).unlink(missing_ok=True)


# ---------- Single session ----------
def process_session(ses_dir: Path, cfg: Config) -> None:
    subject = ses_dir.parent.name
    session = ses_dir.name
    print(f"\n{subject}/{session}")

    scratch = cfg.scratch_dir / subject / session
    dwi  = scratch / "postmc-dwi.mif"
    mask = scratch / "reconmask.mif"

    for label, path in [("DWI", dwi), ("Mask", mask)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    odf_base     = ses_dir / "odf_estimation"
    baseline_dir = odf_base / "baseline_msmt"
    pietsch_dir  = odf_base / "max_pietsch"

    odf_base.mkdir(parents=True, exist_ok=True)

    # Only create RMSE scratch dir when needed
    tmp_ctx = None
    tmp = None
    if cfg.do_rmse:
        tmp_ctx = tempfile.TemporaryDirectory(dir=cfg.tmp_dir, prefix="_tmp_")
        tmp = Path(tmp_ctx.name)

    try:
        # --- Max Pietsch Ay/Ao ---
        if "max_pietsch" in cfg.methods:
            print("==> Max Pietsch Ay/Ao")
            pietsch_dir.mkdir(parents=True, exist_ok=True)

            already_done = (
                not cfg.force
                and valid_mif(pietsch_dir / "wmfod_Ay_norm.mif")
                and valid_mif(pietsch_dir / "wmfod_Ao_norm.mif")
                and valid_mif(pietsch_dir / "csf_norm.mif")
                and (not cfg.do_rmse or valid_mif(pietsch_dir / "rmse.mif"))
            )
            if already_done:
                print("   Already processed — skipping")
            else:
                fod_cmd = [
                    "dwi2fod", "msmt_csd", dwi,
                    cfg.rf_ay,  pietsch_dir / "wmfod_Ay.mif",
                    cfg.rf_ao,  pietsch_dir / "wmfod_Ao.mif",
                    cfg.rf_iso, pietsch_dir / "csf.mif",
                    "-lmax", "8,8,0", "-mask", mask,
                    "-nthreads", cfg.nthreads, "-force",
                ]
                if cfg.do_rmse:
                    fod_cmd += ["-predicted_signal", pietsch_dir / "pred.mif"]
                run(fod_cmd)
                run([
                    "mtnormalise",
                    pietsch_dir / "wmfod_Ay.mif", pietsch_dir / "wmfod_Ay_norm.mif",
                    pietsch_dir / "wmfod_Ao.mif", pietsch_dir / "wmfod_Ao_norm.mif",
                    pietsch_dir / "csf.mif",      pietsch_dir / "csf_norm.mif",
                    "-mask", mask, "-nthreads", cfg.nthreads, "-force",
                ])
                if cfg.do_rmse:
                    if tmp is None:
                        raise RuntimeError("tmp dir required when do_rmse is True")
                    compute_rmse(dwi, pietsch_dir / "pred.mif", mask,
                                 pietsch_dir / "rmse.mif", cfg.nthreads, tmp)

                # Clean up unnormalized intermediates once normalised outputs confirmed
                if (nonempty(pietsch_dir / "wmfod_Ay_norm.mif")
                        and nonempty(pietsch_dir / "wmfod_Ao_norm.mif")
                        and nonempty(pietsch_dir / "csf_norm.mif")):
                    for f in ["wmfod_Ay.mif", "wmfod_Ao.mif", "csf.mif"]:
                        (pietsch_dir / f).unlink(missing_ok=True)

        # --- Baseline MSMT ---
        if "baseline_msmt" in cfg.methods:
            print("==> Baseline MSMT: all shells (3 tissues)")
            resp_dir = baseline_dir / "response"
            dhollander_resp(dwi, mask, resp_dir, cfg.nthreads, cfg.force, cfg.tmp_dir)
            run_msmt3(dwi, baseline_dir, LMAX_MS3, resp_dir, mask, cfg.nthreads, cfg, tmp)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


# ---------- Main ----------
def main() -> None:
    args = parse_args()

    # Resolve paths
    dhcp_dir = args.dhcp_dir
    scratch_dir = args.scratch_dir
    tmp_dir = args.tmp_dir if args.tmp_dir is not None else Path(tempfile.gettempdir())

    # Resolve response function directory
    if args.rf_dir is not None:
        rf_dir = args.rf_dir
    else:
        script_dir = Path(__file__).resolve().parent
        rf_dir = script_dir.parent.parent / "data" / "response_functions"

    rf_ay  = rf_dir / "rf_tissue_32.9"
    rf_ao  = rf_dir / "rf_tissue_44.1"
    rf_iso = rf_dir / "rf_csf_all"

    # Resolve method set
    if args.methods:
        methods = {m.strip() for m in args.methods.split(",") if m.strip()}
        invalid = methods - set(ALL_METHODS)
        if invalid:
            print(f"ERROR: Unknown methods: {', '.join(invalid)}", file=sys.stderr)
            print(f"  Available: {', '.join(ALL_METHODS)}", file=sys.stderr)
            sys.exit(1)
    else:
        methods = set(ALL_METHODS)

    cfg = Config(
        methods=methods,
        force=args.force,
        nthreads=args.nthreads,
        do_rmse=args.rmse,
        dhcp_dir=dhcp_dir,
        scratch_dir=scratch_dir,
        tmp_dir=tmp_dir,
        rf_ay=rf_ay,
        rf_ao=rf_ao,
        rf_iso=rf_iso,
    )

    if cfg.force:
        print("[odf] FORCE mode enabled — will reprocess all files")
    if methods != set(ALL_METHODS):
        print(f"[odf] Running selected methods: {', '.join(sorted(methods))}")

    # Build session list (discover from scratch_dir where inputs live)
    if args.subjects:
        sessions = [dhcp_dir / rel for rel in args.subjects]
    else:
        if not scratch_dir.is_dir():
            print(f"ERROR: Scratch directory not found: {scratch_dir}", file=sys.stderr)
            sys.exit(1)
        scratch_sessions = sorted(p for p in scratch_dir.glob("sub-*/ses-*") if p.is_dir())
        if not scratch_sessions:
            print(f"ERROR: No subjects found in {scratch_dir}", file=sys.stderr)
            sys.exit(1)
        # Map to dhcp_dir for output paths
        sessions = [dhcp_dir / p.relative_to(scratch_dir) for p in scratch_sessions]

    # Validate atlas response files once upfront
    if "max_pietsch" in methods:
        for label, path in [("RF_AY", rf_ay), ("RF_AO", rf_ao), ("RF_ISO", rf_iso)]:
            if not path.exists():
                print(f"ERROR: Atlas response file not found ({label}): {path}", file=sys.stderr)
                sys.exit(1)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Validate MRtrix3 tools are on PATH
    tools = ["mrinfo", "dwi2fod", "mtnormalise", "mrcalc", "mrmath"]
    if "baseline_msmt" in methods:
        tools.append("dwi2response")
    missing = [t for t in tools if not shutil.which(t)]
    if missing:
        print(f"ERROR: MRtrix3 tools not found on PATH: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"ODF Estimation — Processing {len(sessions)} session(s)")

    failed = 0
    for i, ses_dir in enumerate(sessions, 1):
        print(f"\n{'=' * 40}")
        print(f"Processing [{i}/{len(sessions)}]: {ses_dir.parent.name}/{ses_dir.name}")
        print("=" * 40)
        try:
            process_session(ses_dir, cfg)
        except Exception as e:
            print(f"ERROR: {ses_dir.parent.name}/{ses_dir.name}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print("Batch processing complete.")
    print(f"Processed: {len(sessions) - failed}/{len(sessions)} sessions")
    if failed:
        print(f"Failed: {failed} sessions")
        sys.exit(1)


if __name__ == "__main__":
    main()

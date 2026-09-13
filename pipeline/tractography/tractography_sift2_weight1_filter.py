#!/usr/bin/env python3
"""tractography_sift2_weight1_filter.py

Remove SIFT2 weight=1 streamlines, re-run SIFT2, and regenerate connectomes.

Pipeline (for each of Ao and Ay independently):
  1. Identify streamlines where SIFT2 weight == 1 and remove them
     from the tractogram via tckedit → filtered .tck
  2. Re-run tcksift2 on the filtered .tck with the corresponding FOD
     → fresh SIFT2 weights
  3. Generate connectomes from the filtered tractograms with the new weights
  4. Log-transform connectomes (log(x+1) element-wise)
  5. Convert parcellation nodes to surface mesh (label2mesh)

Rationale:
  SIFT2 assigns weight=1 to streamlines it could not meaningfully reweight
  (uninformative / poorly constrained).  Removing them and re-running SIFT2
  allows the algorithm to redistribute weights across genuinely informative
  streamlines, producing cleaner connectomes.

Pre-requisites (must already exist from prior pipeline steps):
  - tractography.py          → tractogram + dual SIFT2 weights (y & o)
  - odf_estimation.py        → wmfod_Ao_norm.mif, wmfod_Ay_norm.mif
  - connectome_generation.py → GM nodes image, LUT, before-filtering connectomes

Outputs are split by pipeline stage:
  Tractogram artifacts → <session>/tractography/max_pietsch_dual_sift2w_w1filter/
  Connectome outputs   → <session>/connectome_gen/

Usage:
    python tractography_sift2_weight1_filter.py --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp
    python tractography_sift2_weight1_filter.py --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp sub-EXAMPLE/ses-00000
    python tractography_sift2_weight1_filter.py --force --nthreads 16 --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp sub-EXAMPLE/ses-00000
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


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


def run_capture(cmd: list[str]) -> str:
    """Run *cmd* and return stripped stdout, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def read_file_value(path: Path) -> str:
    """Read and return the stripped contents of a single-value file, or 'NA'."""
    try:
        return path.read_text().strip() or "NA"
    except OSError:
        return "NA"


def detect_nthreads() -> int:
    """Detect number of hardware threads available."""
    return min(os.cpu_count() or 1, 8)


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


def normalize_txt(src: Path, dst: Path) -> None:
    """Normalise an MRtrix3 text file to one numeric value per line.

    MRtrix3 tools may write text files with comment/header lines starting
    with '#' and all values space-separated on a single line.  This helper
    strips comments/empty lines and splits space-separated values so that
    each numeric value occupies its own line.
    """
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.split():
                fout.write(token + "\n")


def get_tck_count(tck: Path) -> int | None:
    """Return streamline count from *tck* via tckinfo, or None on failure."""
    output = run_capture(["tckinfo", str(tck)])
    if not output:
        return None
    for line in output.splitlines():
        m = re.match(r"\s*count:\s+(\d+)", line)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINWEIGHT_THRESHOLD = 0.000001

PARCELLATION_DEFAULT = "cortical"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SubjectConfig:
    subj_dir: Path
    force: bool
    nthreads: int
    mask: Path | None = None
    act: Path | None = None
    parcellation: str = "cortical"

    # Derived
    subject_id: str = ""
    session_id: str = ""
    tract_outdir: Path = field(default_factory=lambda: Path())
    conn_outdir: Path = field(default_factory=lambda: Path())

    def __post_init__(self):
        self.subject_id = self.subj_dir.parent.name
        self.session_id = self.subj_dir.name
        self.tract_outdir = (
            self.subj_dir / "tractography" / "max_pietsch_dual_sift2w_w1filter"
        )
        self.conn_outdir = self.subj_dir / "connectome_gen"


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def filter_weight1(
    cfg: SubjectConfig,
    label: str,
    w_norm: Path,
    tck_in: Path,
) -> tuple[int, int]:
    """Remove streamlines where SIFT2 weight == 1.

    Creates a zeroed-weights file (weight → 0 for w==1, original otherwise),
    then uses ``tckedit -minweight`` to discard those streamlines.

    Returns ``(n_keep, n_remove)``.
    """
    idx_keep = cfg.tract_outdir / f"streamline_indices_{label}_keep.txt"
    idx_remove = cfg.tract_outdir / f"streamline_indices_{label}_remove.txt"
    tck_out = cfg.tract_outdir / f"tractogram_{label}_filtered.tck"

    need_filter = (
        cfg.force
        or not tck_out.is_file()
        or not idx_keep.is_file()
        or not idx_remove.is_file()
    )

    if need_filter:
        print(f"  [filter-{label}] Removing streamlines with {label} SIFT2 weight == 1...")

        # Read normalised weights and build zeroed-weights + index files
        weights = []
        with open(w_norm) as f:
            for line in f:
                weights.append(line.strip())

        tmp_zeroed = cfg.tract_outdir / f".tmp_weights_{label}_zeroed.txt"
        try:
            with (
                open(tmp_zeroed, "w") as fz,
                open(idx_keep, "w") as fk,
                open(idx_remove, "w") as fr,
            ):
                for i, w in enumerate(weights):
                    if float(w) == 1.0:
                        fz.write("0\n")
                        fr.write(f"{i}\n")
                    else:
                        fz.write(w + "\n")
                        fk.write(f"{i}\n")

            run([
                "tckedit", str(tck_in), str(tck_out),
                "-tck_weights_in", str(tmp_zeroed),
                "-minweight", str(MINWEIGHT_THRESHOLD),
                "-nthreads", str(cfg.nthreads),
                "-force",
            ])
        except (subprocess.CalledProcessError, OSError):
            tck_out.unlink(missing_ok=True)
            raise
        finally:
            tmp_zeroed.unlink(missing_ok=True)

        # Validate counts
        n_keep = sum(1 for w in weights if float(w) != 1.0)
        n_remove = len(weights) - n_keep
        n_tck = get_tck_count(tck_out)
        if n_tck is None:
            raise RuntimeError(
                f"Failed to read streamline count from {tck_out}"
            )
        if n_keep != n_tck:
            raise RuntimeError(
                f"{label} count mismatch {cfg.subject_id}/{cfg.session_id}: "
                f"index={n_keep} tck={n_tck}"
            )
        print(
            f"  [filter-{label}] Removed {n_remove} / {len(weights)} "
            f"streamlines (kept {n_keep})"
        )
    else:
        print(
            f"  [filter-{label}] Filtered tractogram already exists "
            f"(--force to recompute)"
        )
        n_remove = sum(1 for _ in open(idx_remove))
        n_keep = sum(1 for _ in open(idx_keep))
        print(
            f"  [filter-{label}] {n_remove} removed, {n_keep} kept"
        )

    return n_keep, n_remove


def run_sift2(
    cfg: SubjectConfig,
    label: str,
    tck: Path,
    fod: Path,
) -> Path:
    """Re-run tcksift2 on a filtered tractogram.

    Returns the path to the normalised new weights file.
    """
    new_weights = cfg.tract_outdir / f"sift2_weights_{label}_new.txt"
    new_mu = cfg.tract_outdir / f"sift2_mu_{label}_new.txt"
    new_weights_norm = cfg.tract_outdir / f"sift2_weights_{label}_new_norm.txt"

    need_sift2 = (
        cfg.force or not new_weights.is_file()
    )

    if need_sift2:
        print(f"  [sift2-{label}] Re-running SIFT2 on {label}-filtered tractogram...")

        cmd = [
            "tcksift2", str(tck), str(fod), str(new_weights),
            "-out_mu", str(new_mu),
        ]
        if cfg.act is not None:
            cmd += ["-act", str(cfg.act)]
        elif cfg.mask is not None:
            cmd += ["-proc_mask", str(cfg.mask)]
        cmd += ["-nthreads", str(cfg.nthreads), "-force"]

        t0 = time.monotonic()
        try:
            run(cmd)
        except (subprocess.CalledProcessError, OSError):
            new_weights.unlink(missing_ok=True)
            new_mu.unlink(missing_ok=True)
            raise

        elapsed = time.monotonic() - t0
        print(f"  [sift2-{label}] Done in {elapsed:.0f}s")
        print(f"  [sift2-{label}] mu = {read_file_value(new_mu)}")
    else:
        print(
            f"  [sift2-{label}] New weights already exist "
            f"(--force to recompute)"
        )
        if new_mu.is_file():
            print(f"  [sift2-{label}] mu = {read_file_value(new_mu)}")

    # Normalise new weights (strip MRtrix headers)
    normalize_txt(new_weights, new_weights_norm)

    return new_weights_norm


def run_connectome(
    cfg: SubjectConfig,
    label: str,
    tck: Path,
    weights: Path,
    nodes: Path,
    csv_out: Path,
    asg_out: Path,
) -> None:
    """Generate a connectome from a filtered tractogram with new SIFT2 weights."""
    if cfg.force or not csv_out.is_file():
        print(f"  [connectome] Generating {label} connectome...")
        try:
            run([
                "tck2connectome", str(tck), str(nodes), str(csv_out),
                "-tck_weights_in", str(weights),
                "-symmetric", "-zero_diagonal",
                "-stat_edge", "sum",
                "-assignment_radial_search", "4",
                "-out_assignments", str(asg_out),
                "-nthreads", str(cfg.nthreads),
                "-force",
            ])
        except (subprocess.CalledProcessError, OSError):
            csv_out.unlink(missing_ok=True)
            asg_out.unlink(missing_ok=True)
            raise
        print(f"    → {csv_out}")
    else:
        print(f"  [connectome] Reusing {label} connectome: {csv_out}")


def log_transform_csv(
    cfg: SubjectConfig,
    label: str,
    src: Path,
    dst: Path,
) -> None:
    """Apply log(x+1) element-wise to a CSV connectome matrix."""
    if cfg.force or not dst.is_file():
        print(f"  [log-transform] Applying log(x+1) to {label} connectome...")
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                vals = line.strip().split(",")
                transformed = [str(math.log1p(float(v))) for v in vals]
                fout.write(",".join(transformed) + "\n")
        print(f"    → {dst}")
    else:
        print(
            f"  [log-transform] {label} log connectome already exists "
            f"(--force to recompute)"
        )


def apply_mu_scaling(
    csv_in: Path, mu_path: Path, csv_out: Path, force: bool = False,
) -> None:
    """Scale a connectome CSV by the SIFT2 proportionality coefficient mu."""
    if not force and csv_out.is_file():
        print(f"  Reusing mu-scaled connectome: {csv_out}")
        return
    mu = float(mu_path.read_text().strip())
    print(f"  Applying mu scaling ({mu:.6e}) to {csv_in.name} -> {csv_out.name}")
    with open(csv_in) as fin, open(csv_out, "w") as fout:
        for line in fin:
            vals = line.strip().split(",")
            scaled = [str(float(v) * mu) for v in vals]
            fout.write(",".join(scaled) + "\n")


def generate_node_mesh(cfg: SubjectConfig, nodes_gm: Path) -> None:
    """Convert parcellation nodes to a surface mesh (OBJ)."""
    mesh_dir = cfg.tract_outdir / "node_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    nodes_mesh = mesh_dir / "nodes_GM_mesh.obj"

    if cfg.force or not nodes_mesh.is_file():
        print("  [meshes] Converting parcellation nodes to surface meshes...")
        run(["label2mesh", str(nodes_gm), str(nodes_mesh), "-force"])
        print(f"    → {nodes_mesh}")
    else:
        print("  [meshes] Node mesh already exists (--force to recompute)")


# ---------------------------------------------------------------------------
# Subject orchestrator
# ---------------------------------------------------------------------------

def process_subject(cfg: SubjectConfig) -> bool:
    """Run the full SIFT2 w=1 filtering pipeline for one subject."""
    subj = cfg.subj_dir
    if not subj.is_dir():
        print(f"WARNING: Missing subject dir: {subj}", file=sys.stderr)
        return False

    print()
    print(
        f"[{cfg.subject_id}/{cfg.session_id}] "
        f"Starting SIFT2 weight=1 filtering + re-SIFT2 analysis (Ao + Ay)"
    )

    # ---- Input paths ----
    sift2w_dir = subj / "tractography" / "max_pietsch_dual_sift2w"
    odf_old = subj / "odf_estimation" / "max_pietsch" / "wmfod_Ao_norm.mif"
    odf_young = subj / "odf_estimation" / "max_pietsch" / "wmfod_Ay_norm.mif"
    cg_dir = subj / "connectome_gen"

    tck = sift2w_dir / "tractogram.tck"
    if not tck.is_file():
        tck = subj / "tractography" / "max_pietsch_combined" / "tractogram.tck"
    w_old = sift2w_dir / "sift2_weights_o.txt"
    w_young = sift2w_dir / "sift2_weights_y.txt"

    # Before-filtering connectomes (from connectome_generation.py)
    p = cfg.parcellation
    young_before_csv = cg_dir / f"connectome_{p}_young.csv"
    old_before_csv = cg_dir / f"connectome_{p}_old.csv"

    # GM nodes image, LUT, mean b0 (from connectome_generation.py)
    nodes_gm = cg_dir / f"nodes_GM_{p}.mif"
    lut_gm = cg_dir / f"nodes_GM_{p}_lut.txt"
    mean_b0 = cg_dir / "mean_b0.mif"

    # ---- Validate pre-requisites ----
    missing = False
    for label, f in [
        ("tractography", tck),
        ("tractography", w_old),
        ("tractography", w_young),
        ("connectome_gen", young_before_csv),
        ("connectome_gen", old_before_csv),
        ("connectome_gen", nodes_gm),
        ("connectome_gen", lut_gm),
        ("connectome_gen", mean_b0),
        ("odf", odf_old),
        ("odf", odf_young),
    ]:
        if not f.is_file():
            print(f"  MISSING ({label}): {f}", file=sys.stderr)
            missing = True

    if missing:
        print(
            "  ERROR: Pre-requisite files missing. Run odf_estimation.py, "
            "tractography.py and connectome_generation.py first.",
            file=sys.stderr,
        )
        return False

    # ---- ACT 5tt image ----
    if cfg.act is not None:
        print(f"  ACT: {cfg.act}")

    # ---- Output directories ----
    cfg.tract_outdir.mkdir(parents=True, exist_ok=True)
    cfg.conn_outdir.mkdir(parents=True, exist_ok=True)

    # ---- Normalise original SIFT2 weight files ----
    w_old_norm = cfg.tract_outdir / "sift2_weights_o_orig_norm.txt"
    w_young_norm = cfg.tract_outdir / "sift2_weights_y_orig_norm.txt"

    if cfg.force or not w_old_norm.is_file():
        normalize_txt(w_old, w_old_norm)
    if cfg.force or not w_young_norm.is_file():
        normalize_txt(w_young, w_young_norm)

    # ---- Count total streamlines ----
    n_total = sum(1 for _ in open(w_old_norm))

    # ==== Step 1a: Filter old (Ao) — remove weight == 1 streamlines ====
    ao_n_keep, ao_n_remove = filter_weight1(
        cfg, "Ao", w_old_norm, tck,
    )

    # ==== Step 1b: Filter young (Ay) — remove weight == 1 streamlines ====
    ay_n_keep, ay_n_remove = filter_weight1(
        cfg, "Ay", w_young_norm, tck,
    )

    # ==== Step 2a: Re-run SIFT2 on Ao-filtered tractogram ====
    ao_tck_out = cfg.tract_outdir / "tractogram_Ao_filtered.tck"
    ao_new_weights_norm = run_sift2(cfg, "o", ao_tck_out, odf_old)

    # ==== Step 2b: Re-run SIFT2 on Ay-filtered tractogram ====
    ay_tck_out = cfg.tract_outdir / "tractogram_Ay_filtered.tck"
    ay_new_weights_norm = run_sift2(cfg, "y", ay_tck_out, odf_young)

    # ==== Step 3: Connectomes — filtered tractogram + new SIFT2 weights ====
    young_after_csv = cfg.conn_outdir / f"connectome_{p}_young_w1filtered.csv"
    old_after_csv = cfg.conn_outdir / f"connectome_{p}_old_w1filtered.csv"
    young_after_asg = cfg.conn_outdir / f"assignments_{p}_young_w1filtered.txt"
    old_after_asg = cfg.conn_outdir / f"assignments_{p}_old_w1filtered.txt"

    run_connectome(
        cfg, "young (Ay-filtered + new SIFT2)",
        ay_tck_out, ay_new_weights_norm, nodes_gm,
        young_after_csv, young_after_asg,
    )
    run_connectome(
        cfg, "old (Ao-filtered + new SIFT2)",
        ao_tck_out, ao_new_weights_norm, nodes_gm,
        old_after_csv, old_after_asg,
    )

    # ==== Step 3b: Mu-scale filtered connectomes ====
    ao_new_mu = cfg.tract_outdir / "sift2_mu_o_new.txt"
    ay_new_mu = cfg.tract_outdir / "sift2_mu_y_new.txt"

    young_mu_csv = cfg.conn_outdir / f"connectome_{p}_young_w1filtered_mu.csv"
    old_mu_csv = cfg.conn_outdir / f"connectome_{p}_old_w1filtered_mu.csv"

    apply_mu_scaling(young_after_csv, ay_new_mu, young_mu_csv, cfg.force)
    apply_mu_scaling(old_after_csv, ao_new_mu, old_mu_csv, cfg.force)

    # ==== Step 4: Log-transform connectomes ====
    ao_log_csv = cfg.conn_outdir / f"connectome_{p}_old_w1filtered_log.csv"
    ay_log_csv = cfg.conn_outdir / f"connectome_{p}_young_w1filtered_log.csv"

    log_transform_csv(cfg, "Ao", old_after_csv, ao_log_csv)
    log_transform_csv(cfg, "Ay", young_after_csv, ay_log_csv)

    # ==== Step 5: Node surface meshes ====
    generate_node_mesh(cfg, nodes_gm)

    # ==== mrview QC commands ====
    qc_file = cfg.tract_outdir / "mrview_qc_commands.txt"
    now = datetime.now(timezone.utc).isoformat()
    qc_file.write_text(
        f"# mrview QC commands for SIFT2 w=1 filtering\n"
        f"# Subject: {cfg.subject_id}  Session: {cfg.session_id}\n"
        f"# Generated: {now}\n"
        f"\n"
        f"# Ao (old, raw weights):\n"
        f"mrview {mean_b0} \\\n"
        f"  -connectome.init {nodes_gm} \\\n"
        f"  -connectome.load {old_after_csv} &\n"
        f"\n"
        f"# Ao (old, log-transformed weights):\n"
        f"mrview {mean_b0} \\\n"
        f"  -connectome.init {nodes_gm} \\\n"
        f"  -connectome.load {ao_log_csv} &\n"
        f"\n"
        f"# Ay (young, raw weights):\n"
        f"mrview {mean_b0} \\\n"
        f"  -connectome.init {nodes_gm} \\\n"
        f"  -connectome.load {young_after_csv} &\n"
        f"\n"
        f"# Ay (young, log-transformed weights):\n"
        f"mrview {mean_b0} \\\n"
        f"  -connectome.init {nodes_gm} \\\n"
        f"  -connectome.load {ay_log_csv} &\n"
    )
    print(f"  QC commands saved to: {qc_file}")

    # ---- Metadata ----
    info_file = cfg.tract_outdir / "filter_info.txt"
    info_file.write_text(
        f"# SIFT2 weight=1 filtering + re-SIFT2 metadata\n"
        f"pipeline=filter_w1 -> tcksift2 -> tck2connectome\n"
        f"filter_criterion=weight==1\n"
        f"subject={cfg.subject_id}\n"
        f"session={cfg.session_id}\n"
        f"total_streamlines={n_total}\n"
        f"removed_old_Ao={ao_n_remove}\n"
        f"removed_young_Ay={ay_n_remove}\n"
        f"kept_old_Ao={ao_n_keep}\n"
        f"kept_young_Ay={ay_n_keep}\n"
        f"sift2_mu_old_new={read_file_value(ao_new_mu)}\n"
        f"sift2_mu_young_new={read_file_value(ay_new_mu)}\n"
        f"log_connectome_old=connectome_gen/{ao_log_csv.name}\n"
        f"log_connectome_young=connectome_gen/{ay_log_csv.name}\n"
        f"mu_connectome_old=connectome_gen/{old_mu_csv.name}\n"
        f"mu_connectome_young=connectome_gen/{young_mu_csv.name}\n"
        f"node_meshes=node_meshes/nodes_GM_mesh.obj\n"
        f"date={now}\n"
    )

    # ---- Summary ----
    ao_new_weights = cfg.tract_outdir / "sift2_weights_o_new.txt"
    ay_new_weights = cfg.tract_outdir / "sift2_weights_y_new.txt"

    print()
    print(f"  [{cfg.subject_id}/{cfg.session_id}] Done. Outputs in:")
    print(f"    Tractogram dir : {cfg.tract_outdir}/")
    print(f"      - {ao_tck_out.name}")
    print(f"      - {ay_tck_out.name}")
    print(f"      - {ao_new_weights.name}")
    print(f"      - {ay_new_weights.name}")
    print(f"    Connectome dir : {cfg.conn_outdir}/")
    print(f"      - {young_after_csv.name}")
    print(f"      - {old_after_csv.name}")
    print(f"      - {young_mu_csv.name}")
    print(f"      - {old_mu_csv.name}")
    print(f"      - {ao_log_csv.name}")
    print(f"      - {ay_log_csv.name}")
    print(f"    Node meshes    : {cfg.tract_outdir / 'node_meshes' / 'nodes_GM_mesh.obj'}")
    print(
        f"    - Ao: removed {ao_n_remove} / {n_total} "
        f"(mu={read_file_value(ao_new_mu)})"
    )
    print(
        f"    - Ay: removed {ay_n_remove} / {n_total} "
        f"(mu={read_file_value(ay_new_mu)})"
    )

    return True


# ---------------------------------------------------------------------------
# CLI / Batch
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove SIFT2 weight=1 streamlines, re-run SIFT2, "
        "and regenerate connectomes (Ao & Ay independently).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp
  %(prog)s --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp sub-EXAMPLE/ses-00000
  %(prog)s --force --nthreads 16 --dhcp-dir /data/dhcp --scratch-dir /scratch/dhcp sub-EXAMPLE/ses-00000
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
        help="Recompute all outputs even if present",
    )
    parser.add_argument(
        "--act-variant", default="none", metavar="VARIANT",
        help="5tt variant for ACT (default: none = no ACT). "
             "Set to e.g. 'sgm_amyg_hipp' to enable ACT.",
    )
    parser.add_argument(
        "--parcellation", default=PARCELLATION_DEFAULT,
        help=f"Parcellation scheme name (default: {PARCELLATION_DEFAULT})",
    )
    args = parser.parse_args(argv)

    dhcp_dir = args.dhcp_dir
    scratch_dir = args.scratch_dir

    force = args.force
    nthreads = args.nthreads if args.nthreads is not None else detect_nthreads()
    act_variant = args.act_variant
    use_act = act_variant.lower() != "none"

    # Check for required commands
    required_cmds = ["tckedit", "tcksift2", "tckinfo", "tck2connectome", "label2mesh"]
    missing_cmds = [cmd for cmd in required_cmds if not have(cmd)]
    if missing_cmds:
        print(f"ERROR: Missing required commands: {' '.join(missing_cmds)}")
        print("       Please install MRtrix3 or ensure it's in your PATH")
        return 1

    if not dhcp_dir.is_dir():
        print(f"ERROR: DHCP directory not found: {dhcp_dir}")
        return 1

    # Resolve subjects
    if args.subjects:
        subjects_to_process = args.subjects
    else:
        subjects_to_process = discover_subjects(dhcp_dir)
        if not subjects_to_process:
            print(f"ERROR: No subject sessions found under {dhcp_dir}")
            return 1

    print("=" * 60)
    print("SIFT2 Weight=1 Filter + Re-SIFT2 + Connectome (Ao & Ay)")
    print("=" * 60)
    print(f"Subjects     : {len(subjects_to_process)}")
    print(f"Threads      : {nthreads}")
    print(f"Force        : {force}")
    print(f"Parcellation : {args.parcellation}")


    failed = 0
    for i, rel in enumerate(subjects_to_process, 1):
        ses = dhcp_dir / rel
        if not ses.is_dir():
            print(f"WARNING: Directory not found: {ses}, skipping.")
            failed += 1
            continue

        print()
        print("=" * 42)
        print(f"[{i}/{len(subjects_to_process)}] {rel}")
        print("=" * 42)

        # Derive scratch-based mask path
        scratch_subj = scratch_dir / rel
        mask_path = scratch_subj / "reconmask.mif"

        # Resolve ACT 5tt image
        act_path = None
        if use_act:
            act_path = ses / f"5tt_{act_variant}.mif"
            if not act_path.is_file():
                print(f"WARNING: 5tt image not found: {act_path}")
                print("         Running SIFT2 without ACT")
                act_path = None

        cfg = SubjectConfig(
            subj_dir=ses,
            force=force,
            nthreads=nthreads,
            mask=mask_path,
            act=act_path,
            parcellation=args.parcellation,
        )

        try:
            if not process_subject(cfg):
                print(f"FAILED: {rel}", file=sys.stderr)
                failed += 1
        except Exception as exc:
            print(f"ERROR: {rel}: {exc}", file=sys.stderr)
            failed += 1

    print()
    print("=" * 42)
    total = len(subjects_to_process)
    print(f"Complete. Processed {total - failed}/{total} subjects.")
    if failed > 0:
        print(f"Failed: {failed} subjects")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

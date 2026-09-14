#!/usr/bin/env python3
"""connectome_generation.py

Build structural connectomes from dHCP neonatal data using the per-subject
Draw-EM 87-label parcellation (from the dHCP anat pipeline).

Pipeline:
  1. Transform drawem87.mif from T2w space to DWI space using the dHCP
     T2w-to-DWI FSL affine (nearest-neighbour interpolation for labels)
  2. Filter to GM-only labels (Hippocampus + regions ending with ' GM')
  3. Remap to sequential node IDs for connectome generation
  4. Run tck2connectome for combined / young / old / baseline_msmt
  5. Generate QC images and visualization commands

Label source: desc-drawem87_dseg.tsv from the dHCP anat pipeline release
  (Makropoulos et al. 2014, extending Gousias et al. 2012 -- same labels
  as the Alena Uus atlas but computed per-subject via Draw-EM)

Usage:
    python connectome_generation.py --dhcp-dir /path/to/dhcp --scratch-dir /path/to/scratch \\
        --dwi-source /path/to/shard_pipeline --anat-source /path/to/anat_pipeline
    python connectome_generation.py --dhcp-dir /data/dhcp --scratch-dir /data/scratch \\
        --dwi-source /data/shard --anat-source /data/anat \\
        -n 16 sub-EXAMPLE/ses-00000
    python connectome_generation.py --dhcp-dir /data/dhcp --scratch-dir /data/scratch \\
        --dwi-source /data/shard --anat-source /data/anat \\
        --parcellation full_split_stn
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


class PipelineError(RuntimeError):
    """Raised when a processing step fails for a single subject."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_nthreads() -> int:
    """Detect number of hardware threads available."""
    return min(os.cpu_count() or 1, 8)


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


def pick_existing_file(label: str, candidates: list[Path]) -> Path:
    """Return the first candidate that exists, or raise PipelineError."""
    for c in candidates:
        if c.is_file():
            return c
    checked = "\n".join(f"  - {c}" for c in candidates)
    raise PipelineError(f"Could not find {label}. Checked:\n{checked}")


# ---------------------------------------------------------------------------
# GM label parsing
# ---------------------------------------------------------------------------

@dataclass
class GMLabel:
    label_id: int
    name: str


# ---------------------------------------------------------------------------
# Parcellation schemes
# ---------------------------------------------------------------------------

PARCELLATION_SCHEMES: dict[str, dict] = {
    "cortical": {
        "description": "Hippocampus + cortical GM (34 nodes)",
        "extra_ids": set(),
        "label_merges": {},
        "label_renames": {},
    },
    "full_merged_stn": {
        "description": "All subcortical + cerebellum + brainstem, merged thalamus, with STN (47 nodes)",
        "extra_ids": {3, 4, 17, 18, 19, 40, 41, 42, 43, 44, 45, 46, 47},
        "label_merges": {86: 42, 87: 43},
        "label_renames": {42: "Thalamus right", 43: "Thalamus left"},
    },
    "full_split_stn": {
        "description": "All subcortical + cerebellum + brainstem, split thalamus, with STN (49 nodes)",
        "extra_ids": {3, 4, 17, 18, 19, 40, 41, 42, 43, 44, 45, 46, 47, 86, 87},
        "label_merges": {},
        "label_renames": {
            42: "Thalamus right (high T2)",
            43: "Thalamus left (high T2)",
            86: "Thalamus right (low T2)",
            87: "Thalamus left (low T2)",
        },
    },
    "full_merged": {
        "description": "All subcortical + cerebellum + brainstem, merged thalamus, no STN (45 nodes)",
        "extra_ids": {3, 4, 17, 18, 19, 40, 41, 42, 43, 46, 47},
        "label_merges": {86: 42, 87: 43},
        "label_renames": {42: "Thalamus right", 43: "Thalamus left"},
    },
    "full_split": {
        "description": "All subcortical + cerebellum + brainstem, split thalamus, no STN (47 nodes)",
        "extra_ids": {3, 4, 17, 18, 19, 40, 41, 42, 43, 46, 47, 86, 87},
        "label_merges": {},
        "label_renames": {
            42: "Thalamus right (high T2)",
            43: "Thalamus left (high T2)",
            86: "Thalamus right (low T2)",
            87: "Thalamus left (low T2)",
        },
    },
}


def select_gm_labels(
    all_labels: list[tuple[int, str]],
    scheme: dict,
) -> list[GMLabel]:
    """Select GM labels for connectome nodes based on parcellation scheme."""
    extra_ids = scheme["extra_ids"]
    merged_away = set(scheme["label_merges"].keys())
    renames = scheme.get("label_renames", {})

    gm_labels: list[GMLabel] = []
    for lid, name in all_labels:
        if lid in merged_away:
            continue
        is_cortical_gm = name.startswith("Hippocampus") or name.endswith(" GM")
        is_extra = lid in extra_ids
        if is_cortical_gm or is_extra:
            gm_labels.append(GMLabel(lid, renames.get(lid, name)))

    return gm_labels


def parse_drawem87_tsv(tsv_path: Path) -> tuple[list[tuple[int, str]], list[GMLabel]]:
    """Parse the Draw-EM 87-label TSV.

    Returns:
        all_labels: list of (label_id, name) for every row
        gm_labels:  list of GMLabel for GM-only rows (Hippocampus + ' GM')
    """
    if not tsv_path.is_file():
        raise PipelineError(f"Draw-EM 87-label TSV not found: {tsv_path}")

    all_labels: list[tuple[int, str]] = []
    gm_labels: list[GMLabel] = []

    with open(tsv_path) as f:
        header = next(f, None)
        if header is None:
            raise PipelineError(f"Draw-EM 87-label TSV is empty: {tsv_path}")
        cols = header.strip().split("\t")
        if len(cols) < 2:
            raise PipelineError(
                f"Draw-EM 87-label TSV has unexpected format "
                f"(expected tab-separated with >=2 columns): {tsv_path}"
            )

        for lineno, line in enumerate(f, 2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            label_id_str = parts[0].strip()
            roi_name = parts[1].strip()
            if not label_id_str.isdigit():
                raise PipelineError(
                    f"Non-integer label ID '{label_id_str}' "
                    f"at line {lineno} of {tsv_path}"
                )
            lid = int(label_id_str)
            all_labels.append((lid, roi_name))
            if roi_name.startswith("Hippocampus") or roi_name.endswith(" GM"):
                gm_labels.append(GMLabel(lid, roi_name))

    if not gm_labels:
        raise PipelineError(f"No GM labels found in TSV: {tsv_path}")

    return all_labels, gm_labels


def write_lut_in(
    all_labels: list[tuple[int, str]],
    path: Path,
    renames: dict[int, str] | None = None,
    skip_ids: set[int] | None = None,
) -> None:
    """Write the full LUT for labelconvert input."""
    renames = renames or {}
    skip_ids = skip_ids or set()
    with open(path, "w") as f:
        for lid, name in all_labels:
            if lid in skip_ids:
                continue
            safe_name = renames.get(lid, name).replace(" ", "_")
            f.write(f"{lid} {safe_name}\n")


def write_lut_out(gm_labels: list[GMLabel], path: Path) -> None:
    """Write the target LUT for labelconvert (sequential IDs -> GM names)."""
    with open(path, "w") as f:
        for i, gm in enumerate(gm_labels, 1):
            f.write(f"{i} {gm.name.replace(' ', '_')}\n")


def write_lut_gm(gm_labels: list[GMLabel], path: Path) -> None:
    """Write a human-readable LUT mapping new -> original IDs."""
    with open(path, "w") as f:
        f.write("# new_node_id  original_label_id  ROI_name\n")
        for i, gm in enumerate(gm_labels, 1):
            f.write(f"{i:3d}  {gm.label_id:3d}  {gm.name}\n")


# ---------------------------------------------------------------------------
# Label merging
# ---------------------------------------------------------------------------

def merge_labels_in_image(
    cfg: "SubjectConfig",
    input_image: Path,
    label_merges: dict[int, int],
) -> Path:
    """Replace label IDs in *input_image* according to *label_merges*."""
    p = cfg.parcellation
    output = cfg.outdir / f"nodes_87_merged_{p}.mif"
    if cfg.force or not output.is_file():
        merge_desc = ", ".join(f"{s}->{t}" for s, t in label_merges.items())
        print(f"==> Merging labels in 87-label image: {merge_desc}")
        current = input_image
        temps: list[Path] = []
        try:
            for i, (src, tgt) in enumerate(label_merges.items()):
                is_last = i == len(label_merges) - 1
                out = output if is_last else cfg.outdir / f"_merge_tmp_{i}.mif"
                if not is_last:
                    temps.append(out)
                run([
                    "mrcalc", str(current), str(src), "-eq",
                    str(tgt), str(current), "-if",
                    "-datatype", "uint16",
                    "-nthreads", str(cfg.nthreads),
                    str(out), "-force",
                ])
                current = out
        except (subprocess.CalledProcessError, OSError):
            for t in temps:
                t.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            raise
        for t in temps:
            t.unlink(missing_ok=True)
    else:
        print(f"==> Reusing merged labels image: {output}")
    return output


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubjectConfig:
    subj_dir: Path
    mask: Path
    dwi: Path
    drawem87: Path
    scratch_subj: Path
    force: bool
    force_qc: bool
    nthreads: int
    parcellation: str = "cortical"
    # Derived
    subject_id: str = ""
    session_id: str = ""
    outdir: Path = field(default_factory=lambda: Path())

    def __post_init__(self):
        self.subject_id = self.subj_dir.parent.name
        self.session_id = self.subj_dir.name
        self.outdir = self.subj_dir / "connectome_gen"


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def compute_mean_b0(cfg: SubjectConfig) -> Path:
    """Compute or reuse the mean b=0 image."""
    mean_b0 = cfg.outdir / "mean_b0.mif"
    if cfg.force or not mean_b0.is_file():
        print(f"==> Computing DWI mean b=0 image from: {cfg.dwi}")
        mean_b0_tmp = cfg.outdir / "mean_b0_tmp.mif"
        print(f"   $ dwiextract {cfg.dwi} -bzero - | mrmath - mean -axis 3 {mean_b0_tmp} -force")
        p1 = subprocess.Popen(
            ["dwiextract", str(cfg.dwi), "-bzero", "-"],
            stdout=subprocess.PIPE,
        )
        try:
            p2 = subprocess.Popen(
                ["mrmath", "-", "mean", "-axis", "3", str(mean_b0_tmp), "-force"],
                stdin=p1.stdout,
            )
        except Exception:
            p1.kill()
            p1.wait()
            raise
        finally:
            assert p1.stdout is not None
            p1.stdout.close()
        p2.communicate()
        if p1.wait() != 0 or p2.returncode != 0:
            mean_b0_tmp.unlink(missing_ok=True)
            raise PipelineError("Failed to compute mean b=0 image")
        mean_b0_tmp.rename(mean_b0)
    else:
        print(f"==> Reusing existing mean b=0 image: {mean_b0}")
    return mean_b0


def transform_drawem87_to_dwi(
    cfg: SubjectConfig,
    mean_b0: Path,
    dwi_source_dir: Path,
) -> tuple[Path, Path]:
    """Transform drawem87 from T2w to DWI space. Returns (nodes_87, xfm_path)."""
    nodes_87 = cfg.outdir / "nodes_dhcp_struct_87_in_dwi.mif"
    xfm_out = cfg.outdir / "T2w_to_dwi_mrtrix.txt"

    if cfg.force or not nodes_87.is_file():
        dhcp_xfm_dir = dwi_source_dir / cfg.subject_id / cfg.session_id / "xfm"
        fsl_t2_to_dwi = (
            dhcp_xfm_dir
            / f"{cfg.subject_id}_{cfg.session_id}_from-T2w_to-dwi_mode-image.mat"
        )
        if not fsl_t2_to_dwi.is_file():
            raise PipelineError(
                f"dHCP T2w-to-DWI transform not found: {fsl_t2_to_dwi}"
            )

        t2w_ref = cfg.scratch_subj / "T2w.mif"
        if not t2w_ref.is_file():
            raise PipelineError(f"T2w reference not found: {t2w_ref}")

        print("==> Converting FSL T2w->DWI affine to MRtrix format")
        run([
            "transformconvert", str(fsl_t2_to_dwi), str(t2w_ref), str(mean_b0),
            "flirt_import", str(xfm_out), "-force",
        ])

        print("==> Transforming Draw-EM 87-label parcellation to DWI space...")
        nodes_87_tmp = cfg.outdir / "nodes_87_unmasked_tmp.mif"
        run([
            "mrtransform", str(cfg.drawem87),
            "-linear", str(xfm_out),
            "-template", str(mean_b0),
            "-interp", "nearest",
            "-datatype", "uint16",
            "-nthreads", str(cfg.nthreads),
            str(nodes_87_tmp), "-force",
        ])

        run([
            "mrcalc", str(nodes_87_tmp), str(cfg.mask), "-multiply",
            "-datatype", "uint16",
            "-nthreads", str(cfg.nthreads),
            str(nodes_87), "-force",
        ])
        nodes_87_tmp.unlink(missing_ok=True)

        print(f"  87-label parcellation in DWI space: {nodes_87}")
        output = run_capture([
            "mrstats", str(nodes_87), "-mask", str(cfg.mask),
            "-ignorezero", "-output", "count",
        ])
        if output:
            print(f"  Voxel count: {output}")
    else:
        print(f"==> Reusing existing 87-label parcellation in DWI space: {nodes_87}")

    return nodes_87, xfm_out


def create_qc_parcellation_overlay(
    cfg: SubjectConfig, mean_b0: Path, nodes_87: Path,
) -> None:
    """Create parcellation QC overlay (colourised labels)."""
    qc_overlay = cfg.outdir / "qc_parcellation_overlay.mif"
    if cfg.force or cfg.force_qc or not qc_overlay.is_file():
        if mean_b0.is_file() and nodes_87.is_file():
            print("==> Creating parcellation QC overlay...")
            run([
                "label2colour", str(nodes_87), str(qc_overlay),
                "-nthreads", str(cfg.nthreads), "-force",
            ])
            print(
                f"  View with: mrview {mean_b0} "
                f"-overlay.load {qc_overlay} -overlay.opacity 0.5"
            )


def build_gm_parcellation(
    cfg: SubjectConfig,
    nodes_87: Path,
    gm_labels: list[GMLabel],
    lut_in: Path,
    lut_out: Path,
) -> tuple[Path, Path]:
    """Build GM-only parcellation with sequential node IDs. Returns (nodes_gm, lut_gm)."""
    p = cfg.parcellation
    n_nodes = len(gm_labels)
    nodes_gm = cfg.outdir / f"nodes_GM_{p}.mif"
    lut_gm = cfg.outdir / f"nodes_GM_{p}_lut.txt"

    if cfg.force or not nodes_gm.is_file() or not lut_gm.is_file():
        print(
            f"Building {n_nodes}-node GM parcellation [{p}] "
            f"from 87-label image (labelconvert)..."
        )
        print(f"Remapping original label IDs -> sequential node IDs (1-{n_nodes})")

        stamp = cfg.outdir / f".nodes_{p}_validated"
        if stamp.is_file():
            stamp.unlink()

        nodes_gm_tmp = cfg.outdir / f"nodes_GM_{p}_unmasked_tmp.mif"
        try:
            run([
                "labelconvert", str(nodes_87), str(lut_in), str(lut_out), str(nodes_gm_tmp),
                "-nthreads", str(cfg.nthreads), "-force",
            ])

            run([
                "mrcalc", str(nodes_gm_tmp), str(cfg.mask), "-multiply",
                "-datatype", "uint16",
                "-nthreads", str(cfg.nthreads),
                str(nodes_gm), "-force",
            ])
        except (subprocess.CalledProcessError, OSError):
            nodes_gm_tmp.unlink(missing_ok=True)
            nodes_gm.unlink(missing_ok=True)
            raise
        nodes_gm_tmp.unlink(missing_ok=True)

        write_lut_gm(gm_labels, lut_gm)

        print(f"Created GM parcellation: {nodes_gm}")
        print(f"Created LUT: {lut_gm}")
    else:
        print(f"Reusing existing {n_nodes}-node GM parcellation: {nodes_gm}")

    return nodes_gm, lut_gm


def create_gm_colour_image(cfg: SubjectConfig, nodes_gm: Path) -> Path:
    """Create colourised GM node image."""
    nodes_gm_colour = cfg.outdir / f"nodes_GM_{cfg.parcellation}_colour.mif"
    if cfg.force or not nodes_gm_colour.is_file():
        print(f"Building colourized GM node image: {nodes_gm_colour}")
        run([
            "label2colour", str(nodes_gm), str(nodes_gm_colour),
            "-nthreads", str(cfg.nthreads), "-force",
        ])
    else:
        print(f"Reusing existing colourized GM node image: {nodes_gm_colour}")
    return nodes_gm_colour


def validate_nodes(
    cfg: SubjectConfig,
    nodes_gm: Path,
    gm_labels: list[GMLabel],
) -> None:
    """Validate that GM nodes have voxels within the brain mask."""
    n_nodes = len(gm_labels)
    stamp = cfg.outdir / f".nodes_{cfg.parcellation}_validated"

    if not cfg.force and stamp.is_file():
        print(
            f"Node validation already completed "
            f"(using {n_nodes} GM nodes, --force to rerun)"
        )
        return

    print(f"Validating {n_nodes} GM nodes in masked region...")
    nonempty = 0
    empty_nodes: list[tuple[int, GMLabel]] = []

    for node_id in range(1, n_nodes + 1):
        p1 = subprocess.Popen(
            ["mrcalc", str(nodes_gm), str(node_id), "-eq", "-"],
            stdout=subprocess.PIPE,
        )
        try:
            p2 = subprocess.Popen(
                ["mrstats", "-", "-mask", str(cfg.mask), "-output", "count", "-ignorezero"],
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
                text=True,
            )
        except Exception:
            p1.kill()
            p1.wait()
            raise
        finally:
            assert p1.stdout is not None
            p1.stdout.close()
        stdout, _ = p2.communicate()
        if p1.wait() != 0 or p2.returncode != 0:
            raise PipelineError(
                f"Failed to check voxel count for node {node_id}"
            )

        vox = stdout.strip() if stdout else "0"
        if vox == "0" or not vox:
            empty_nodes.append((node_id, gm_labels[node_id - 1]))
        else:
            nonempty += 1

    print(f"  -> {nonempty}/{n_nodes} nodes have voxels within the brain mask.")

    if empty_nodes:
        print(f"  WARNING: {len(empty_nodes)} nodes have zero voxels:", file=sys.stderr)
        for nid, gm in empty_nodes:
            print(f"           Node {nid}(orig:{gm.label_id}) - {gm.name}", file=sys.stderr)

    min_nodes = n_nodes // 4
    if nonempty < min_nodes:
        raise PipelineError(
            f"Too few non-empty nodes ({nonempty} / {n_nodes}). "
            f"T2w->DWI transform may have failed.\n"
            f"       Inspect nodes_dhcp_struct_87_in_dwi.mif overlaid on "
            f"mean_b0.mif in mrview."
        )

    if nonempty < n_nodes // 2:
        print(
            f"  WARNING: Parcellation quality is poor ({nonempty}/{n_nodes} nodes). "
            f"Connectomes may be unreliable.",
            file=sys.stderr,
        )
        print("           Check T2w->DWI alignment visually!", file=sys.stderr)

    stamp.touch()


def compute_connectome(
    cfg: SubjectConfig,
    variant: str,
    tck: Path,
    wts: Path,
    nodes_image: Path,
) -> None:
    """Run tck2connectome for one variant."""
    p = cfg.parcellation
    csv = cfg.outdir / f"connectome_{p}_{variant}.csv"
    asg = cfg.outdir / f"assignments_{p}_{variant}.txt"

    if cfg.force or not csv.is_file() or not asg.is_file():
        print(f"tck2connectome [{p}] ({variant})...")
        csv_tmp = cfg.outdir / f"connectome_{p}_{variant}_tmp.csv"
        asg_tmp = cfg.outdir / f"assignments_{p}_{variant}_tmp.txt"
        try:
            run([
                "tck2connectome", str(tck), str(nodes_image), str(csv_tmp),
                "-tck_weights_in", str(wts),
                "-symmetric", "-zero_diagonal",
                "-stat_edge", "sum",
                "-assignment_radial_search", "4",
                "-out_assignments", str(asg_tmp),
                "-nthreads", str(cfg.nthreads),
                "-force",
            ])
            csv_tmp.rename(csv)
            asg_tmp.rename(asg)
        except (subprocess.CalledProcessError, OSError) as exc:
            csv_tmp.unlink(missing_ok=True)
            asg_tmp.unlink(missing_ok=True)
            raise PipelineError(f"tck2connectome failed ({variant})") from exc
    else:
        print(f"Reusing existing connectome ({variant}): {csv}")
        print("  WARNING: weights file may have changed since this was generated.", file=sys.stderr)
        print("           Re-run with --force to regenerate.", file=sys.stderr)


def compute_fa_map(cfg: SubjectConfig) -> Path:
    """Compute FA map from DWI via dwi2tensor (DKI) + tensor2metric.

    Uses -dkt for kurtosis-augmented tensor fitting on multi-shell data
    (Bastiani et al. 2019, DOI:10.1016/j.neuroimage.2018.05.064).
    """
    tensor = cfg.outdir / "dt.mif"
    dkt = cfg.outdir / "dkt.mif"
    fa = cfg.outdir / "FA.mif"

    if cfg.force or not fa.is_file():
        print("Computing FA map (dwi2tensor -dkt -> tensor2metric)...")
        tensor_tmp = cfg.outdir / "dt_tmp.mif"
        dkt_tmp = cfg.outdir / "dkt_tmp.mif"
        try:
            run([
                "dwi2tensor", str(cfg.dwi), str(tensor_tmp),
                "-dkt", str(dkt_tmp),
                "-mask", str(cfg.mask),
                "-nthreads", str(cfg.nthreads),
                "-force",
            ])
            tensor_tmp.rename(tensor)
            dkt_tmp.rename(dkt)
        except (subprocess.CalledProcessError, OSError) as exc:
            tensor_tmp.unlink(missing_ok=True)
            dkt_tmp.unlink(missing_ok=True)
            raise PipelineError("dwi2tensor failed") from exc

        fa_tmp = cfg.outdir / "FA_tmp.mif"
        try:
            run([
                "tensor2metric", str(tensor),
                "-fa", str(fa_tmp),
                "-mask", str(cfg.mask),
                "-nthreads", str(cfg.nthreads),
                "-force",
            ])
            fa_tmp.rename(fa)
        except (subprocess.CalledProcessError, OSError) as exc:
            fa_tmp.unlink(missing_ok=True)
            raise PipelineError("tensor2metric failed") from exc
    else:
        print(f"Reusing existing FA map: {fa}")

    return fa


def sample_fa_along_tracks(
    cfg: SubjectConfig,
    variant: str,
    tck: Path,
    fa: Path,
) -> Path:
    """Run tcksample to get mean FA per streamline."""
    p = cfg.parcellation
    fa_per_streamline = cfg.outdir / f"fa_per_streamline_{p}_{variant}.csv"

    if cfg.force or not fa_per_streamline.is_file():
        print(f"tcksample [{p}] ({variant}): sampling FA along streamlines...")
        fa_tmp = cfg.outdir / f"fa_per_streamline_{p}_{variant}_tmp.csv"
        try:
            run([
                "tcksample", str(tck), str(fa), str(fa_tmp),
                "-stat_tck", "mean",
                "-precise",
                "-nthreads", str(cfg.nthreads),
                "-force",
            ])
            fa_tmp.rename(fa_per_streamline)
        except (subprocess.CalledProcessError, OSError) as exc:
            fa_tmp.unlink(missing_ok=True)
            raise PipelineError(f"tcksample failed ({variant})") from exc
    else:
        print(f"Reusing FA per-streamline values ({variant}): {fa_per_streamline}")

    return fa_per_streamline


def compute_fa_connectome(
    cfg: SubjectConfig,
    variant: str,
    tck: Path,
    fa_per_streamline: Path,
    nodes_image: Path,
    weights_path: Path | None = None,
    suffix: str = "",
) -> None:
    """Run tck2connectome with FA weighting (mean FA per edge)."""
    p = cfg.parcellation
    tag = f"{variant}_fa{suffix}"
    csv = cfg.outdir / f"connectome_{p}_{tag}.csv"

    if cfg.force or not csv.is_file():
        print(f"tck2connectome [{p}] ({tag}): FA-weighted connectome...")
        csv_tmp = cfg.outdir / f"connectome_{p}_{tag}_tmp.csv"
        cmd = [
            "tck2connectome", str(tck), str(nodes_image), str(csv_tmp),
            "-scale_file", str(fa_per_streamline),
            "-symmetric", "-zero_diagonal",
            "-stat_edge", "mean",
            "-assignment_radial_search", "4",
        ]
        if weights_path is not None:
            cmd += ["-tck_weights_in", str(weights_path)]
        cmd += ["-nthreads", str(cfg.nthreads), "-force"]
        try:
            run(cmd)
            csv_tmp.rename(csv)
        except (subprocess.CalledProcessError, OSError) as exc:
            csv_tmp.unlink(missing_ok=True)
            raise PipelineError(f"tck2connectome FA failed ({tag})") from exc
    else:
        print(f"Reusing existing FA connectome ({tag}): {csv}")


def create_50k_subset(cfg: SubjectConfig, base_tck: Path, baseline_dir: Path) -> Path:
    """Extract 50k streamline subset for visualization."""
    tck_50k = baseline_dir / "tractogram_50k.tck"
    if cfg.force or not tck_50k.is_file():
        print("Extracting 50k streamline subset for visualization...")
        run(["tckedit", str(base_tck), str(tck_50k), "-number", "50000", "-force"])
    return tck_50k


def write_qc_commands(
    cfg: SubjectConfig,
    mean_b0: Path,
    nodes_87: Path,
    nodes_gm_colour: Path,
    nodes_image: Path,
    tck_50k: Path,
    n_nodes: int,
) -> None:
    """Write QC visualization commands file."""
    p = cfg.parcellation
    qc_file = cfg.outdir / f"mrview_qc_commands_{p}.txt"
    qc_file.write_text(f"""\
# QC Visualization Commands for Draw-EM Connectome Pipeline
# Subject: {cfg.subject_id}  Session: {cfg.session_id}
# Parcellation scheme: {p}  GM Nodes: {n_nodes}
# T2w->DWI: dHCP FSL affine (via transformconvert)

# 1. 87-label parcellation in DWI space
mrview {mean_b0} -mode 2 -fov 190 -noannotations -interpolation 0 \\
  -overlay.load {nodes_87} -overlay.opacity 0.7 -overlay.interpolation 0 &

# 2. GM Parcellation [{p}]: {n_nodes} nodes (remapped to 1-{n_nodes})
mrview {mean_b0} -mode 2 -fov 190 -noannotations -interpolation 0 \\
  -overlay.load {nodes_gm_colour} -overlay.opacity 0.7 -overlay.interpolation 0 &

# 3. GM parcellation + 50k streamline subset
mrview {mean_b0} -mode 2 -fov 190 -noannotations -interpolation 0 \\
  -overlay.load {nodes_gm_colour} -overlay.opacity 0.4 -overlay.interpolation 0 \\
  -tractography.load {tck_50k} &

# 4. Connectome graph view
mrview {mean_b0} \\
  -connectome.init {nodes_image} \\
  -connectome.load {cfg.outdir / f"connectome_{p}_combined.csv"} &
""")
    print(f"QC commands saved to: {qc_file}")


def apply_mu_scaling(
    csv_in: Path, mu_path: Path, csv_out: Path, force: bool = False,
) -> None:
    """Scale a connectome CSV by the SIFT2 proportionality coefficient mu."""
    if not force and csv_out.is_file():
        print(f"Reusing mu-scaled connectome: {csv_out}")
        return
    mu = float(mu_path.read_text().strip())
    print(f"Applying mu scaling ({mu:.6e}) to {csv_in.name} -> {csv_out.name}")
    with open(csv_in) as fin, open(csv_out, "w") as fout:
        for line in fin:
            vals = line.strip().split(",")
            scaled = [str(float(v) * mu) for v in vals]
            fout.write(",".join(scaled) + "\n")


# ---------------------------------------------------------------------------
# Single-subject processing
# ---------------------------------------------------------------------------

def process_subject(
    cfg: SubjectConfig,
    dwi_source_dir: Path,
    scheme: dict,
    all_labels: list[tuple[int, str]],
    gm_labels: list[GMLabel],
) -> bool:
    """Process a single subject/session. Returns True on success."""
    if not cfg.subj_dir.is_dir():
        print(f"ERROR: Subject/session directory not found: {cfg.subj_dir}", file=sys.stderr)
        return False

    print()
    print(f"Subject: {cfg.subject_id}  Session: {cfg.session_id}")

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    if not cfg.mask.is_file():
        print(f"ERROR: Brain mask not found: {cfg.mask}", file=sys.stderr)
        return False
    if not cfg.dwi.is_file():
        print(f"ERROR: DWI not found: {cfg.dwi}", file=sys.stderr)
        return False
    if not cfg.drawem87.is_file():
        print(f"ERROR: Draw-EM 87-label parcellation not found: {cfg.drawem87}", file=sys.stderr)
        print("       Run copy_data_dhcp.py first to convert drawem87.mif", file=sys.stderr)
        return False

    try:
        return _process_subject_inner(cfg, dwi_source_dir, scheme, all_labels, gm_labels)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: Command failed: {shlex.join(exc.cmd)}", file=sys.stderr)
        return False


def _process_subject_inner(
    cfg: SubjectConfig,
    dwi_source_dir: Path,
    scheme: dict,
    all_labels: list[tuple[int, str]],
    gm_labels: list[GMLabel],
) -> bool:
    """Core processing logic for a single subject (may raise PipelineError)."""
    mean_b0 = compute_mean_b0(cfg)
    nodes_87, mrtrix_xfm = transform_drawem87_to_dwi(cfg, mean_b0, dwi_source_dir)

    fa_map = compute_fa_map(cfg)

    create_qc_parcellation_overlay(cfg, mean_b0, nodes_87)

    label_merges = scheme["label_merges"]
    nodes_87_original = nodes_87
    if label_merges:
        nodes_87 = merge_labels_in_image(cfg, nodes_87, label_merges)

    baseline_dir = cfg.subj_dir / "tractography" / "max_pietsch_combined"
    sift2w_dir = cfg.subj_dir / "tractography" / "max_pietsch_dual_sift2w"
    baseline_msmt_dir = cfg.subj_dir / "tractography" / "baseline_msmt"

    base_tck = pick_existing_file("baseline tractogram", [
        baseline_dir / "tractogram.tck",
    ])
    base_wts = pick_existing_file("baseline SIFT2 weights", [
        baseline_dir / "sift2_weights.txt",
    ])
    base_mu_file = pick_existing_file("baseline SIFT2 mu", [
        baseline_dir / "sift2_mu.txt",
    ])
    sift2w_tck = pick_existing_file("sift2w tractogram", [
        sift2w_dir / "tractogram.tck",
        baseline_dir / "tractogram.tck",
    ])
    yng_wts = pick_existing_file("young SIFT2 weights", [
        sift2w_dir / "sift2_weights_y.txt",
    ])
    yng_mu_file = pick_existing_file("young SIFT2 mu", [
        sift2w_dir / "sift2_mu_y.txt",
    ])
    old_wts = pick_existing_file("old SIFT2 weights", [
        sift2w_dir / "sift2_weights_o.txt",
    ])
    old_mu_file = pick_existing_file("old SIFT2 mu", [
        sift2w_dir / "sift2_mu_o.txt",
    ])

    baseline_msmt_tck = baseline_msmt_dir / "tractogram.tck"
    baseline_msmt_wts = baseline_msmt_dir / "sift2_weights.txt"
    baseline_msmt_mu_file = baseline_msmt_dir / "sift2_mu.txt"
    have_baseline_msmt = all(
        f.is_file()
        for f in [baseline_msmt_tck, baseline_msmt_wts, baseline_msmt_mu_file]
    )
    if not have_baseline_msmt:
        for f in [baseline_msmt_tck, baseline_msmt_wts, baseline_msmt_mu_file]:
            if not f.is_file():
                print(f"WARNING: baseline_msmt tractography file missing: {f}", file=sys.stderr)
        print("         Skipping baseline_msmt connectome generation.", file=sys.stderr)

    p = cfg.parcellation
    n_nodes = len(gm_labels)
    lut_in = cfg.outdir / f"drawem87_{p}_lut_in.txt"
    lut_out = cfg.outdir / f"drawem87_{p}_lut_out.txt"
    write_lut_in(
        all_labels, lut_in,
        renames=scheme.get("label_renames"),
        skip_ids=set(label_merges.keys()),
    )
    write_lut_out(gm_labels, lut_out)

    nodes_gm, lut_gm = build_gm_parcellation(
        cfg, nodes_87, gm_labels, lut_in, lut_out,
    )
    nodes_gm_colour = create_gm_colour_image(cfg, nodes_gm)

    validate_nodes(cfg, nodes_gm, gm_labels)

    nodes_image = nodes_gm

    compute_connectome(cfg, "combined", base_tck, base_wts, nodes_image)
    compute_connectome(cfg, "young", sift2w_tck, yng_wts, nodes_image)
    compute_connectome(cfg, "old", sift2w_tck, old_wts, nodes_image)
    if have_baseline_msmt:
        compute_connectome(
            cfg, "baseline_msmt", baseline_msmt_tck, baseline_msmt_wts, nodes_image,
        )

    apply_mu_scaling(
        cfg.outdir / f"connectome_{p}_combined.csv",
        base_mu_file,
        cfg.outdir / f"connectome_{p}_combined_mu.csv",
        cfg.force,
    )
    apply_mu_scaling(
        cfg.outdir / f"connectome_{p}_young.csv",
        yng_mu_file,
        cfg.outdir / f"connectome_{p}_young_mu.csv",
        cfg.force,
    )
    apply_mu_scaling(
        cfg.outdir / f"connectome_{p}_old.csv",
        old_mu_file,
        cfg.outdir / f"connectome_{p}_old_mu.csv",
        cfg.force,
    )
    if have_baseline_msmt:
        apply_mu_scaling(
            cfg.outdir / f"connectome_{p}_baseline_msmt.csv",
            baseline_msmt_mu_file,
            cfg.outdir / f"connectome_{p}_baseline_msmt_mu.csv",
            cfg.force,
        )

    fa_combined = sample_fa_along_tracks(cfg, "combined", base_tck, fa_map)
    compute_fa_connectome(cfg, "combined", base_tck, fa_combined, nodes_image)
    compute_fa_connectome(
        cfg, "combined", base_tck, fa_combined, nodes_image,
        weights_path=base_wts, suffix="_sift2w",
    )

    if have_baseline_msmt:
        fa_baseline = sample_fa_along_tracks(
            cfg, "baseline_msmt", baseline_msmt_tck, fa_map,
        )
        compute_fa_connectome(
            cfg, "baseline_msmt", baseline_msmt_tck, fa_baseline, nodes_image,
        )
        compute_fa_connectome(
            cfg, "baseline_msmt", baseline_msmt_tck, fa_baseline, nodes_image,
            weights_path=baseline_msmt_wts, suffix="_sift2w",
        )

    tck_50k = create_50k_subset(cfg, base_tck, baseline_dir)

    write_qc_commands(
        cfg, mean_b0, nodes_87_original, nodes_gm_colour, nodes_image,
        tck_50k, n_nodes,
    )

    print(f"Done: {cfg.subject_id} {cfg.session_id}")
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
        description=(
            "Build structural connectomes from dHCP neonatal data using the "
            "per-subject Draw-EM 87-label parcellation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --dhcp-dir /data/dhcp --scratch-dir /data/scratch \\
      --dwi-source /data/shard --anat-source /data/anat
  %(prog)s --dhcp-dir /data/dhcp --scratch-dir /data/scratch \\
      --dwi-source /data/shard --anat-source /data/anat \\
      --parcellation full_split_stn
  %(prog)s --dhcp-dir /data/dhcp --scratch-dir /data/scratch \\
      --dwi-source /data/shard --anat-source /data/anat \\
      -n 16 sub-EXAMPLE/ses-00000
""",
    )
    parser.add_argument(
        "subjects", nargs="*", metavar="subject/session",
        help="One or more sub-XX/ses-YY paths (default: discover all in --dhcp-dir)",
    )
    parser.add_argument(
        "--dhcp-dir", type=Path, required=True,
        help="Directory containing processed sub-XX/ses-YY/ session data",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, required=True,
        help="Directory containing DWI/mask inputs (postmc-dwi.mif, reconmask.mif, etc.)",
    )
    parser.add_argument(
        "--dwi-source", type=Path, required=True,
        help="dHCP rel3 dMRI SHARD pipeline directory (for T2w-to-DWI transforms)",
    )
    parser.add_argument(
        "--anat-source", type=Path, required=True,
        help="dHCP rel3 anat pipeline directory (for drawem87 TSV, participants.tsv)",
    )
    parser.add_argument(
        "-n", "--nthreads", type=int, default=None, metavar="N",
        help="Number of threads (default: auto-detect, max 8)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute all outputs even if present",
    )
    parser.add_argument(
        "--force-qc", action="store_true",
        help="Regenerate only QC images",
    )
    parser.add_argument(
        "--parcellation",
        choices=list(PARCELLATION_SCHEMES.keys()),
        default="cortical",
        help=(
            "Parcellation scheme: cortical (34 nodes), "
            "full_merged (45), full_merged_stn (47), "
            "full_split (47), full_split_stn (49). "
            "Default: cortical"
        ),
    )
    args = parser.parse_args(argv)

    nthreads = args.nthreads if args.nthreads is not None else detect_nthreads()
    scheme_name = args.parcellation
    scheme = PARCELLATION_SCHEMES[scheme_name]

    dhcp_dir = args.dhcp_dir
    scratch_dir = args.scratch_dir
    dwi_source_dir = args.dwi_source
    anat_source_dir = args.anat_source

    print(f"Using {nthreads} thread(s)")
    print(f"Parcellation scheme: {scheme_name} ({scheme['description']})")

    if args.subjects:
        subjects_to_process = args.subjects
    else:
        if not dhcp_dir.is_dir():
            print(f"ERROR: DHCP directory not found: {dhcp_dir}", file=sys.stderr)
            return 1
        subjects_to_process = discover_subjects(dhcp_dir)

    if not subjects_to_process:
        print("ERROR: No subjects / sessions found to process.", file=sys.stderr)
        return 1

    print(
        f"Connectome Generation - processing {len(subjects_to_process)} subject(s)"
    )

    drawem87_tsv = anat_source_dir / "desc-drawem87_dseg.tsv"
    print(f"Extracting GM labels from: {drawem87_tsv}")
    all_labels, _ = parse_drawem87_tsv(drawem87_tsv)
    gm_labels = select_gm_labels(all_labels, scheme)
    if not gm_labels:
        print(
            f"ERROR: No GM labels selected for scheme '{scheme_name}'. "
            f"Check the Draw-EM 87-label TSV and parcellation scheme definition.",
            file=sys.stderr,
        )
        return 1
    n_nodes = len(gm_labels)
    print(f"Selected {n_nodes} GM nodes [{scheme_name}]:")
    for i, gm in enumerate(gm_labels, 1):
        print(f"  {i:3d} <- label {gm.label_id:3d}  {gm.name}")
    if scheme["label_merges"]:
        merge_desc = ", ".join(f"{s}->{t}" for s, t in scheme["label_merges"].items())
        print(f"Label merges (applied to image): {merge_desc}")

    required_cmds = [
        "mrcalc", "tck2connectome", "dwiextract", "mrmath", "mrstats",
        "label2colour", "labelconvert", "transformconvert", "mrtransform", "tckedit",
        "dwi2tensor", "tensor2metric", "tcksample",
    ]
    missing = [cmd for cmd in required_cmds if not have(cmd)]
    if missing:
        print(
            f"ERROR: Missing required commands: {' '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    failed = 0
    for i, rel in enumerate(subjects_to_process, 1):
        ses = dhcp_dir / rel
        if not ses.is_dir():
            print(f"WARNING: Skipping missing session dir: {ses}", file=sys.stderr)
            failed += 1
            continue

        print()
        print("=" * 40)
        print(f"[{i}/{len(subjects_to_process)}] {rel}")
        print("=" * 40)

        scratch_subj = scratch_dir / rel

        cfg = SubjectConfig(
            subj_dir=ses,
            mask=scratch_subj / "reconmask.mif",
            dwi=scratch_subj / "postmc-dwi.mif",
            drawem87=scratch_subj / "drawem87.mif",
            scratch_subj=scratch_subj,
            force=args.force,
            force_qc=args.force_qc,
            nthreads=nthreads,
            parcellation=scheme_name,
        )

        if not process_subject(cfg, dwi_source_dir, scheme, all_labels, gm_labels):
            print(f"ERROR: Failed for {rel}", file=sys.stderr)
            failed += 1

    print()
    print(f"Batch done. Failed subjects: {failed}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

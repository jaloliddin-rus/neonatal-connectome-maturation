# Neonatal Structural Connectome Maturation

Quantifying structural brain connectivity development in neonates using diffusion MRI data from the developing Human Connectome Project (dHCP).

## Overview

This pipeline implements a two-component multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD) framework (Pietsch et al. 2019) to decompose white matter fibre orientation distributions into younger (immature) and older (mature) tissue components. SIFT2-weighted connectomes are built from each component independently, yielding a per-edge **maturation index**:

```
M = w_o / (w_o + w_y)
```

where `w_o` and `w_y` are the SIFT2-weighted connection strengths from the older and younger response functions, respectively. Higher M indicates more mature connectivity.

## Repository Structure

```
conference/         MICCAI PIPPI 2026 workshop paper
  paper.pdf         Paper (added after publication)
  pipeline/         Pipeline scripts (34-node cortical parcellation)
  data/             Response functions (Pietsch et al. 2019 atlas)

journal/            Journal paper (forthcoming, may use different parameters)
```

## Pipeline

```
data_prep/          Convert dHCP BIDS data to MRtrix format, select subjects
     |
odf_est/            Two-component MSMT-CSD (Pietsch atlas responses)
     |
tractography/       Whole-brain iFOD2 tractography + dual SIFT2
     |                  + SIFT2 weight=1 filtering
connectome_gen/     Draw-EM parcellation -> DWI space, build connectomes
```

## Requirements

**System:**
- Python >= 3.10
- [MRtrix3](https://www.mrtrix.org/) >= 3.0.4
- [ANTs](https://github.com/ANTsX/ANTs) >= 2.x

**Python packages:**
```
pip install -r requirements.txt
```

## Data

- Requires access to [dHCP release data](https://biomedia.github.io/dHCP-release-notes/)
- Pietsch et al. (2019) atlas response functions are included in `conference/data/response_functions/`

## Usage

Each pipeline stage requires paths to the dHCP data directories:

```bash
# 1. Convert dHCP data to MRtrix format
python conference/pipeline/data_prep/copy_data_dhcp.py \
    --dwi-source /path/to/rel3_dhcp_dmri_shard_pipeline \
    --anat-source /path/to/rel3_dhcp_anat_pipeline \
    --output-dir /path/to/output \
    --scan-info /path/to/scan_info.csv

# 2. ODF estimation (two-component MSMT-CSD)
python conference/pipeline/odf_est/odf_estimation.py \
    --dhcp-dir /path/to/processed \
    --scratch-dir /path/to/scratch

# 3. Tractography + SIFT2
python conference/pipeline/tractography/tractography.py \
    --dhcp-dir /path/to/processed \
    --scratch-dir /path/to/scratch

# 4. SIFT2 weight=1 filtering
python conference/pipeline/tractography/tractography_sift2_weight1_filter.py \
    --dhcp-dir /path/to/processed \
    --scratch-dir /path/to/scratch

# 5. Connectome generation
python conference/pipeline/connectome_gen/connectome_generation.py \
    --dhcp-dir /path/to/processed \
    --scratch-dir /path/to/scratch \
    --dwi-source /path/to/rel3_dhcp_dmri_shard_pipeline \
    --anat-source /path/to/rel3_dhcp_anat_pipeline
```

## Citation

If you use this code, please cite:

> Rustamov J, Leysen S, Radwan A, Christiaens D, Damseh R. *Mapping Structural Connectivity in the Neonatal Developing Brain using Multi-Component Tissue Modelling*. In: Perinatal, Preterm and Paediatric Image Analysis (PIPPI 2026), MICCAI 2026 Satellite Events. Lecture Notes in Computer Science. Springer; 2026 (to appear).

```bibtex
@inproceedings{rustamov2026mapping,
  title     = {Mapping Structural Connectivity in the Neonatal Developing Brain using Multi-Component Tissue Modelling},
  author    = {Rustamov, Jaloliddin and Leysen, Siebe and Radwan, Ahmed and Christiaens, Daan and Damseh, Rafat},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Satellite Events},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {Perinatal, Preterm and Paediatric Image Analysis (PIPPI 2026) workshop. To appear}
}
```

A machine-readable version is in [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button uses it.

## License

MIT License. See [LICENSE](LICENSE).

# Cascading Accessibility Risk in Urban Networks
## Graph Policy Learning for Resilient Public-Service Planning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.12.0%2Bcu126-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Reproducibility-v1.1.0--B-2F6F9F" alt="Reproducibility">
  <img src="https://img.shields.io/badge/Study%20area-MEL%2C%20France-555555" alt="Study area">
  <a href="https://doi.org/10.5281/zenodo.22660511">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22660511.svg" alt="DOI">
  </a>
</p>

This repository contains the public reproducibility pipeline for the paper:

> **Cascading Accessibility Risk in Urban Networks: Graph Policy Learning for Resilient Public-Service Planning**

The code implements the empirical workflow linking potential accessibility,
disruption trajectories, sequential public restoration, and graph based policy
learning for **Métropole Européenne de Lille (MEL), France**.

### Reproducibility resources

- **Source code and computational workflow:** this GitHub repository.
- **Archived empirical and reproducibility data:** [Zenodo Dataset — DOI: 10.5281/zenodo.22660511](https://doi.org/10.5281/zenodo.22660511).

The Zenodo archive contains the empirical and reproducibility data associated
with the study, while this repository contains the computational pipeline used
to construct the analysis, perform the statistical evaluation, and generate the
reported tables and figures.

The canonical public entry point is the `reproduce` stage. It checks or
reconstructs the required public-data inputs, validates the frozen disruption
design, trains the graph policies and controlled ablations, performs exact
held-out evaluation, regenerates the publication source data, figures, and
tables, and assembles a checksummed reproducibility bundle.

## 1. Repository scope

The pipeline is organized around the empirical counterparts of the manuscript's analytical framework:

- **Urban system and Cascading Accessibility Risk (CAR):** spatial units, essential health-service opportunities, road-network accessibility, disruption trajectories, and cumulative accessibility consequences.
- **Sequential public intervention:** capacity restoration under a pathwise budget and a common empirical action correspondence.
- **Graph policy learning:** a parameterized graph policy trained under a frozen numerical protocol and compared with benchmark and ablation specifications.
- **Held-out evaluation:** trajectory-separated ordinary and structural holdouts, exact full-network outcome evaluation, paired inference, and publication-ready outputs.

The code is deliberately **fail-fast** for provenance, leakage, feasibility, protocol drift, and numerical consistency. It does not silently replace missing official hazard information with a proxy in the publication protocol.

---

## 2. Reference computational environment

The final reference run used the following GPU environment:

| Component | Reference configuration |
|---|---|
| Pipeline version | `1.1.0-B` |
| Python | Python 3.x; exact runtime is recorded automatically in the reproduction manifest |
| PyTorch | `2.12.0+cu126` |
| CUDA | Available |
| Reference GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Tensor precision | `float32` |
| Numerical seeds | `101, 202, 303, 404, 505, 606, 707, 808, 909, 1001` |
| Operating-system support | Windows-safe multiprocessing (`spawn`); portable execution is also supported |
| Study CRS | EPSG:2154 |
| Filosofi source CRS | EPSG:3035 |

CUDA is preferred automatically when available. GPU tensors remain in the parent process, while CPU-heavy work can use spawned workers where appropriate.

### Strict versus portable reproduction

Two reproduction modes are intentionally distinguished.

**Exact reference-environment check**

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce --strict-reference-env
```

With `--strict-reference-env`, execution requires the reference PyTorch build, CUDA availability, and the reference GPU name. A mismatch causes a fail-fast error.

**Portable reproduction**

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce
```

Without the strict flag, runtime differences are recorded in the provenance files and reproduction is allowed to proceed. This is the recommended mode for reviewers using a different CUDA GPU or compatible environment.

> **Important:** the frozen ten-seed publication design is enforced by the scientific stages. Changing the local seed list does not redefine the published numerical protocol.

---

## 3. Software requirements

The pipeline uses the scientific Python and geospatial ecosystem, including:

- `numpy`
- `pandas`
- `scipy`
- `networkx`
- `requests`
- `geopandas`
- `shapely`
- `pyproj`
- `osmnx`
- `pyrosm`
- `rasterio`
- `pyogrio`
- `matplotlib`
- `torch`

A CUDA-enabled PyTorch installation is recommended for graph-policy training.

For the **exact reference environment**, install the PyTorch build matching `2.12.0+cu126` using the installation procedure appropriate for your operating system and NVIDIA driver. For portable reproduction, a compatible recent PyTorch/CUDA environment can be used; the pipeline records the detected versions automatically.


## 4. Public data and provenance

The empirical application is constructed from public sources. The pipeline records resolved resources, local paths, and SHA-256 hashes in machine-readable manifests.

| Empirical object | Provider / source | Pipeline use |
|---|---|---|
| MEL study boundary | OpenStreetMap / Nominatim via OSMnx | Study-area geometry |
| Road network | OpenStreetMap; cached Geofabrik PBF through Pyrosm, with Overpass fallback where applicable | Directed driving network |
| Essential health services | OpenStreetMap | Hospitals, clinics, and emergency-ward tagged facilities; maximum retained facilities: 40 |
| Population and socioeconomic attributes | INSEE Filosofi 2021, resolved through data.gouv.fr | 200 m population grid and vulnerability attributes |
| Flood-hazard support | DREAL Hauts-de-France / Géorisques, TRI Lille | Empirical spatial support for disruption construction |
| Géorisques WMS layers | `ALEA_SYNT_01_01FOR_FXX`, `ALEA_SYNT_01_02MOY_FXX`, `ALEA_SYNT_01_04FAI_FXX` | Frequent, mean, and rare fluvial-flood support |

The publication configuration sets `hazard_fallback_proxy = false`. If the required machine-readable official hazard support cannot be resolved, the standard workflow stops rather than treating a debugging proxy as observed hazard data.

Raw and processed source provenance is stored under:

```text
CEUS_LILLE_DATA/
├── data_raw/
├── data_processed/
└── manifests/
```

The data manifest records source metadata and checksums. Users should consult the original providers for the applicable data licences, attribution requirements, and terms of reuse.

---

## 5. Frozen publication design

The table below summarizes the principal parameters enforced or used by the final `v1.1.0-B` publication workflow.

| Block | Parameter | Publication value |
|---|---|---:|
| Spatial | Maximum modeled residential zones | 1,800 |
| Spatial | Vulnerability rule | Bottom income/living-standard quartile |
| Spatial | Vulnerable quantile | 0.25 |
| Services | Maximum retained health facilities | 40 |
| Services | Service supply | Unit opportunity weight |
| Network | Network type | Driving |
| Network | Capacity per lane | 900 veh/h |
| Network | Reference demand scale | 0.08 veh/h/person |
| Routing | BPR alpha | 0.15 |
| Routing | BPR beta | 4.0 |
| Routing | Assignment iterations | 6 |
| Routing | Damping | 0.45 |
| Routing | Route-choice temperature | 0.08 |
| Routing | Nearest services per zone | 3 |
| Routing | Candidate paths per OD | 4 |
| Accessibility | Impedance | Exponential |
| Accessibility | Half-life | 20 min |
| Accessibility | Temporal discount factor | 0.95 |
| Disruptions | Horizon `T` | 5 |
| Disruptions | Severity levels | 0.25, 0.50, 0.75, 1.00 |
| Disruptions | Spatial extents | 300, 800, 1,600 m |
| Scenarios | Training trajectories | 180 |
| Scenarios | Validation trajectories | 60 |
| Scenarios | Ordinary test trajectories | 100 |
| Scenarios | Structural holdout trajectories | 80 |
| Structural holdout | Withheld combination | Severity 1.00 × extent 1,600 m |
| Intervention | Restoration fraction of remaining deficit | 0.60 |
| Intervention | Restoration retention | 0.90 |
| Intervention | Reference budget | Median cost of 4 actions |
| Intervention | Candidate cap | 160 edges per decision step |
| Intervention | Final screening rule | `r3_abs_deficit_x_candidate_path_use` |
| Risk | Vulnerable-group loading | 0.50 |
| Risk | Tail-risk loading | 0.75 |
| Risk | CVaR level | 0.90 |
| Graph policy | Hidden dimension | 64 |
| Graph policy | Message-passing layers | 2 |
| Graph policy | Learning rate | `3e-4` |
| Graph policy | Weight decay | `1e-5` |
| Graph policy | Publication training epochs | 30 |
| Graph policy | Episodes per epoch | 6 |
| Graph policy | Validation frequency | Every 5 epochs |
| Graph policy | Early-stopping patience | 3 |
| Graph policy | Training validation subset | 8 trajectories |
| Graph policy | Training accessibility evaluator | Candidate-path surrogate |
| Graph policy | Final evaluation | Exact full-network accessibility |
| Numerical design | Independent numerical seeds | 10 |
| Statistics | Bootstrap resamples | 10,000 |
| Statistics | Confidence level | 95% |
| Statistics | Sign-flip resamples | 100,000 |
| Statistics | Holm adjustment | Enabled |

The scientific stages apply a **publication protocol lock** after configuration merging. In particular, the ten canonical seeds, the Top-160 action correspondence, and the final training/evaluation settings are frozen to prevent accidental protocol drift.

---

## 6. Recommended directory layout

Place the pipeline in the repository root. By default, all data and generated artifacts are stored in `CEUS_LILLE_DATA/`.

```text
repository/
├── ceus_lille_q1pp_pipeline_v1_1_0_B.py
├── requirements.txt
├── README.md
└── CEUS_LILLE_DATA/
    ├── ceus_lille_params.json
    ├── data_raw/
    ├── data_processed/
    ├── scenarios/
    ├── models/
    ├── evaluation/
    ├── figure_data/
    ├── figures/
    ├── table_data/
    ├── tables/
    ├── manifests/
    ├── logs/
    └── reproducibility_v1_1_0/
```

A different root can be supplied with `--root`:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --root /path/to/CEUS_LILLE_DATA --stage reproduce
```

---

## 7. Canonical reproduction workflow

### 7.1 Reviewer workflow

From the repository directory:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce
```

For the exact reference hardware/software check:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce --strict-reference-env
```

The workflow is **resumable**. If an interrupted `v1.1.0` run is detected, completed checkpoints and compatible work files are reused.

To restart learned models and final evaluation while preserving processed public data, scenarios, baseline products, and reusable exact-state cache:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce --fresh-run
```

### 7.2 What `reproduce` executes

The canonical workflow performs seven ordered stages:

1. **Baseline / reference engine** — validates or constructs the public-data inputs, scenarios, routing/accessibility baseline, and CAR objects.
2. **Primary graph-policy training** — trains B5 policy-gradient and B5-PPO variants under the frozen ten-seed design.
3. **Controlled ablation training** — estimates A1–A5.
4. **Evaluation preparation** — constructs benchmark objects, audits/reuses the exact-state cache, and freezes the B2 action sequence.
5. **Exact held-out evaluation** — evaluates B0–B5 and the trained variants on the prescribed held-out trajectories.
6. **Publication source data** — creates versioned CSV sources for the main tables and figures.
7. **Publication assets and reproducibility bundle** — renders the final outputs and creates the checksummed bundle.

---

## 8. Modular and expert stages

Individual stages are exposed for transparent inspection and recovery:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage init-config
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage download
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage process
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage disruptions
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage baseline
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage train
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage train-ablations
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage evaluate
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage publication-data
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage figures
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage tables
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage publication-assets
```

The script also exposes audit and precomputation stages intended for expert diagnostics and long-run recovery. For ordinary reproduction, use `--stage reproduce` rather than manually composing these internal stages.

To inspect all currently supported command-line options:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --help
```

---

## 9. Configuration

Generate the explicit default configuration with:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage init-config
```

This creates:

```text
CEUS_LILLE_DATA/ceus_lille_params.json
```

To overwrite an existing generated configuration deliberately:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage init-config --force-config
```

An additional JSON override can be supplied with:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --config my_override.json --stage reproduce
```

For scientific stages, the final publication protocol lock is applied **after** configuration merging. Consequently, publication-critical settings cannot be silently changed by an override.

---

## 10. Output products

The final publication namespace is versioned so that `v1.1.0` outputs are not mixed with legacy runs.

### Main publication figures

The pipeline creates exactly **five main figures**, rendered from saved versioned CSV source data. Figures are exported as publication-quality PDF and 600 dpi PNG files.

```text
CEUS_LILLE_DATA/
├── figure_data/q1pp_v1_1_0/
└── figures/main_v1_1_0/
```

### Main publication tables

The pipeline creates exactly **five main tables** from saved CSV sources, with CSV and LaTeX outputs.

```text
CEUS_LILLE_DATA/
├── table_data/q1pp_v1_1_0/
└── tables/main_v1_1_0/
```

The five table sources are:

```text
table1_open_data_empirical_design.csv
table2_policy_specifications.csv
table3_exact_performance_by_regime.csv
table4_paired_inference.csv
table5_ablation_optimizer_robustness.csv
```

---

## 11. Reproducibility bundle

A successful canonical run creates:

```text
CEUS_LILLE_DATA/reproducibility_v1_1_0/
├── figures/
├── tables/
├── figure_data/
├── table_data/
├── logs/
│   └── pipeline.log
├── manifests/
├── model_checkpoint_checksums.csv
├── reproducibility_manifest.json
└── run_state.json
```

The bundle is designed to make the publication outputs independently auditable:

- `reproducibility_manifest.json` inventories bundled files and their SHA-256 hashes;
- `model_checkpoint_checksums.csv` records checkpoint paths, sizes, and SHA-256 hashes without duplicating the model binaries;
- `run_state.json` records whether the reproduction run completed;
- the environment manifest records Python, platform, package versions, PyTorch, CUDA, GPU, and canonical-seed checks;
- publication figures and tables are accompanied by their source CSVs;
- the pipeline log preserves the execution trace.

A completed run should report a final message of the form:

```text
CEUS LILLE PUBLIC REPRODUCTION WORKFLOW v1.1.0 COMPLETE
```

---

## 12. Reproducibility checks

The workflow performs explicit checks before and during reproduction.

### Numerical protocol

The canonical numerical seeds are fixed to:

```text
[101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
```

A mismatch is treated as protocol drift.

### Runtime provenance

At startup the pipeline records, and optionally strictly verifies:

```text
torch=2.12.0+cu126
CUDA=True
GPU=NVIDIA GeForce RTX 3060 Laptop GPU
```

### Processed network consistency

When cached processed data are reused, the projected GraphML and processed edge table are checked for edge-count consistency before network reconstruction/download is skipped.

### Scenario integrity

The frozen scenario bank contains **420 complete trajectories**:

```text
180 training
 60 validation
100 ordinary test
 80 structural holdout
```

The workflow checks both split counts and the presence of every scenario file before reuse.

### Exact evaluation cache

The expensive exact-state precomputation is reusable only when its completion manifest and cache pass consistency checks. A valid completed cache is reused rather than recomputed.

### Output integrity

The final reproducibility bundle is checksummed recursively with SHA-256.

---

## 13. Reproducing only publication assets

If exact evaluation has already completed and the goal is only to regenerate the publication source files and rendered assets:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage publication-data
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage figures
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage tables
```

or:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage publication-assets
```

The publication readers require the completed exact-evaluation outputs; they do not silently substitute incomplete work files.

---

## 14. Reproducibility notes

### Determinism

The numerical design fixes ten seeds, but `deterministic_torch` is `false` in the reference configuration. Exact bitwise identity across different GPUs, CUDA libraries, drivers, operating systems, and parallel execution environments should therefore **not** be assumed. The strict mode verifies the designated reference runtime; the portable mode records deviations transparently.

### Structural holdout

The structural holdout is a **prespecified withheld disruption configuration on the same reference network**: severity `1.0` and spatial extent `1600 m`. It should not be interpreted as cross-city, cross-network, or distribution-free generalization.

### Spatial representation

The computational model uses at most 1,800 residential zones through a deterministic population-weighted stratified reduction. This is a computational reduction, not a probability sample of MEL.

### Essential-service representation

The empirical service layer retains up to 40 health-service locations with unit opportunity weights. The resulting accessibility object represents potential spatial access under the implemented network and impedance specification; it does not encode facility capacity, waiting time, realized utilization, service quality, or health-care need.

---

## 15. Troubleshooting

**CUDA is not detected**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `False` is returned, verify the NVIDIA driver and the installed CUDA-enabled PyTorch build. Portable CPU execution may still be possible for compatible stages, but the strict reference-environment check will fail.

**`--strict-reference-env` fails on another NVIDIA GPU**

This is expected. The flag intentionally requires the exact reference GPU name in addition to the PyTorch/CUDA checks. Use:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce
```

to perform portable reproduction while recording the environment difference.

**A public data endpoint is temporarily unavailable**

Re-run the pipeline after the endpoint becomes available. Existing valid cached files are reused. The publication workflow intentionally fails rather than silently substituting unofficial hazard data.

**A long run is interrupted**

Run the same command again:

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce
```

The reproduction workflow is designed to resume compatible completed work.

**A completely fresh learning/evaluation run is required**

```bash
python ceus_lille_q1pp_pipeline_v1_1_0_B.py --stage reproduce --fresh-run
```

This resets learned-model and final-evaluation state while preserving expensive reusable deterministic inputs and compatible caches.

---

## 16. Citation

If you use this repository, please cite the associated paper:

> **Cascading Accessibility Risk in Urban Networks: Graph Policy Learning for Resilient Public-Service Planning.**

A complete bibliographic citation and DOI should be added here once assigned by the journal.

---

## 17. Data attribution

This repository does not claim ownership of the underlying public datasets. Users should cite and comply with the terms of the original data providers, including **OpenStreetMap contributors**, **INSEE / Filosofi**, **data.gouv.fr**, **DREAL Hauts-de-France**, and **Géorisques**, as applicable to the retrieved resources.

The generated manifests provide the provenance information needed to identify the resources actually used by a run.

---

## 18. Reproducibility statement

The public `v1.1.0-B` workflow is designed so that the paper-facing computational results can be traced from public-data acquisition and processed inputs through the frozen scenario design, model training, exact held-out evaluation, statistical comparisons, publication source CSVs, and rendered tables and figures. Runtime metadata, model-checkpoint hashes, output hashes, logs, and run state are retained to make deviations from the reference computation explicit rather than silent.

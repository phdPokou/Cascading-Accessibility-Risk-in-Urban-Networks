#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CEUS Lille Q1++ reproducible pipeline
====================================

Empirical pipeline aligned with Sections 3--5 of the manuscript:
  * Urban System and Cascading Accessibility Risk
  * Sequential Public Intervention Problem
  * Graph Based Policy Learning for Sequential Intervention

Main stages
-----------
  reproduce         end-to-end public reproduction: data checks -> training -> exact evaluation -> Q1++ outputs
  init-config       write a fully explicit JSON configuration
  download          acquire/cache Lille public data
  process           build spatial units, services and directed network
  disruptions       generate trajectory-separated disruption scenarios
  baseline          calibrate routing/accessibility and uncontrolled CAR objects
  train             estimate B5 graph policy for 10 numerical seeds
  evaluate          evaluate B0--B5 and A1--A5 on common held-out trajectories
  publication-data  build versioned CSV source data for five Q1++ figures and five tables
  figures           render five Q1++ main figures strictly from saved CSV source data
  tables            render five Q1++ main tables from saved CSV source data
  publication-assets build CSV sources, then render all five figures and five tables
  all               run the complete pipeline in dependency order

The script is deliberately fail-fast about data provenance, leakage, feasibility,
and numerical identities. It never silently substitutes test information into training.

Public-data strategy
--------------------
  * Study-area boundary: OpenStreetMap through OSMnx/Nominatim.
  * Road network + essential-health facilities: OpenStreetMap via cached
    Geofabrik PBF/Pyrosm, with Overpass fallback.
  * Population/socioeconomic attributes: INSEE Filosofi 2021 200m grid, resolved
    dynamically from data.gouv.fr.
  * Flood-hazard support: DREAL Hauts-de-France, "Aléa débordement de cours d’eau -
    TRI Lille", resolved dynamically from data.gouv.fr. If a machine-readable vector
    resource is temporarily unavailable, the script stops by default rather than
    pretending that a proxy is observed hazard data. An explicit opt-in proxy mode is
    available for debugging only and is tagged in every manifest/output.

Windows / CUDA
--------------
  * multiprocessing uses the 'spawn' context and all entry points are protected by
    if __name__ == '__main__'.
  * GPU tensors remain in the parent process; CPU-heavy scenario evaluation can be
    parallelized in spawned workers when GPU is not being used for that stage.
  * CUDA is preferred automatically when available.

The public ``reproduce`` stage is the canonical reviewer workflow. It is resumable,
fail-fast, records environment/package provenance, and assembles a checksummed
``reproducibility_v1_1_0`` bundle containing publication figures, tables, source CSVs,
logs, manifests, and trained-checkpoint checksums.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import tarfile
import textwrap
import time
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Windows-safe threaded CPU fallback. CUDA stages stay in the parent process; we do not
# launch competing CUDA contexts from spawned workers. Override with CEUS_CPU_THREADS.
_CPU_THREADS = int(os.environ.get("CEUS_CPU_THREADS", max(1, min(16, (os.cpu_count() or 4) - 1))))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

try:
    import requests
except Exception as exc:
    raise RuntimeError("Missing dependency 'requests'. Install the provided requirements file.") from exc

# Heavy optional imports are loaded lazily by require_* helpers.

SCRIPT_VERSION = "1.1.0-B"
CONFIG_SCHEMA_VERSION = "2026-09-01"
DATA_ROOT_NAME = "CEUS_LILLE_DATA"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CANONICAL_SEEDS = [
    101, 202, 303, 404, 505,
    606, 707, 808, 909, 1001,
]
DEFAULT_SEEDS = list(CANONICAL_SEEDS)

REFERENCE_TORCH_VERSION = "2.12.0+cu126"
REFERENCE_CUDA_AVAILABLE = True
REFERENCE_GPU_NAME = "NVIDIA GeForce RTX 3060 Laptop GPU"
REPRO_BUNDLE_NAME = "reproducibility_v1_1_0"


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": CONFIG_SCHEMA_VERSION,
    "script_version": SCRIPT_VERSION,
    "project": {
        "name": "CEUS Lille Cascading Accessibility Risk",
        "study_area": "Métropole Européenne de Lille, France",
        "crs_metric": "EPSG:2154",
        "crs_filosofi": "EPSG:3035",
        "retrieval_date": None,
    },
    "compute": {
        "prefer_cuda": True,
        "cuda_device": 0,
        "torch_float": "float32",
        "deterministic_torch": False,
        "cpu_workers": max(1, min(16, (os.cpu_count() or 4) - 1)),
        "windows_spawn": True,
        "seeds": DEFAULT_SEEDS,
        # Frozen publication design: exactly ten prespecified numerical seeds.
        "numerical_seed_limit": 10,
    },
    "data": {
        "overwrite_downloads": False,
        "user_agent": "CEUS-Lille-Accessibility-Risk/1.0 research contact=local",
        "datagouv_api": "https://www.data.gouv.fr/api/1",
        "filosofi_dataset_slug": "revenus-pauvrete-et-niveau-de-vie-donnees-carroyees-2019-et-2021-dispositif-fichier-localise-social-et-fiscal-filosofi",
        "filosofi_year": 2021,
        "hazard_dataset_query": "Aléa débordement de cours d’eau TRI Lille",
        "hazard_dataset_title": "Aléa débordement de cours d’eau - TRI Lille",
        "hazard_tri_id": "59DREAL20140002",
        "hazard_national_dataset_query": "Territoire à risque d’inondation TRI du SIG Directive inondation France métropolitaine rapportage 2020",
        "hazard_wms_base_url": "https://georisques.gouv.fr/services/di_fxx_2020",
        "hazard_wms_layers": [
            "ALEA_SYNT_01_01FOR_FXX",
            "ALEA_SYNT_01_02MOY_FXX",
            "ALEA_SYNT_01_04FAI_FXX"
        ],
        "hazard_fallback_proxy": False,
        "osm_network_type": "drive",
        "osm_service_tags": {
            "amenity": ["hospital", "clinic"],
            "healthcare": ["hospital", "clinic", "emergency_ward"]
        },
        "osm_acquisition_order": ["cache", "geofabrik", "overpass"],
        "geofabrik_pbf_url": "https://download.geofabrik.de/europe/france/nord-pas-de-calais-latest.osm.pbf",
        "geofabrik_pbf_filename": "nord-pas-de-calais-latest.osm.pbf",
        "overpass_endpoints": [
            "https://overpass-api.de/api",
            "https://overpass.kumi.systems/api",
            "https://overpass.private.coffee/api"
        ],
        "overpass_attempts_per_endpoint": 1,
        "overpass_timeout_seconds": 75,
        "overpass_retry_pause_seconds": 3,
        "max_service_facilities": 40,
    },
    "spatial": {
        "population_column_candidates": ["ind", "Ind", "population", "Population", "men", "MEN"],
        "income_column_candidates": ["med21", "MED21", "med", "MED", "niveau_vie_median", "revenu_median", "nivvie_median"],
        "poverty_column_candidates": ["tp60", "TP60", "taux_pauvrete", "poverty_rate", "part_men_pauv"],
        "vulnerability_rule": "bottom_income_quartile",
        "vulnerable_quantile": 0.25,
        "aggregate_cells_to_snapped_node": True,
        "min_population": 1.0,
        "max_model_zones": 1800,
        "zone_reduction": "population_weighted_stratified",
        "service_supply": "unit",
        "snap_max_m": 3000.0,
    },
    "network": {
        "default_speed_kph": {
            "motorway": 100, "trunk": 80, "primary": 50, "secondary": 50,
            "tertiary": 40, "residential": 30, "unclassified": 30,
            "living_street": 20, "service": 20
        },
        "default_lanes": {
            "motorway": 2.5, "trunk": 2.0, "primary": 1.5, "secondary": 1.25,
            "tertiary": 1.0, "residential": 1.0, "unclassified": 1.0,
            "living_street": 1.0, "service": 1.0
        },
        "capacity_per_lane_vph": 900.0,
        "reference_demand_scale_vph_per_person": 0.08,
        "bpr_alpha": 0.15,
        "bpr_beta": 4.0,
        "routing_iterations": 6,
        "routing_damping": 0.45,
        "route_choice_temperature": 0.08,
        "nearest_services_per_zone": 3,
        "candidate_paths_per_od": 4,
        "k_shortest_weight": "travel_time",
        "route_progress_every": 100,
        "route_checkpoint_every": 250,
        "betweenness_sample_sources": 400,
        "betweenness_progress_every": 5,
        "betweenness_checkpoint_every": 10,
        "reference_consistency_tolerance": 1e-8,
    },
    "accessibility": {
        "impedance": "exponential",
        "kappa_calibration": "half_life_minutes",
        "half_life_minutes": 20.0,
        "gamma": 0.95,
        "signed_loss_robustness": True,
    },
    "disruptions": {
        "horizon_T": 5,
        "severity_levels": [0.25, 0.50, 0.75, 1.00],
        "spatial_extent_m": [300.0, 800.0, 1600.0],
        "profiles": {
            "short": [1.00, 0.65, 0.30, 0.00, 0.00, 0.00],
            "persistent": [1.00, 0.92, 0.80, 0.65, 0.48, 0.30],
            "recurrent": [1.00, 0.55, 0.25, 0.70, 0.40, 0.15]
        },
        "n_train": 180,
        "n_validation": 60,
        "n_test": 100,
        "n_structural_test": 80,
        "structural_holdout": {
            "type": "severity_extent_combination",
            "severity": 1.0,
            "extent_m": 1600.0
        },
    },
    "intervention": {
        "restoration_fraction_of_remaining_deficit": 0.60,
        "restoration_retention": 0.90,
        "edge_cost_scale": 1.0,
        "edge_cost_length_power": 1.0,
        "reference_budget_median_actions": 4.0,
        "budget_multiplier_baseline": 1.0,
        "budget_grid": [0.5, 1.0, 1.5, 2.0],
        "max_candidate_edges_per_step": 160,
        "candidate_screening_rule": "r3_abs_deficit_x_candidate_path_use",
        "candidate_deficit_tolerance": 1e-12,
    },
    "risk": {
        "lambda_equity": 0.50,
        "lambda_tail": 0.75,
        "cvar_alpha": 0.90,
    },
    "graph_policy": {
        "method": "episodic_policy_gradient",
        "hidden_dim": 64,
        "message_layers": 2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "epochs": 40,
        "episodes_per_epoch": 6,
        "entropy_coef": 0.01,
        "gradient_clip": 2.0,
        "validation_every": 5,
        "early_stopping_patience": 4,
        # Fixed deterministic subset used only for early stopping. Final model
        # evaluation still uses the complete validation/test scenario sets.
        "validation_scenarios_training": 12,
        # Exact receptive-field computation for the decision GNN: with two
        # message layers, two topological hops around the current candidate
        # actions contain every node/edge that can influence their embeddings.
        "local_context_hops": 2,
        "local_context_cache_size": 512,
        "capacity_weighted_message_robustness": True,
    },
    "graph_ppo": {
        "enabled": True,
        "learning_rate": 3e-4,
        "clip_epsilon": 0.20,
        "update_epochs": 4,
        "entropy_coef": 0.01,
        "gradient_clip": 2.0,
        "epochs": 40,
        "episodes_per_epoch": 6,
        "validation_every": 5,
        "early_stopping_patience": 4,
        "validation_scenarios_training": 12
    },
    "baselines": {
        "random_repetitions": 20,
        "b4_search_candidates": 250,
        "b4_search_elite_fraction": 0.10,
        "b4_search_rounds": 8,
    },
    "statistics": {
        "bootstrap_resamples": 10000,
        "confidence_level": 0.95,
        "holm_adjust": True,
        "wilcoxon_robustness": True,
        "signflip_resamples": 100000,
        "rank_overlap_fracs": [0.05, 0.10, 0.20],
    },
    "output": {
        "save_parquet": True,
        "save_csv": True,
        "save_geojson": True,
        "save_models": True,
        "make_figures": True,
        "make_tables": True,
    },
    "progress": {
        "baseline_scenario_every": 5,
        "baseline_checkpoint_every": 5,
        "training_car_every": 5,
        "training_car_checkpoint_every": 5,
        "b3_edge_every": 5,
        "evaluation_scenario_every": 10,
    }
}


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    def rec(dst, src):
        for k, v in src.items():
            if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
                rec(dst[k], v)
            else:
                dst[k] = v
    rec(out, override)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass



def active_numerical_seeds(cfg: Dict[str, Any]) -> List[int]:
    """Return the frozen ten-seed numerical design used in the paper."""
    seeds = [int(x) for x in cfg["compute"]["seeds"]]
    limit = int(cfg["compute"].get("numerical_seed_limit", 10))
    active = seeds[:limit]
    if limit != 10 or active != CANONICAL_SEEDS:
        raise RuntimeError(
            "Numerical design drift detected. The public v1.1.0 protocol requires "
            f"exactly {CANONICAL_SEEDS}, got limit={limit}, active={active}."
        )
    return active

def require_geospatial():
    try:
        import geopandas as gpd
        import shapely
        import osmnx as ox
        return gpd, shapely, ox
    except Exception as exc:
        raise RuntimeError(
            "Geospatial dependencies missing. Install geopandas, shapely, pyproj, rtree and osmnx."
        ) from exc


def union_geometry(gdf):
    """GeoPandas/Shapely compatibility helper without deprecated unary_union warnings."""
    geom = gdf.geometry
    if hasattr(geom, "union_all"):
        return geom.union_all()
    return geom.unary_union


def require_network():
    try:
        import networkx as nx
        import scipy
        return nx, scipy
    except Exception as exc:
        raise RuntimeError("Missing networkx/scipy dependencies.") from exc


def require_torch():
    try:
        import torch
        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for the graph-policy stage.") from exc


@dataclasses.dataclass
class Paths:
    root: Path
    raw: Path
    processed: Path
    scenarios: Path
    models: Path
    eval: Path
    figure_data: Path
    figures: Path
    table_data: Path
    tables: Path
    manifests: Path
    logs: Path
    config: Path

    @classmethod
    def build(cls, root: Path) -> "Paths":
        root = root.resolve()
        return cls(
            root=root,
            raw=root / "data_raw",
            processed=root / "data_processed",
            scenarios=root / "scenarios",
            models=root / "models",
            eval=root / "evaluation",
            figure_data=root / "figure_data",
            figures=root / "figures",
            table_data=root / "table_data",
            tables=root / "tables",
            manifests=root / "manifests",
            logs=root / "logs",
            config=root / "ceus_lille_params.json",
        )

    def mkdirs(self):
        for p in dataclasses.asdict(self).values():
            p = Path(p)
            (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# -----------------------------------------------------------------------------
# Provenance and data download
# -----------------------------------------------------------------------------

def package_versions() -> Dict[str, str]:
    pkgs = ["numpy", "pandas", "scipy", "networkx", "geopandas", "shapely", "pyproj", "osmnx", "torch", "matplotlib"]
    out = {}
    try:
        import importlib.metadata as md
        for p in pkgs:
            with contextlib.suppress(Exception):
                out[p] = md.version(p)
    except Exception:
        pass
    return out


def datagouv_get_dataset(api: str, slug: str, session: requests.Session) -> Dict[str, Any]:
    url = f"{api.rstrip('/')}/datasets/{slug}/"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _norm_title(value: Any) -> str:
    """Accent/punctuation-insensitive normalization for dataset-title matching."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def datagouv_search(
    api: str,
    query: str,
    session: requests.Session,
    page_size: int = 50,
    exact_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Search data.gouv.fr, with strict title matching when requested.

    The previous implementation ranked by token overlap and update date. For
    near-duplicate datasets such as TRI Lille / TRI Lens, that can select the
    wrong territory. Hazard acquisition must therefore use an exact normalized
    title match and fail if that exact dataset is absent.
    """
    url = f"{api.rstrip('/')}/datasets/"
    r = session.get(url, params={"q": query, "page_size": page_size}, timeout=60)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError(f"No data.gouv.fr dataset found for query: {query}")

    if exact_title:
        target = _norm_title(exact_title)
        exact = [d for d in rows if _norm_title(d.get("title")) == target]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return max(exact, key=lambda d: d.get("last_update", "") or "")
        available = [str(d.get("title", "")) for d in rows[:20]]
        raise RuntimeError(
            "Exact data.gouv.fr dataset title not found. "
            f"Expected={exact_title!r}; returned titles={available!r}"
        )

    qnorm = _norm_title(query)
    qtokens = set(qnorm.split())

    def score(d):
        title = _norm_title(d.get("title", ""))
        ttokens = set(title.split())
        overlap = len(qtokens & ttokens)
        # Penalize explicit wrong-TRI names when the query names a TRI.
        penalty = 0
        if "lille" in qtokens and "lille" not in ttokens:
            penalty -= 100
        return (penalty + overlap, d.get("last_update", "") or "")

    return max(rows, key=score)


def choose_resource(dataset: Dict[str, Any], preferred_exts: Sequence[str], name_hints: Sequence[str] = ()) -> Dict[str, Any]:
    resources = dataset.get("resources", [])
    if not resources:
        raise RuntimeError(f"Dataset '{dataset.get('title')}' has no downloadable resources.")
    ext_rank = {e.lower().lstrip("."): i for i, e in enumerate(preferred_exts)}
    hints = [h.lower() for h in name_hints]
    def res_score(r):
        url = str(r.get("url") or "")
        title = str(r.get("title") or "") + " " + str(r.get("description") or "")
        fmt = str(r.get("format") or "").lower().lstrip(".")
        suffix = Path(url.split("?")[0]).suffix.lower().lstrip(".")
        ext = fmt or suffix
        er = ext_rank.get(ext, 999)
        hint_score = sum(h in title.lower() or h in url.lower() for h in hints)
        size = r.get("filesize") or r.get("file_size") or 0
        try:
            size = int(size)
        except Exception:
            size = 0
        return (hint_score, -er, size)
    preferred = []
    for r in resources:
        url = str(r.get("url") or "").split("?")[0]
        fmt = str(r.get("format") or "").lower().lstrip(".") or Path(url).suffix.lower().lstrip(".")
        if fmt in ext_rank:
            preferred.append(r)
    ranked = sorted(preferred or resources, key=res_score, reverse=True)
    best = ranked[0]
    url = best.get("url")
    if not url:
        raise RuntimeError("Chosen data.gouv resource has no URL.")
    return best


def download_file(url: str, dest: Path, session: requests.Session, overwrite: bool = False, logger: Optional[Logger] = None) -> Path:
    if dest.exists() and not overwrite:
        if logger: logger.log(f"Reuse cached {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if logger: logger.log(f"Download {url}")
    with session.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return dest


def infer_extension(resource: Dict[str, Any]) -> str:
    fmt = str(resource.get("format") or "").strip().lower().lstrip(".")
    if fmt in {"parquet", "csv", "zip", "gpkg", "geojson", "json", "shp"}:
        return "." + fmt
    url = str(resource.get("url") or "").split("?")[0]
    suf = Path(url).suffix
    return suf if suf else ".bin"


def _rank_vector_candidate(path: Path) -> Tuple[int, int, str]:
    """Prefer likely hazard vector datasets over generic metadata files."""
    name = path.name.lower()
    ext_rank = {
        ".gpkg": 0, ".shp": 1, ".gml": 2, ".geojson": 3,
        ".json": 4, ".kml": 5, ".sqlite": 6,
    }.get(path.suffix.lower(), 20)
    hazard_tokens = ("alea", "aléa", "inond", "flood", "tri", "debord", "débord", "lille")
    token_score = -sum(tok in name for tok in hazard_tokens)
    return (ext_rank, token_score, name)


def _is_readable_vector_dataset(path: Path) -> bool:
    """Probe candidate files with GDAL/OGR through pyogrio."""
    if not path.is_file():
        return False
    if path.suffix.lower() in {".dbf", ".shx", ".prj", ".cpg", ".qmd", ".sbn", ".sbx"}:
        return False
    try:
        import pyogrio
        layers = pyogrio.list_layers(path)
        return layers is not None and len(layers) > 0
    except Exception:
        return False


def _find_extracted_vector(out_dir: Path, log: Optional[Logger] = None) -> Optional[Path]:
    """Find the best actual OGR-readable vector layer in an extracted archive."""
    likely: List[Path] = []
    for ext in ("*.gpkg", "*.shp", "*.gml", "*.geojson", "*.json", "*.kml", "*.sqlite"):
        likely.extend(p for p in out_dir.rglob(ext) if p.is_file())
    likely = sorted(set(likely), key=_rank_vector_candidate)
    for p in likely:
        if _is_readable_vector_dataset(p):
            if log:
                log.log(f"Hazard vector layer resolved: {p.name}")
            return p

    all_files = [p for p in out_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    for p in sorted(all_files, key=lambda p: (_rank_vector_candidate(p), -p.stat().st_size)):
        if _is_readable_vector_dataset(p):
            if log:
                log.log(f"Hazard vector layer resolved by GDAL probe: {p.name}")
            return p

    if log:
        listing = "; ".join(
            f"{p.relative_to(out_dir)} ({p.stat().st_size} bytes)"
            for p in all_files[:40]
        )
        log.log("No OGR-readable vector dataset found after extraction. Archive contents: " + (listing or "<empty>"))
    return None


def find_vector_in_archive(archive_path: Path, out_dir: Path, log: Optional[Logger] = None) -> Optional[Path]:
    """Extract ZIP/TAR archives irrespective of their filename extension.

    Geo-IDE ATOM download URLs often return a ZIP payload from an extensionless
    endpoint. data.gouv metadata may therefore lead us to save the payload as
    ``.bin`` even though its bytes are a valid ZIP archive. Detection must be
    content-based, not suffix-based.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
    else:
        return None

    # Some public-data archives contain nested ZIPs.
    nested_archives = list(out_dir.rglob("*.zip"))
    for nested in nested_archives:
        nested_dir = nested.with_suffix("")
        nested_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(nested, "r") as zf:
                zf.extractall(nested_dir)
        except Exception:
            pass

    return _find_extracted_vector(out_dir, log)


def resolve_vector_payload(download_path: Path, out_dir: Path, log: Optional[Logger] = None) -> Optional[Path]:
    """Resolve a downloaded Geo-IDE payload to a readable vector file.

    Resolution order:
      1. ZIP/TAR signature, regardless of extension;
      2. known vector suffix;
      3. content sniffing for GeoPackage/GeoJSON saved as ``.bin``.
    """
    # 1) Archive-by-signature. This is the expected Geo-IDE ATOM case.
    if zipfile.is_zipfile(download_path) or tarfile.is_tarfile(download_path):
        if log:
            log.log(
                f"Hazard payload {download_path.name} detected as archive by file signature; extracting"
            )
        return find_vector_in_archive(download_path, out_dir, log)

    # 2) Explicit vector extension.
    if download_path.suffix.lower() in {".gpkg", ".geojson", ".json", ".shp", ".gml", ".kml", ".sqlite"}:
        return download_path

    # 3) Sniff common extensionless vector payloads.
    head = download_path.read_bytes()[:4096]
    stripped = head.lstrip()

    # GeoPackage is SQLite with mandatory GPKG metadata tables.
    if head.startswith(b"SQLite format 3"):
        resolved = out_dir / "tri_lille_hazard.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download_path, resolved)
        if log:
            log.log(f"Hazard payload detected as SQLite/GeoPackage; copied to {resolved.name}")
        return resolved

    # GeoJSON / JSON feature collection.
    if stripped.startswith((b"{", b"[")):
        resolved = out_dir / "tri_lille_hazard.geojson"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download_path, resolved)
        if log:
            log.log(f"Hazard payload detected as JSON/GeoJSON; copied to {resolved.name}")
        return resolved

    # XML/HTML is usually an ATOM metadata/error response, not the vector itself.
    lower = stripped[:512].lower()
    if lower.startswith(b"<?xml") or lower.startswith(b"<feed") or b"<html" in lower:
        if log:
            log.log(
                "Hazard payload appears to be XML/HTML rather than a vector archive."
            )
        return None

    return None


def extract_official_download_urls_from_xml(xml_path: Path) -> List[str]:
    """Extract HTTP(S) linkage URLs from ISO/INSPIRE metadata XML."""
    import xml.etree.ElementTree as ET
    urls: List[str] = []
    try:
        root = ET.parse(xml_path).getroot()
        for elem in root.iter():
            txt = (elem.text or "").strip()
            if txt.startswith(("http://", "https://")):
                urls.append(txt)
            for val in elem.attrib.values():
                sval = str(val).strip()
                if sval.startswith(("http://", "https://")):
                    urls.append(sval)
    except Exception:
        raw = xml_path.read_text(encoding="utf-8", errors="ignore")
        urls.extend(re.findall(r'https?://[^\s"\'<>]+', raw))

    # HTML/XML entities can survive parsing in text nodes.
    import html
    out: List[str] = []
    for u in urls:
        u = html.unescape(u).rstrip(".,);]")
        if u not in out:
            out.append(u)
    return out


def resolve_vector_from_metadata_archive(
    unpack_dir: Path,
    session: requests.Session,
    log: Optional[Logger] = None,
) -> Optional[Path]:
    """Follow official data links embedded in a metadata-only Geo-IDE archive.

    Geo-IDE's dataType=dataset endpoint can return a ZIP containing only ISO
    metadata XML. In that case, the XML contains distribution/onLine resources.
    We test only official government URLs and accept a result only if its payload
    resolves to an actual OGR-readable vector dataset.
    """
    xml_files = [p for p in unpack_dir.rglob("*.xml") if p.is_file()]
    if not xml_files:
        return None

    urls: List[str] = []
    for xp in xml_files:
        urls.extend(extract_official_download_urls_from_xml(xp))

    allowed_hosts = (
        "developpement-durable.gouv.fr",
        "geo-ide",
        "georisques.gouv.fr",
    )
    # Prioritize direct resource / WFS / downloadable vector links.
    def rank(u: str):
        low = u.lower()
        score = 0
        for token in ("getresource", "download", "service=wfs", "request=getfeature",
                      ".zip", ".gpkg", ".shp", ".gml", ".geojson"):
            if token in low:
                score += 1
        return (-score, len(u))

    candidates = []
    for u in urls:
        low = u.lower()
        if any(host in low for host in allowed_hosts):
            if any(tok in low for tok in (
                "getresource", "download", "service=wfs", "request=getfeature",
                ".zip", ".gpkg", ".shp", ".gml", ".geojson"
            )):
                candidates.append(u)
    candidates = sorted(dict.fromkeys(candidates), key=rank)

    if log:
        log.log(
            f"Metadata-only hazard archive detected; found "
            f"{len(candidates)} plausible official distribution link(s)"
        )

    for idx, url in enumerate(candidates[:20]):
        try:
            r = session.get(url, stream=True, timeout=(30, 180), allow_redirects=True)
            r.raise_for_status()
            tmp = unpack_dir / f"metadata_link_{idx:02d}.bin"
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            if tmp.stat().st_size == 0:
                continue

            resolved = resolve_vector_payload(
                tmp,
                unpack_dir / f"metadata_link_{idx:02d}_unpacked",
                log,
            )
            if resolved is not None and _is_readable_vector_dataset(resolved):
                if log:
                    log.log(f"Hazard vector resolved from metadata distribution link: {url}")
                return resolved
        except Exception as exc:
            if log:
                log.log(
                    f"Skip metadata distribution link {idx+1}: "
                    f"{type(exc).__name__}: {exc}"
                )

    return None



def _hazard_filename_score(path: Path) -> Tuple[int, int, int, str]:
    """Rank extracted TRI files toward actual flood-surface layers."""
    n = _norm_title(path.name)
    surface = int("surface" in n and ("inond" in n or "flood" in n))
    alea = int("alea" in n or "hazard" in n)
    tri = int("tri" in n)
    # Prefer flood surface, then hazard, then generic TRI.
    return (-surface, -alea, -tri, path.name.lower())


def resolve_best_hazard_vector_from_archive(
    archive_path: Path,
    out_dir: Path,
    log: Optional[Logger] = None,
) -> Optional[Path]:
    """Extract an official archive and choose the most relevant readable flood layer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
    else:
        return resolve_vector_payload(archive_path, out_dir, log)

    files = [p for p in out_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    readable = [p for p in files if _is_readable_vector_dataset(p)]
    if not readable:
        return None
    readable = sorted(readable, key=_hazard_filename_score)
    chosen = readable[0]
    if log:
        log.log(f"National TRI fallback resolved hazard candidate: {chosen.name}")
    return chosen



def _wms_capability_layer_names(capabilities_xml: bytes) -> List[str]:
    """Extract advertised WMS layer names."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(capabilities_xml)
    out: List[str] = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "Layer":
            for child in elem:
                if child.tag.split("}")[-1] == "Name" and child.text:
                    out.append(child.text.strip())
                    break
    return out


def _png_hazard_mask(content: bytes) -> np.ndarray:
    """Convert a transparent WMS PNG into a boolean hazard-support mask."""
    from PIL import Image
    img = Image.open(io.BytesIO(content)).convert("RGBA")
    arr = np.asarray(img)
    alpha = arr[..., 3]
    rgb = arr[..., :3]

    # Preferred case: transparent background.
    if np.any(alpha < 255):
        mask = alpha > 0
    else:
        # Defensive fallback for servers ignoring TRANSPARENT=TRUE.
        # Use white background requested explicitly below.
        mask = np.any(rgb < 248, axis=2)

    # Exclude fully white/near-white pixels even when alpha is opaque.
    mask &= np.any(rgb < 248, axis=2)
    return mask


def _polygonize_mask(mask: np.ndarray, bounds: Tuple[float, float, float, float]):
    """Polygonize a binary WMS mask in EPSG:4326."""
    try:
        import rasterio.features
        import rasterio.transform
    except Exception as exc:
        raise RuntimeError(
            "Rasterio is required to polygonize the official Géorisques WMS hazard "
            "surface. Install rasterio in the geospatial environment."
        ) from exc

    from shapely.geometry import shape
    minx, miny, maxx, maxy = bounds
    h, w = mask.shape
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)

    geoms = []
    data = mask.astype(np.uint8)
    for geom, value in rasterio.features.shapes(data, mask=mask, transform=transform):
        if int(value) == 1:
            g = shape(geom)
            if not g.is_empty and g.area > 0:
                geoms.append(g)
    return geoms


def download_georisques_wms_hazard(
    cfg: Dict[str, Any],
    paths: Paths,
    sess: requests.Session,
    overwrite: bool,
    log: Logger,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Acquire official TRI flood surfaces from the Géorisques WMS.

    The ALEA_SYNT_* layers are WMS raster layers, not WFS feature layers.
    We therefore request transparent high-resolution map rasters over MEL and
    polygonize their non-transparent hazard support. At the MEL extent, the
    default 2048x2048 request yields an effective pixel size of roughly 15--20 m,
    which is adequate for edge-exposure screening while preserving provenance.
    """
    gpd, _, _ = require_geospatial()
    base = cfg["data"].get(
        "hazard_wms_base_url",
        "https://georisques.gouv.fr/services/di_fxx_2020",
    )
    desired_layers = list(cfg["data"].get("hazard_wms_layers", [
        "ALEA_SYNT_01_01FOR_FXX",
        "ALEA_SYNT_01_02MOY_FXX",
        "ALEA_SYNT_01_04FAI_FXX",
    ]))

    out_path = paths.raw / "georisques_tri_lille_fluvial_surfaces.geojson"
    if out_path.exists() and not overwrite:
        try:
            chk = gpd.read_file(out_path)
            if not chk.empty:
                log.log(f"Reuse cached official Géorisques WMS hazard: {out_path.name}")
                return out_path, {
                    "provider": "Géorisques WMS / SIG Directive inondation - Rapportage 2020",
                    "service_url": base,
                    "vector_path": str(out_path),
                    "proxy": False,
                    "layers": desired_layers,
                    "tri_id": cfg["data"].get("hazard_tri_id", "59DREAL20140002"),
                    "sha256": sha256_file(out_path),
                    "derivation": "transparent WMS raster polygonization",
                }
        except Exception:
            pass

    log.log("Query official Géorisques WMS GetCapabilities")
    cap = sess.get(
        base,
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
        timeout=(30, 120),
    )
    cap.raise_for_status()
    available = _wms_capability_layer_names(cap.content)
    missing = [x for x in desired_layers if x not in available]
    if missing:
        raise RuntimeError(
            f"Required Géorisques WMS layer(s) absent: {missing!r}. "
            f"Advertised layers include: {available[:80]!r}"
        )

    boundary = gpd.read_file(paths.raw / "mel_boundary.geojson").to_crs(4326)
    minx, miny, maxx, maxy = map(float, boundary.total_bounds)
    mel_geom = union_geometry(boundary)
    width = height = 2048

    rows = []
    layer_manifest = []

    for layer in desired_layers:
        content = None
        last_error = None

        # WMS 1.3.0 EPSG:4326 uses latitude/longitude axis order. CRS:84 uses
        # longitude/latitude and is less ambiguous. Try CRS:84 first, then
        # EPSG:4326, then WMS 1.1.1.
        attempts = [
            ("1.3.0", "CRS:84", f"{minx},{miny},{maxx},{maxy}", "crs"),
            ("1.3.0", "EPSG:4326", f"{miny},{minx},{maxy},{maxx}", "crs"),
            ("1.1.1", "EPSG:4326", f"{minx},{miny},{maxx},{maxy}", "srs"),
        ]

        used = None
        for version, crs_value, bbox, crs_key in attempts:
            params = {
                "service": "WMS",
                "version": version,
                "request": "GetMap",
                "layers": layer,
                "styles": "",
                crs_key: crs_value,
                "bbox": bbox,
                "width": width,
                "height": height,
                "format": "image/png",
                "transparent": "TRUE",
                "bgcolor": "0xFFFFFF",
                "exceptions": "application/vnd.ogc.se_xml",
            }
            try:
                r = sess.get(base, params=params, timeout=(30, 300))
                r.raise_for_status()
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "image" not in ctype and not r.content.startswith(b"\x89PNG"):
                    raise RuntimeError(
                        f"Expected PNG from WMS GetMap, got Content-Type={ctype!r}, "
                        f"head={r.content[:120]!r}"
                    )
                mask = _png_hazard_mask(r.content)
                if mask.any():
                    content = r.content
                    used = (version, crs_value)
                    break
                last_error = RuntimeError("WMS PNG contained no non-background hazard pixels.")
            except Exception as exc:
                last_error = exc

        if content is None:
            raise RuntimeError(
                f"Géorisques WMS returned no MEL hazard pixels for layer {layer!r}. "
                f"Last error={last_error}"
            )

        mask = _png_hazard_mask(content)
        geoms = _polygonize_mask(mask, (minx, miny, maxx, maxy))
        if not geoms:
            raise RuntimeError(f"Polygonization produced no geometry for WMS layer {layer!r}.")

        if layer.endswith("01FOR_FXX"):
            pclass, rp = "frequent_decennial", 10
        elif layer.endswith("02MOY_FXX"):
            pclass, rp = "mean_centennial", 100
        elif layer.endswith("04FAI_FXX"):
            pclass, rp = "rare_millennial", 1000
        else:
            pclass, rp = "other", np.nan

        kept = 0
        for g in geoms:
            if g.intersects(mel_geom):
                gg = g.intersection(mel_geom)
                if not gg.is_empty:
                    rows.append({
                        "hazard_layer": layer,
                        "hazard_probability_class": pclass,
                        "hazard_return_period_years": rp,
                        "geometry": gg,
                    })
                    kept += 1

        if kept == 0:
            raise RuntimeError(f"WMS layer {layer!r} has no polygon intersecting MEL.")

        layer_manifest.append({
            "layer": layer,
            "n_polygons_mel": kept,
            "wms_version": used[0],
            "crs": used[1],
            "width": width,
            "height": height,
        })
        log.log(
            f"Géorisques WMS layer {layer}: {kept:,} MEL hazard polygon(s) "
            f"from {width}x{height} raster"
        )

    combined = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    combined = combined.to_crs(cfg["project"]["crs_metric"])
    combined.to_file(out_path, driver="GeoJSON")

    info = {
        "provider": "Géorisques WMS / SIG Directive inondation - Rapportage 2020",
        "service_url": base,
        "vector_path": str(out_path),
        "proxy": False,
        "layers": layer_manifest,
        "tri_id": cfg["data"].get("hazard_tri_id", "59DREAL20140002"),
        "hazard_type": "fluvial_overflow_surface",
        "probability_classes": [
            "frequent_decennial",
            "mean_centennial",
            "rare_millennial",
        ],
        "derivation": "official WMS raster support polygonized at 2048x2048 over MEL",
        "sha256": sha256_file(out_path),
    }
    return out_path, info


def download_georisques_national_tri_fallback(
    cfg: Dict[str, Any],
    paths: Paths,
    sess: requests.Session,
    overwrite: bool,
    log: Logger,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Compatibility wrapper: use official Géorisques WFS vector service."""
    return download_georisques_wms_hazard(cfg, paths, sess, overwrite, log)


def filter_hazard_layer_to_tri_lille(
    hz,
    tri_id: str,
    log: Optional[Logger] = None,
):
    """Filter a national/regional TRI layer to Lille when identifying attributes exist."""
    if hz.empty:
        return hz

    target_id = _norm_title(tri_id)
    target_name = "lille"
    candidate_cols = []
    for c in hz.columns:
        if c == hz.geometry.name:
            continue
        lc = _norm_title(c)
        if any(tok in lc for tok in ("tri", "id", "code", "nom", "name", "libelle", "territ")):
            candidate_cols.append(c)

    matched = None
    matched_col = None
    for c in candidate_cols:
        s = hz[c].astype(str).map(_norm_title)
        mask = s.str.contains(target_id, regex=False, na=False) | s.str.contains(target_name, regex=False, na=False)
        if mask.any():
            matched = hz.loc[mask].copy()
            matched_col = c
            break

    if matched is not None and not matched.empty:
        if log:
            log.log(
                f"Hazard layer filtered to TRI Lille using column '{matched_col}': "
                f"{len(matched):,}/{len(hz):,} feature(s)"
            )
        return matched

    # If the layer has no identifying field (common for a department-specific
    # surface layer), do not silently discard it. Spatial clipping to MEL happens
    # later through edge intersection.
    if log:
        log.log(
            "Hazard layer has no usable TRI Lille identifier field; retaining the "
            "official layer and relying on MEL spatial intersection."
        )
    return hz


def require_pyrosm():
    try:
        from pyrosm import OSM
        return OSM
    except Exception as exc:
        raise RuntimeError(
            "Local Geofabrik PBF parsing requires 'pyrosm'. "
            "Install it in this environment with: pip install pyrosm"
        ) from exc


def ensure_geofabrik_pbf(
    cfg: Dict[str, Any],
    paths: Paths,
    sess: requests.Session,
    log: Logger,
) -> Path:
    """Download once and cache the regional OSM PBF used for both POIs and roads."""
    url = str(cfg["data"]["geofabrik_pbf_url"])
    name = str(cfg["data"].get(
        "geofabrik_pbf_filename",
        "nord-pas-de-calais-latest.osm.pbf",
    ))
    dest = paths.raw / name
    if dest.exists() and dest.stat().st_size > 10_000_000:
        log.log(
            f"Reuse cached Geofabrik OSM PBF: {dest.name} "
            f"({dest.stat().st_size / 1024**2:.1f} MB)"
        )
        return dest

    log.log(
        "Download Geofabrik Nord-Pas-de-Calais OSM PBF once "
        f"for reproducible local OSM extraction: {url}"
    )
    download_file(url, dest, sess, overwrite=False, logger=log)
    if not dest.exists() or dest.stat().st_size <= 10_000_000:
        raise RuntimeError(
            f"Downloaded Geofabrik PBF is unexpectedly small: {dest}"
        )
    log.log(
        f"Geofabrik OSM PBF cached: {dest.name} "
        f"({dest.stat().st_size / 1024**2:.1f} MB, sha256={sha256_file(dest)[:16]}...)"
    )
    return dest


def geofabrik_health_services(
    cfg: Dict[str, Any],
    pbf_path: Path,
    polygon,
    log: Logger,
):
    """Extract hospital/clinic/emergency POIs locally from the cached PBF."""
    gpd, _, _ = require_geospatial()
    OSM = require_pyrosm()
    minx, miny, maxx, maxy = map(float, polygon.bounds)

    t0 = time.perf_counter()
    log.log(
        f"Parse health facilities locally from Geofabrik PBF "
        f"within MEL bbox [{minx:.4f},{miny:.4f},{maxx:.4f},{maxy:.4f}]"
    )
    osm = OSM(str(pbf_path), bounding_box=[minx, miny, maxx, maxy])

    # Pyrosm can filter POIs by OSM tags. Use the union query and then enforce
    # the exact study definitions explicitly below.
    pois = osm.get_pois(
        custom_filter={
            "amenity": ["hospital", "clinic"],
            "healthcare": ["hospital", "clinic", "emergency_ward"],
        }
    )
    if pois is None or len(pois) == 0:
        raise RuntimeError("Geofabrik/Pyrosm returned no candidate health POIs in MEL bbox.")

    if pois.crs is None:
        pois = pois.set_crs(4326)
    else:
        pois = pois.to_crs(4326)

    amenity = (
        pois["amenity"].astype(str).str.lower()
        if "amenity" in pois.columns else pd.Series("", index=pois.index)
    )
    healthcare = (
        pois["healthcare"].astype(str).str.lower()
        if "healthcare" in pois.columns else pd.Series("", index=pois.index)
    )
    keep = (
        amenity.isin(["hospital", "clinic"])
        | healthcare.isin(["hospital", "clinic", "emergency_ward"])
    )
    pois = pois.loc[keep].copy()
    pois = pois[
        pois.geometry.notna()
        & ~pois.geometry.is_empty
        & pois.geometry.intersects(polygon)
    ].copy()
    if pois.empty:
        raise RuntimeError("No Geofabrik health POI remains after exact MEL spatial filtering.")

    log.log(
        f"Geofabrik local health extraction complete: {len(pois):,} raw feature(s) "
        f"in {time.perf_counter()-t0:.1f}s"
    )
    return pois


def geofabrik_drive_graph(
    cfg: Dict[str, Any],
    pbf_path: Path,
    polygon,
    log: Logger,
):
    """Build the directed MEL driving graph locally from the same cached OSM PBF."""
    _, _, ox = require_geospatial()
    nx, _ = require_network()
    OSM = require_pyrosm()

    minx, miny, maxx, maxy = map(float, polygon.bounds)
    t0 = time.perf_counter()
    log.log(
        f"Parse directed driving network locally from Geofabrik PBF "
        f"within MEL bbox [{minx:.4f},{miny:.4f},{maxx:.4f},{maxy:.4f}]"
    )
    osm = OSM(str(pbf_path), bounding_box=[minx, miny, maxx, maxy])
    nodes, edges = osm.get_network(nodes=True, network_type="driving")
    if nodes is None or edges is None or len(nodes) == 0 or len(edges) == 0:
        raise RuntimeError("Geofabrik/Pyrosm returned an empty driving network.")

    G = osm.to_graph(nodes, edges, graph_type="networkx")
    if G is None or G.number_of_edges() == 0:
        raise RuntimeError("Pyrosm could not create a directed NetworkX graph.")

    # Keep only the weakly connected component with maximum size, matching the
    # former OSMnx retain_all=False behavior.
    if not nx.is_weakly_connected(G):
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    G.graph["crs"] = "EPSG:4326"

    # The bbox is intentionally slightly larger than the exact polygon; trim
    # graph nodes to MEL using OSMnx's polygon truncation when available.
    try:
        G = ox.truncate.truncate_graph_polygon(
            G, polygon, truncate_by_edge=True
        )
    except Exception as exc:
        log.log(
            f"Polygon graph trim unavailable ({type(exc).__name__}: {exc}); "
            "retain bbox graph and rely on MEL snapping/intersection."
        )

    if G.number_of_edges() == 0:
        raise RuntimeError("Driving graph became empty after MEL trimming.")

    log.log(
        f"Geofabrik local road extraction complete: "
        f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
        f"in {time.perf_counter()-t0:.1f}s"
    )
    return G


def download_osm_health_services(
    cfg: Dict[str, Any],
    polygon,
    log: Logger,
):
    """Download OSM health facilities with endpoint failover and explicit logs.

    Public Overpass instances occasionally time out under load. A single endpoint
    failure is therefore a transport-layer event, not evidence that the OSM query
    or study design is invalid. We retry a small prespecified mirror set and keep
    the first successful non-empty result.
    """
    _, _, ox = require_geospatial()

    endpoints = list(cfg["data"].get("overpass_endpoints", [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://overpass.private.coffee/api",
    ]))
    attempts = max(1, int(cfg["data"].get("overpass_attempts_per_endpoint", 2)))
    timeout = max(30, int(cfg["data"].get("overpass_timeout_seconds", 90)))
    pause = max(0.0, float(cfg["data"].get("overpass_retry_pause_seconds", 5)))

    # Preserve global OSMnx settings because this helper should not leak state
    # into the later road-network download.
    old_url = getattr(ox.settings, "overpass_url", None)
    old_timeout = getattr(ox.settings, "requests_timeout", None)

    errors = []
    try:
        if hasattr(ox.settings, "requests_timeout"):
            ox.settings.requests_timeout = timeout

        for endpoint_idx, endpoint in enumerate(endpoints, start=1):
            if hasattr(ox.settings, "overpass_url"):
                ox.settings.overpass_url = endpoint.rstrip("/")

            for attempt in range(1, attempts + 1):
                t0 = time.perf_counter()
                log.log(
                    f"OSM health facilities: Overpass endpoint "
                    f"{endpoint_idx}/{len(endpoints)} {endpoint} "
                    f"attempt {attempt}/{attempts}, timeout={timeout}s"
                )
                try:
                    feats = ox.features_from_polygon(
                        polygon,
                        cfg["data"]["osm_service_tags"],
                    )
                    elapsed = time.perf_counter() - t0
                    if feats is None or feats.empty:
                        raise RuntimeError("Overpass returned an empty health-facility result.")
                    log.log(
                        f"OSM health facilities downloaded successfully from {endpoint}: "
                        f"{len(feats):,} raw feature(s) in {elapsed:.1f}s"
                    )
                    return feats
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    msg = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    errors.append({
                        "endpoint": endpoint,
                        "attempt": attempt,
                        "elapsed_sec": round(elapsed, 3),
                        "error": msg,
                    })
                    log.log(
                        f"Overpass attempt failed after {elapsed:.1f}s: {msg}"
                    )
                    if attempt < attempts and pause > 0:
                        log.log(f"Retry Overpass in {pause:.0f}s")
                        time.sleep(pause)

        raise RuntimeError(
            "All configured Overpass endpoints failed for the OSM health-facility query. "
            f"Attempts={errors!r}"
        )
    finally:
        if hasattr(ox.settings, "overpass_url") and old_url is not None:
            ox.settings.overpass_url = old_url
        if hasattr(ox.settings, "requests_timeout") and old_timeout is not None:
            ox.settings.requests_timeout = old_timeout


def download_public_data(cfg: Dict[str, Any], paths: Paths, log: Logger) -> Dict[str, Any]:
    gpd, _, ox = require_geospatial()
    sess = requests.Session()
    sess.headers.update({"User-Agent": cfg["data"]["user_agent"]})
    overwrite = bool(cfg["data"]["overwrite_downloads"])
    api = cfg["data"]["datagouv_api"]
    manifest: Dict[str, Any] = {"created_utc": utc_now(), "sources": {}, "warnings": []}

    # 1) MEL boundary from OSM/Nominatim.
    boundary_path = paths.raw / "mel_boundary.geojson"
    if not boundary_path.exists() or overwrite:
        log.log("Resolve Métropole Européenne de Lille boundary with OSMnx/Nominatim")
        b = ox.geocode_to_gdf(cfg["project"]["study_area"])
        if b.empty:
            raise RuntimeError("OSM/Nominatim returned no boundary for MEL.")
        b.to_file(boundary_path, driver="GeoJSON")
    manifest["sources"]["study_boundary"] = {
        "provider": "OpenStreetMap/Nominatim via OSMnx",
        "query": cfg["project"]["study_area"],
        "path": str(boundary_path),
        "sha256": sha256_file(boundary_path),
    }

    # 2) Filosofi 2021 grid.
    ds = datagouv_get_dataset(api, cfg["data"]["filosofi_dataset_slug"], sess)
    res = choose_resource(ds, ["parquet", "zip", "csv"], [str(cfg["data"]["filosofi_year"]), "2021"])
    ext = infer_extension(res)
    filo_path = download_file(res["url"], paths.raw / f"filosofi_{cfg['data']['filosofi_year']}{ext}", sess, overwrite, log)
    manifest["sources"]["filosofi"] = {
        "provider": "INSEE/Filosofi via data.gouv.fr",
        "dataset_title": ds.get("title"),
        "dataset_id": ds.get("id"),
        "resource_title": res.get("title"),
        "resource_url": res.get("url"),
        "path": str(filo_path),
        "sha256": sha256_file(filo_path),
    }

    # 3) DREAL TRI Lille flood-hazard dataset.
    hazard_info = None
    try:
        hds = datagouv_search(
            api,
            cfg["data"]["hazard_dataset_query"],
            sess,
            exact_title=cfg["data"].get(
                "hazard_dataset_title",
                "Aléa débordement de cours d’eau - TRI Lille",
            ),
        )
        if "lille" not in _norm_title(hds.get("title", "")).split():
            raise RuntimeError(
                f"Hazard dataset territory mismatch: selected title={hds.get('title')!r}"
            )
        log.log(
            f"Resolved hazard dataset exactly: {hds.get('title')} "
            f"(id={hds.get('id')})"
        )
        hres = choose_resource(
            hds,
            ["gpkg", "geojson", "zip", "shp", "gml", "json"],
            ["lille", "alea", "aléa", "tri"],
        )
        hext = infer_extension(hres)
        hpath = download_file(
            hres["url"],
            paths.raw / f"tri_lille_hazard{hext}",
            sess,
            overwrite,
            log,
        )
        hazard_unpack = paths.raw / "tri_lille_hazard_unpacked"
        vector_path = resolve_vector_payload(
            hpath,
            hazard_unpack,
            log,
        )
        if vector_path is None:
            vector_path = resolve_vector_from_metadata_archive(
                hazard_unpack,
                sess,
                log,
            )
        hazard_info = {
            "provider": "DREAL Hauts-de-France via data.gouv.fr",
            "dataset_title": hds.get("title"),
            "dataset_id": hds.get("id"),
            "resource_title": hres.get("title"),
            "resource_url": hres.get("url"),
            "archive_path": str(hpath),
            "vector_path": str(vector_path) if vector_path else None,
            "sha256": sha256_file(hpath),
            "proxy": False,
        }
        if vector_path is None:
            raise RuntimeError(
                "The legacy DREAL TRI Lille resource is metadata-only/non-vector."
            )
    except Exception as dreal_exc:
        # The 2015 DREAL data.gouv record currently exposes a stale Geo-IDE
        # resource that can resolve to metadata for another TRI. Do not keep
        # retrying that broken endpoint. Fall back to the official national
        # Géorisques Directive-Inondation Rapportage 2020 GIS dataset.
        try:
            vector_path, hazard_info = download_georisques_national_tri_fallback(
                cfg, paths, sess, overwrite, log
            )
            if vector_path is None:
                raise RuntimeError(
                    "Official national TRI Rapportage 2020 archive contained no "
                    "OGR-readable vector layer."
                )
            log.log(
                "Using official Géorisques TRI Rapportage 2020 WMS for Lille "
                f"(TRI id={cfg['data'].get('hazard_tri_id', '59DREAL20140002')})"
            )
            manifest["warnings"].append(
                "Legacy DREAL TRI Lille data.gouv/Geo-IDE resource was unusable; "
                "official Géorisques Rapportage 2020 used instead."
            )
        except Exception as national_exc:
            if not cfg["data"].get("hazard_fallback_proxy", False):
                raise RuntimeError(
                    "Neither the legacy DREAL TRI Lille resource nor the official "
                    "Géorisques TRI Rapportage 2020 dataset could be resolved to a "
                    "machine-readable vector hazard layer. No proxy was substituted."
                ) from national_exc
            manifest["warnings"].append(
                f"Hazard vector unavailable; explicit OSM-waterway proxy enabled. "
                f"DREAL error={dreal_exc}; national fallback error={national_exc}"
            )
            hazard_info = {
                "provider": "OSM waterway-distance proxy",
                "proxy": True,
                "reason": f"DREAL={dreal_exc}; national={national_exc}",
            }
    manifest["sources"]["flood_hazard"] = hazard_info

    # 4) Raw OSM health features. Prefer one locally cached Geofabrik snapshot
    # for deterministic, API-independent extraction; Overpass remains fallback.
    services_path = paths.raw / "osm_health_services.geojson"
    if services_path.exists() and not overwrite:
        log.log(f"Reuse cached OSM health facilities: {services_path.name}")
    else:
        boundary = gpd.read_file(boundary_path).to_crs(4326)
        geom = union_geometry(boundary)
        log.log("Acquire essential-health facilities from OpenStreetMap")
        feats = None
        source = None
        errors = []

        order = list(cfg["data"].get(
            "osm_acquisition_order", ["cache", "geofabrik", "overpass"]
        ))
        if "geofabrik" in order:
            try:
                pbf = ensure_geofabrik_pbf(cfg, paths, sess, log)
                feats = geofabrik_health_services(cfg, pbf, geom, log)
                source = "Geofabrik PBF + Pyrosm"
            except Exception as exc:
                errors.append(f"Geofabrik: {type(exc).__name__}: {exc}")
                log.log(
                    f"Geofabrik health extraction unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )

        if feats is None and "overpass" in order:
            try:
                feats = download_osm_health_services(cfg, geom, log)
                source = "Overpass via OSMnx"
            except Exception as exc:
                errors.append(f"Overpass: {type(exc).__name__}: {exc}")

        if feats is None:
            raise RuntimeError(
                "All OSM health-facility acquisition routes failed. "
                + " | ".join(errors)
            )

        feats = feats.reset_index(drop=False)
        feats.to_file(services_path, driver="GeoJSON")
        log.log(
            f"Cached raw OSM health facilities: {services_path.name}; source={source}"
        )
    manifest["sources"]["health_services"] = {
        "provider": "OpenStreetMap/Overpass via OSMnx",
        "tags": cfg["data"]["osm_service_tags"],
        "path": str(services_path),
        "sha256": sha256_file(services_path),
    }

    pbf_cached = paths.raw / str(cfg["data"].get(
        "geofabrik_pbf_filename", "nord-pas-de-calais-latest.osm.pbf"
    ))
    if pbf_cached.exists():
        manifest["sources"]["osm_geofabrik_snapshot"] = {
            "provider": "Geofabrik / OpenStreetMap",
            "url": cfg["data"].get("geofabrik_pbf_url"),
            "path": str(pbf_cached),
            "bytes": int(pbf_cached.stat().st_size),
            "sha256": sha256_file(pbf_cached),
            "retrieved_utc": utc_now(),
        }

    manifest["packages"] = package_versions()
    manifest["python"] = sys.version
    manifest["platform"] = platform.platform()
    write_json(paths.manifests / "data_manifest.json", manifest)
    log.log("Public data download complete")
    return manifest


# -----------------------------------------------------------------------------
# Spatial units, services, network
# -----------------------------------------------------------------------------

def detect_column(df: pd.DataFrame, candidates: Sequence[str], numeric: bool = True) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            c = cols[cand.lower()]
            if not numeric or pd.api.types.is_numeric_dtype(df[c]):
                return c
    # partial match fallback
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in str(c).lower():
                if not numeric or pd.api.types.is_numeric_dtype(df[c]):
                    return c
    return None


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Numeric ratio with nonpositive/invalid denominators mapped to NaN."""
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = n / d.where(d > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def load_filosofi_vector(raw_path: Path, crs_filosofi: str):
    gpd, _, _ = require_geospatial()
    suf = raw_path.suffix.lower()
    if suf == ".parquet":
        try:
            gdf = gpd.read_parquet(raw_path)
            if gdf.crs is None:
                gdf = gdf.set_crs(crs_filosofi)
            return gdf
        except Exception:
            df = pd.read_parquet(raw_path)
    elif suf == ".csv":
        df = pd.read_csv(raw_path, low_memory=False)
    elif suf == ".zip":
        with zipfile.ZipFile(raw_path) as zf:
            names = zf.namelist()
            pq = [n for n in names if n.lower().endswith(".parquet")]
            shp = [n for n in names if n.lower().endswith(".shp")]
            if shp:
                return gpd.read_file(f"zip://{raw_path}!{shp[0]}")
            if pq:
                tmp = Path(tempfile.mkdtemp()) / Path(pq[0]).name
                with zf.open(pq[0]) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                return gpd.read_parquet(tmp)
        raise RuntimeError("Unsupported Filosofi ZIP content.")
    else:
        raise RuntimeError(f"Unsupported Filosofi resource format: {suf}")

    # Try WKT/geometry columns or 200m grid x/y identifiers.
    from shapely import wkt
    geom_col = next((c for c in df.columns if str(c).lower() in {"geometry", "geom", "wkt"}), None)
    if geom_col is not None:
        geom = df[geom_col].map(lambda x: wkt.loads(x) if isinstance(x, str) else x)
        return gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs=crs_filosofi)
    xcol = detect_column(df, ["x", "x_laea", "x3035", "easting"])
    ycol = detect_column(df, ["y", "y_laea", "y3035", "northing"])
    if xcol and ycol:
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        # Some open versions store x,y in 100m units.
        if np.nanmedian(np.abs(x)) < 100000:
            x = x * 100.0
            y = y * 100.0
        geom = gpd.points_from_xy(x, y, crs=crs_filosofi)
        return gpd.GeoDataFrame(df, geometry=geom, crs=crs_filosofi)
    raise RuntimeError("Could not infer geometry from Filosofi resource.")


def normalize_highway(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    s = str(value or "unclassified")
    return s if s else "unclassified"


def parse_numeric_tag(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)) and value:
        vals = [parse_numeric_tag(v, np.nan) for v in value]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else default
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(m.group()) if m else default


def parquet_safe_scalar(value: Any) -> Any:
    """Convert heterogeneous OSM object values to deterministic Arrow-safe scalars.

    OSMnx edge attributes such as ``osmid``, ``name`` or ``ref`` may contain a
    scalar on one edge and a list on another. PyArrow rejects such mixed object
    columns ("cannot mix list and non-list"). We preserve their information as
    stable JSON/text scalars in persisted diagnostic tables, while all numerical
    model primitives are stored in dedicated typed columns.
    """
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, sort_keys=False, default=str)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str)
    # Shapely geometry is handled by GeoPandas and must not be stringified.
    if hasattr(value, "geom_type"):
        return value
    return str(value)


def sanitize_geodataframe_for_parquet(gdf):
    """Return a copy whose non-geometry object columns are Arrow-safe.

    Numeric, boolean, datetime and geometry columns are kept unchanged.
    Object/string-like columns are converted to deterministic nullable strings.
    """
    out = gdf.copy()
    geom_name = out.geometry.name if hasattr(out, "geometry") else None
    for c in out.columns:
        if c == geom_name:
            continue
        s = out[c]
        if pd.api.types.is_object_dtype(s.dtype):
            out[c] = s.map(parquet_safe_scalar).astype("string")
    return out


def reduce_zones(zones, max_n: int, seed: int):
    """Deterministic population-weighted stratified reduction for computational tractability.

    It preserves the full processed 200m grid on disk. The reduced model sample is explicitly
    tagged and is only used when the number of populated units exceeds max_n.
    """
    if len(zones) <= max_n:
        zones = zones.copy()
        zones["source_zone_id"] = zones["zone_id"].astype(int)
        zones["zone_id"] = np.arange(len(zones), dtype=int)
        zones["model_weight"] = 1.0
        return zones
    rng = np.random.default_rng(seed)
    z = zones.copy().sort_values("P_i", ascending=False)
    # Always retain the most populated 20%; sample the rest PPS.
    n_top = max(1, int(0.2 * max_n))
    top = z.iloc[:n_top].copy()
    rest = z.iloc[n_top:].copy()
    n_rest = max_n - n_top
    probs = rest["P_i"].to_numpy(float)
    probs = probs / probs.sum()
    sel = rng.choice(len(rest), size=n_rest, replace=False, p=probs)
    sampled = rest.iloc[np.sort(sel)].copy()
    out = pd.concat([top, sampled], ignore_index=True)
    out["source_zone_id"] = out["zone_id"].astype(int)
    out["zone_id"] = np.arange(len(out), dtype=int)
    out["model_weight"] = 1.0
    return out


def build_spatial_units(cfg: Dict[str, Any], paths: Paths, log: Logger):
    gpd, _, ox = require_geospatial()
    manifest = read_json(paths.manifests / "data_manifest.json")
    filo_path = Path(manifest["sources"]["filosofi"]["path"])
    boundary = gpd.read_file(paths.raw / "mel_boundary.geojson").to_crs(cfg["project"]["crs_metric"])
    poly = union_geometry(boundary)
    log.log("Load and clip Filosofi 2021 grid to MEL")
    gdf = load_filosofi_vector(filo_path, cfg["project"]["crs_filosofi"]).to_crs(cfg["project"]["crs_metric"])
    if gdf.geometry.geom_type.isin(["Point"]).all():
        mask = gdf.geometry.within(poly)
    else:
        mask = gdf.geometry.centroid.within(poly)
    gdf = gdf.loc[mask].copy()

    pcol = detect_column(gdf, cfg["spatial"]["population_column_candidates"])
    if pcol is None:
        raise RuntimeError(
            f"Could not detect a Filosofi population column. "
            f"Available columns: {list(gdf.columns)[:120]}"
        )

    # Filosofi 2021 200m does not necessarily expose a precomputed median-income
    # or poverty-rate field. Official variables include ind, men, men_pauv and
    # ind_snv (sum of winsorized individual living standards). We therefore use
    # a strict, prespecified hierarchy:
    #   (1) direct income/living-standard field if present;
    #   (2) direct poverty-rate field if present;
    #   (3) derived mean living standard = ind_snv / ind;
    #   (4) derived household poverty share = men_pauv / men.
    # This keeps the baseline rule as a bottom-income/living-standard quartile
    # whenever the required official ingredients are available.
    icol = detect_column(gdf, cfg["spatial"]["income_column_candidates"])
    poverty_col = detect_column(gdf, cfg["spatial"]["poverty_column_candidates"])
    ind_snv_col = detect_column(gdf, ["ind_snv", "Ind_snv", "IND_SNV"])
    men_pauv_col = detect_column(gdf, ["men_pauv", "Men_pauv", "MEN_PAUV"])
    men_col = detect_column(gdf, ["men", "Men", "MEN"])

    gdf["P_i"] = pd.to_numeric(gdf[pcol], errors="coerce").fillna(0.0).clip(lower=0)
    gdf = gdf[gdf["P_i"] >= cfg["spatial"]["min_population"]].copy()

    vuln_source = None
    if icol is not None:
        gdf["X_income"] = pd.to_numeric(gdf[icol], errors="coerce")
        vuln_var = "X_income"
        vuln_source = f"direct:{icol}"
        direction = "bottom"
        qprob = float(cfg["spatial"]["vulnerable_quantile"])
    elif poverty_col is not None:
        gdf["X_poverty"] = pd.to_numeric(gdf[poverty_col], errors="coerce")
        vuln_var = "X_poverty"
        vuln_source = f"direct:{poverty_col}"
        direction = "top"
        qprob = 1.0 - float(cfg["spatial"]["vulnerable_quantile"])
    elif ind_snv_col is not None:
        # ind_snv is the sum of winsorized individual living standards and ind is
        # the number of individuals. Their ratio is a cell-level mean living
        # standard; it is not mislabeled as a median.
        gdf["X_living_standard_mean"] = safe_ratio(gdf[ind_snv_col], gdf[pcol])
        vuln_var = "X_living_standard_mean"
        vuln_source = f"derived:{ind_snv_col}/{pcol}"
        direction = "bottom"
        qprob = float(cfg["spatial"]["vulnerable_quantile"])
    elif men_pauv_col is not None and men_col is not None:
        gdf["X_poverty_share"] = safe_ratio(gdf[men_pauv_col], gdf[men_col])
        vuln_var = "X_poverty_share"
        vuln_source = f"derived:{men_pauv_col}/{men_col}"
        direction = "top"
        qprob = 1.0 - float(cfg["spatial"]["vulnerable_quantile"])
    else:
        raise RuntimeError(
            "Could not construct the prespecified vulnerability variable from "
            "Filosofi 2021. Expected either a direct income/poverty field, "
            "ind_snv with ind, or men_pauv with men. "
            f"Available columns: {list(gdf.columns)[:120]}"
        )

    valid = pd.to_numeric(gdf[vuln_var], errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_valid = int(valid.notna().sum())
    if n_valid == 0:
        raise RuntimeError(
            f"Vulnerability variable '{vuln_var}' contains no finite value after clipping to MEL."
        )
    gdf[vuln_var] = valid
    q = valid.dropna().quantile(qprob)
    if not np.isfinite(q):
        raise RuntimeError(f"Non-finite vulnerability threshold for '{vuln_var}'.")

    if direction == "bottom":
        gdf["is_vulnerable"] = (gdf[vuln_var] <= q).fillna(False)
    else:
        gdf["is_vulnerable"] = (gdf[vuln_var] >= q).fillna(False)

    threshold = float(q)
    log.log(
        f"Vulnerability variable: {vuln_var} ({vuln_source}); "
        f"direction={direction}; threshold={threshold:.6g}; valid_cells={n_valid:,}"
    )
    gdf["zone_id"] = np.arange(len(gdf), dtype=int)
    # Preserve centroids for later snapping/modeling.
    cent = gdf.geometry.centroid
    gdf["x_m"] = cent.x
    gdf["y_m"] = cent.y

    full_path = paths.processed / "spatial_units_filosofi_2021.geojson"
    gdf.to_file(full_path, driver="GeoJSON")
    model = reduce_zones(gdf, int(cfg["spatial"]["max_model_zones"]), cfg["compute"]["seeds"][0])
    model.to_file(paths.processed / "spatial_units_model.geojson", driver="GeoJSON")
    pd.DataFrame({
        "variable": [
            "population_column",
            "vulnerability_variable",
            "vulnerability_source",
            "vulnerability_threshold",
            "vulnerability_direction",
            "vulnerability_valid_cells",
            "n_full",
            "n_model",
            "population_full",
            "population_model"
        ],
        "value": [
            pcol,
            vuln_var,
            vuln_source,
            threshold,
            direction,
            n_valid,
            len(gdf),
            len(model),
            gdf.P_i.sum(),
            model.P_i.sum()
        ]
    }).to_csv(paths.processed / "spatial_unit_audit.csv", index=False)
    log.log(f"Spatial units: {len(gdf):,} populated cells; {len(model):,} model units")
    return model


def build_services(cfg: Dict[str, Any], paths: Paths, log: Logger):
    gpd, _, _ = require_geospatial()
    raw = gpd.read_file(paths.raw / "osm_health_services.geojson").to_crs(cfg["project"]["crs_metric"])
    if raw.empty:
        raise RuntimeError("Health-service file is empty.")
    # Convert non-point geometries to representative points.
    raw = raw[~raw.geometry.is_empty & raw.geometry.notna()].copy()
    raw["geometry"] = raw.geometry.representative_point()
    # Remove exact/near duplicates by 20m rounded coordinate key.
    raw["_gx"] = (raw.geometry.x / 20.0).round().astype(int)
    raw["_gy"] = (raw.geometry.y / 20.0).round().astype(int)
    raw = raw.drop_duplicates(["_gx", "_gy"]).copy()
    # Prefer hospitals/emergency if too many.
    priority = pd.Series(0, index=raw.index)
    for c in [c for c in raw.columns if str(c).lower() in {"amenity", "healthcare", "emergency"}]:
        s = raw[c].astype(str).str.lower()
        priority += s.str.contains("hospital|emergency").astype(int) * 2 + s.str.contains("clinic").astype(int)
    raw["priority"] = priority
    max_j = int(cfg["data"]["max_service_facilities"])
    if len(raw) > max_j:
        raw = raw.sort_values(["priority"], ascending=False).head(max_j).copy()
    raw["service_id"] = np.arange(len(raw), dtype=int)
    raw["O_j"] = 1.0
    out = raw[["service_id", "O_j", "priority", "geometry"]].copy()
    out.to_file(paths.processed / "essential_health_services.geojson", driver="GeoJSON")
    log.log(f"Essential health facilities retained: {len(out)}")
    return out


def build_network(cfg: Dict[str, Any], paths: Paths, log: Logger):
    gpd, _, ox = require_geospatial()
    nx, _ = require_network()
    projected_graphml = paths.processed / "mel_drive_network_projected.graphml"
    nodes_parquet = paths.processed / "network_nodes.parquet"
    edges_parquet = paths.processed / "network_edges.parquet"
    if (
        projected_graphml.exists()
        and nodes_parquet.exists()
        and edges_parquet.exists()
        and not cfg["data"]["overwrite_downloads"]
    ):
        # This is the canonical restart path for the frozen empirical system.
        G = load_graphml_compat(projected_graphml, log)
        nodes = gpd.read_parquet(nodes_parquet)
        edges = gpd.read_parquet(edges_parquet)
        for c in ("edge_id", "q0_sec", "K0_vph"):
            if c in edges.columns:
                edges[c] = pd.to_numeric(edges[c], errors="raise")
        if G.number_of_nodes() != len(nodes) or G.number_of_edges() != len(edges):
            raise RuntimeError(
                "Processed GraphML/GeoParquet cache is internally inconsistent: "
                f"graph=({G.number_of_nodes():,} nodes,{G.number_of_edges():,} edges), "
                f"tables=({len(nodes):,},{len(edges):,})."
            )
        log.log(
            "Reuse validated processed road network cache: "
            f"{projected_graphml.name} | {len(nodes):,} nodes, {len(edges):,} directed edges"
        )
        return G, nodes, edges

    boundary = gpd.read_file(paths.raw / "mel_boundary.geojson").to_crs(4326)
    geom = union_geometry(boundary)
    graphml = paths.raw / "mel_drive_network.graphml"
    if graphml.exists() and not cfg["data"]["overwrite_downloads"]:
        log.log("Reuse cached MEL road network")
        G = load_graphml_compat(graphml, log)
    else:
        log.log("Acquire/simplify directed MEL road network from OpenStreetMap")
        G = None
        source = None
        errors = []
        order = list(cfg["data"].get(
            "osm_acquisition_order", ["cache", "geofabrik", "overpass"]
        ))

        if "geofabrik" in order:
            try:
                sess = requests.Session()
                sess.headers.update({"User-Agent": cfg["data"]["user_agent"]})
                pbf = ensure_geofabrik_pbf(cfg, paths, sess, log)
                G = geofabrik_drive_graph(cfg, pbf, geom, log)
                source = "Geofabrik PBF + Pyrosm"
            except Exception as exc:
                errors.append(f"Geofabrik: {type(exc).__name__}: {exc}")
                log.log(
                    f"Geofabrik road extraction unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                G = None

        if G is None and "overpass" in order:
            endpoints = list(cfg["data"].get("overpass_endpoints", [
                "https://overpass-api.de/api",
                "https://overpass.kumi.systems/api",
                "https://overpass.private.coffee/api",
            ]))
            attempts = max(1, int(cfg["data"].get("overpass_attempts_per_endpoint", 1)))
            timeout = max(30, int(cfg["data"].get("overpass_timeout_seconds", 75)))
            pause = max(0.0, float(cfg["data"].get("overpass_retry_pause_seconds", 3)))

            old_url = getattr(ox.settings, "overpass_url", None)
            old_timeout = getattr(ox.settings, "requests_timeout", None)
            try:
                if hasattr(ox.settings, "requests_timeout"):
                    ox.settings.requests_timeout = timeout
                for endpoint_idx, endpoint in enumerate(endpoints, start=1):
                    if hasattr(ox.settings, "overpass_url"):
                        ox.settings.overpass_url = endpoint.rstrip("/")
                    for attempt in range(1, attempts + 1):
                        t0 = time.perf_counter()
                        log.log(
                            f"OSM road network: Overpass endpoint "
                            f"{endpoint_idx}/{len(endpoints)} {endpoint} "
                            f"attempt {attempt}/{attempts}, timeout={timeout}s"
                        )
                        try:
                            G = ox.graph_from_polygon(
                                geom,
                                network_type=cfg["data"]["osm_network_type"],
                                simplify=True,
                                retain_all=False,
                            )
                            if G is None or G.number_of_edges() == 0:
                                raise RuntimeError("Overpass returned an empty road graph.")
                            source = f"Overpass {endpoint}"
                            log.log(
                                f"OSM road network downloaded from {endpoint}: "
                                f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
                                f"in {time.perf_counter()-t0:.1f}s"
                            )
                            break
                        except Exception as exc:
                            elapsed = time.perf_counter() - t0
                            errors.append(
                                f"{endpoint} attempt {attempt}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            log.log(
                                f"Road-network Overpass attempt failed after {elapsed:.1f}s: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            G = None
                            if attempt < attempts and pause > 0:
                                log.log(f"Retry Overpass in {pause:.0f}s")
                                time.sleep(pause)
                    if G is not None:
                        break
            finally:
                if hasattr(ox.settings, "overpass_url") and old_url is not None:
                    ox.settings.overpass_url = old_url
                if hasattr(ox.settings, "requests_timeout") and old_timeout is not None:
                    ox.settings.requests_timeout = old_timeout

        if G is None:
            raise RuntimeError(
                "All OSM road-network acquisition routes failed. "
                + " | ".join(errors)
            )

        # OSMnx can derive speeds/travel times on either OSMnx- or Pyrosm-built
        # OSM-compatible graphs.
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, graphml)
        log.log(
            f"Cached raw MEL road network: {graphml.name}; source={source}"
        )
    G = ox.project_graph(G, to_crs=cfg["project"]["crs_metric"])
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges = edges.reset_index()
    nodes = nodes.reset_index()
    speed_defaults = cfg["network"]["default_speed_kph"]
    lane_defaults = cfg["network"]["default_lanes"]
    cap_lane = float(cfg["network"]["capacity_per_lane_vph"])
    q0 = []
    k0 = []
    hwy_norm = []
    for _, r in edges.iterrows():
        h = normalize_highway(r.get("highway"))
        hwy_norm.append(h)
        sp = parse_numeric_tag(r.get("speed_kph"), speed_defaults.get(h, 30.0))
        length = float(r.get("length", r.geometry.length))
        tt = r.get("travel_time")
        tt = parse_numeric_tag(tt, length / max(sp / 3.6, 0.1))
        lanes = parse_numeric_tag(r.get("lanes"), lane_defaults.get(h, 1.0))
        q0.append(max(tt, 0.01))
        k0.append(max(100.0, lanes * cap_lane))
    edges["highway_norm"] = hwy_norm
    edges["q0_sec"] = q0
    edges["K0_vph"] = k0
    edges["edge_id"] = np.arange(len(edges), dtype=int)
    edge_key_to_id = {
        (
            canonical_graph_id(r.u),
            canonical_graph_id(r.v),
            canonical_graph_id(r.key),
        ): int(r.edge_id)
        for r in edges.itertuples()
    }
    # Persist graph tables.
    #
    # OSMnx attributes are heterogeneous by design: for example ``osmid`` can be
    # an integer on one edge and a list of integers on another. PyArrow cannot
    # serialize a single object column that mixes scalar and list values. Keep the
    # in-memory ``edges`` object unchanged for modelling, but sanitize only the
    # persisted GeoParquet/GeoJSON copy. Core model columns (u, v, key, edge_id,
    # geometry, q0_sec, K0_vph, highway_norm) retain explicit typed values.
    nodes_disk = sanitize_geodataframe_for_parquet(nodes)
    edges_disk = sanitize_geodataframe_for_parquet(edges)
    nodes_disk.to_parquet(paths.processed / "network_nodes.parquet", index=False)
    edges_disk.to_parquet(paths.processed / "network_edges.parquet", index=False)
    edges_disk.to_file(paths.processed / "network_edges.geojson", driver="GeoJSON")
    ox.save_graphml(G, paths.processed / "mel_drive_network_projected.graphml")
    write_json(paths.processed / "edge_key_to_id.json", {"|".join(map(str,k)): v for k,v in edge_key_to_id.items()})
    log.log(f"Network: {len(nodes):,} nodes, {len(edges):,} directed edges")
    return G, nodes, edges


def snap_zones_services(cfg, paths, G, zones, services, log):
    _, _, ox = require_geospatial()
    z_ll = zones.to_crs(G.graph["crs"])
    s_ll = services.to_crs(G.graph["crs"])
    z_nodes = ox.distance.nearest_nodes(G, X=z_ll.geometry.centroid.x.to_numpy(), Y=z_ll.geometry.centroid.y.to_numpy())
    s_nodes = ox.distance.nearest_nodes(G, X=s_ll.geometry.x.to_numpy(), Y=s_ll.geometry.y.to_numpy())
    zones = zones.copy(); services = services.copy()
    zones["rho_node"] = np.asarray(z_nodes, dtype=np.int64)
    services["sigma_node"] = np.asarray(s_nodes, dtype=np.int64)
    zones.to_file(paths.processed / "spatial_units_model_snapped.geojson", driver="GeoJSON")
    services.to_file(paths.processed / "essential_health_services_snapped.geojson", driver="GeoJSON")
    return zones, services


def process_all(cfg, paths, log):
    gpd, _, ox = require_geospatial()
    zones = build_spatial_units(cfg, paths, log)
    services = build_services(cfg, paths, log)
    G, nodes, edges = build_network(cfg, paths, log)
    zones, services = snap_zones_services(cfg, paths, G, zones, services, log)
    manifest = read_json(paths.manifests / "data_manifest.json")
    manifest["processed"] = {
        "n_zones_full": int(len(gpd.read_file(paths.processed / "spatial_units_filosofi_2021.geojson"))),
        "n_zones_model": int(len(zones)),
        "n_services": int(len(services)),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "population_model": float(zones.P_i.sum()),
        "vulnerable_population_model": float(zones.loc[zones.is_vulnerable.astype(bool), "P_i"].sum()),
    }
    write_json(paths.manifests / "data_manifest.json", manifest)
    return G, zones, services, edges


# -----------------------------------------------------------------------------
# Routing/accessibility engine
# -----------------------------------------------------------------------------

def load_graphml_compat(path: Path, log: Optional[Logger] = None):
    """Load OSMnx- or Pyrosm-generated GraphML without strict boolean coercion."""
    nx, _ = require_network()
    if log:
        log.log(f"Load GraphML with compatibility reader: {path.name}")
    G0 = nx.read_graphml(path)
    if isinstance(G0, nx.MultiDiGraph):
        G = G0
    else:
        G = nx.MultiDiGraph()
        G.graph.update(G0.graph)
        for n, d in G0.nodes(data=True):
            G.add_node(n, **dict(d))
        if isinstance(G0, (nx.MultiGraph, nx.MultiDiGraph)):
            for u, v, k, d in G0.edges(keys=True, data=True):
                G.add_edge(u, v, key=k, **dict(d))
        else:
            for u, v, d in G0.edges(data=True):
                G.add_edge(u, v, key=0, **dict(d))
    mapping = {}
    for n in list(G.nodes):
        cn = canonical_graph_id(n)
        if cn != n:
            mapping[n] = cn
    if mapping:
        G = nx.relabel_nodes(G, mapping, copy=True)
    true_tokens = {"true", "1", "yes", "y", "t"}
    false_tokens = {"false", "0", "no", "n", "f"}
    def _to_bool_or_raw(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in true_tokens:
            return True
        if s in false_tokens:
            return False
        return v
    for _, d in G.nodes(data=True):
        for a in ("x", "y"):
            if a in d:
                try:
                    d[a] = float(d[a])
                except Exception:
                    pass
    float_attrs = {"length", "speed_kph", "travel_time", "q0_sec", "capacity_vph", "lanes_num", "freeflow_time_sec"}
    for _, _, _, d in G.edges(keys=True, data=True):
        for a in ("oneway", "reversed"):
            if a in d:
                d[a] = _to_bool_or_raw(d[a])
        for a in float_attrs:
            if a in d:
                try:
                    d[a] = float(d[a])
                except Exception:
                    pass
    if not G.graph.get("crs"):
        G.graph["crs"] = "EPSG:4326"
    if log:
        log.log(f"Compatibility GraphML loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges")
    return G


def canonical_graph_id(value: Any) -> str:
    """Canonical string representation for OSM/GraphML node and edge identifiers.

    GeoParquet, GeoJSON and GraphML can round-trip the same integral identifier as
    ``453266774``, ``"453266774"`` or ``453266774.0``.  Raw ``str(value)`` is
    therefore not a stable join key.  Integral numeric representations are
    normalized to their integer decimal form; non-numeric identifiers are left
    as stripped strings.
    """
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    s = str(value).strip()
    if not s:
        return s

    # Normalize strings such as "453266774.0" without altering genuinely
    # non-numeric OSM/GraphML identifiers.
    try:
        x = float(s)
        if np.isfinite(x) and x.is_integer():
            # Guard against scientific notation / large integer formatting.
            if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
                return str(int(x))
    except Exception:
        pass
    return s


def prepare_reference_graph(cfg, paths, log):
    gpd, _, ox = require_geospatial()
    nx, _ = require_network()
    G = load_graphml_compat(paths.processed / "mel_drive_network_projected.graphml", log)
    edges = gpd.read_parquet(paths.processed / "network_edges.parquet")
    # GeoParquet sanitization may store OSM identifiers as strings. Downstream
    # lookup functions are string-normalized, while edge_id/q0/K0 remain numeric.
    for c in ("edge_id", "q0_sec", "K0_vph"):
        if c in edges.columns:
            edges[c] = pd.to_numeric(edges[c], errors="raise")
    zones = gpd.read_file(paths.processed / "spatial_units_model_snapped.geojson")
    services = gpd.read_file(paths.processed / "essential_health_services_snapped.geojson")
    # GraphML / GeoJSON can represent the same OSM id as int, string, or
    # integral float. Use one canonical key everywhere.
    node_map = {canonical_graph_id(n): n for n in G.nodes}
    zones["rho_node"] = zones["rho_node"].map(
        lambda x: node_map.get(canonical_graph_id(x), x)
    )
    services["sigma_node"] = services["sigma_node"].map(
        lambda x: node_map.get(canonical_graph_id(x), x)
    )

    missing_zone_nodes = [
        x for x in zones["rho_node"].tolist()
        if canonical_graph_id(x) not in node_map
    ]
    missing_service_nodes = [
        x for x in services["sigma_node"].tolist()
        if canonical_graph_id(x) not in node_map
    ]
    if missing_zone_nodes or missing_service_nodes:
        raise RuntimeError(
            "Snapped node identifiers cannot be reconciled with the projected GraphML: "
            f"missing_zones={len(missing_zone_nodes)}, "
            f"missing_services={len(missing_service_nodes)}."
        )
    return G, zones, services, edges


def edge_lookup_from_edges(edges: pd.DataFrame) -> Dict[Tuple[str,str,str], int]:
    lut: Dict[Tuple[str, str, str], int] = {}
    for r in edges.itertuples():
        key = (
            canonical_graph_id(r.u),
            canonical_graph_id(r.v),
            canonical_graph_id(r.key),
        )
        eid = int(r.edge_id)
        if key in lut and lut[key] != eid:
            raise RuntimeError(
                f"Non-unique canonical edge key {key}: edge_id={lut[key]} and {eid}."
            )
        lut[key] = eid
    return lut


def edge_weight_set(G, edges: pd.DataFrame, attr: str, values: np.ndarray):
    lut = {
        (
            canonical_graph_id(r.u),
            canonical_graph_id(r.v),
            canonical_graph_id(r.key),
        ): float(values[int(r.edge_id)])
        for r in edges.itertuples()
    }
    missing = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        ek = (
            canonical_graph_id(u),
            canonical_graph_id(v),
            canonical_graph_id(k),
        )
        if ek in lut:
            d[attr] = lut[ek]
        else:
            missing += 1
            d[attr] = float(d.get("travel_time", 1.0))
    if missing:
        raise RuntimeError(
            f"{missing} GraphML edge(s) could not be reconciled with network_edges.parquet "
            f"while setting '{attr}'."
        )


def nearest_service_pairs(G, zones, services, weight: str, k: int):
    nx, _ = require_network()
    service_nodes = list(dict.fromkeys(services["sigma_node"].tolist()))
    service_info = {n: float(services.loc[services.sigma_node == n, "O_j"].sum()) for n in service_nodes}
    Gr = G.reverse(copy=False)
    # distances service->all in reversed graph = all->service in original.
    dist_maps = {}
    for sn in service_nodes:
        dist_maps[sn] = nx.single_source_dijkstra_path_length(Gr, sn, weight=weight)
    rows = []
    for z in zones.itertuples():
        cand = []
        for sn in service_nodes:
            d = dist_maps[sn].get(z.rho_node, math.inf)
            if np.isfinite(d):
                cand.append((d, sn, service_info[sn]))
        cand.sort(key=lambda x: x[0])
        for rank, (d, sn, oj) in enumerate(cand[:k]):
            rows.append({"zone_id": int(z.zone_id), "origin_node": z.rho_node, "service_node": sn, "service_rank": rank, "c0_sec": float(d), "O_j": float(oj)})
    return pd.DataFrame(rows)


def collapse_multidigraph_for_paths(G, weight: str):
    """Build the simple directed graph used by Yen/shortest_simple_paths once.

    v1.0.9 rebuilt this ~70k-edge graph for every OD pair, which dominated
    runtime. The collapsed graph depends only on the reference edge weights and
    is therefore identical for every OD query in this stage.
    """
    nx, _ = require_network()
    H = nx.DiGraph()
    for u, v, key, d in G.edges(keys=True, data=True):
        w = float(d.get(weight, d.get("travel_time", 1.0)))
        if not H.has_edge(u, v) or w < H[u][v]["weight"]:
            H.add_edge(u, v, weight=w, key=key)
    return H


def k_shortest_edge_paths(H, origin, dest, edge_lut, k: int):
    """Return up to k loopless edge-id paths on a pre-collapsed DiGraph."""
    nx, _ = require_network()
    try:
        gen = nx.shortest_simple_paths(H, origin, dest, weight="weight")
        out = []
        for _ in range(k):
            nodes = next(gen)
            eids = []
            for a, b in zip(nodes[:-1], nodes[1:]):
                key = H[a][b]["key"]
                ek = (
                    canonical_graph_id(a),
                    canonical_graph_id(b),
                    canonical_graph_id(key),
                )
                eid = edge_lut.get(ek)
                if eid is None:
                    raise RuntimeError(
                        "Route-path edge is absent from the canonical edge lookup: "
                        f"raw=({a!r}, {b!r}, {key!r}), canonical={ek}. "
                        "This indicates a GraphML/GeoParquet identifier mismatch."
                    )
                eids.append(eid)
            out.append(eids)
        return out
    except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
        return []


def build_route_choice_set(cfg, paths, G, zones, services, edges, log):
    cache = paths.processed / "route_choice_paths.json"
    pairs_path = paths.processed / "zone_service_pairs.parquet"
    work_cache = paths.processed / "route_choice_od_cache.json"

    if cache.exists() and pairs_path.exists():
        final_pairs = pd.read_parquet(pairs_path)
        final_paths = read_json(cache)["paths"]
        log.log(
            f"Reuse cached route-choice set: {len(final_pairs):,} OD pairs, "
            f"{sum(len(x) for x in final_paths):,} candidate paths"
        )
        return final_pairs, final_paths

    edge_lut = edge_lookup_from_edges(edges)
    edge_weight_set(G, edges, "q0_model", edges.q0_sec.to_numpy(float))

    t0 = time.perf_counter()
    log.log("Build nearest-service OD set")
    pairs = nearest_service_pairs(
        G, zones, services, "q0_model",
        int(cfg["network"]["nearest_services_per_zone"])
    )
    if pairs.empty:
        raise RuntimeError("No reachable zone-service pairs under the reference network.")
    log.log(
        f"Nearest-service OD set ready: {len(pairs):,} pairs "
        f"in {time.perf_counter()-t0:.1f}s"
    )

    # Reference demand split by baseline impedance.
    kappa = math.log(2.0) / (
        float(cfg["accessibility"]["half_life_minutes"]) * 60.0
    )
    pop = zones.set_index("zone_id")["P_i"].to_dict()
    pairs["grav"] = pairs["O_j"] * np.exp(-kappa * pairs["c0_sec"])
    pairs["grav_sum"] = pairs.groupby("zone_id")["grav"].transform("sum")
    pairs["demand_vph"] = pairs.apply(
        lambda r: pop[int(r.zone_id)]
        * cfg["network"]["reference_demand_scale_vph_per_person"]
        * r.grav / max(r.grav_sum, 1e-12),
        axis=1,
    )

    # Critical optimization: collapse the 71k-edge MultiDiGraph ONCE, not once
    # per OD pair as in v1.0.9.
    t_collapse = time.perf_counter()
    log.log(
        f"Collapse MultiDiGraph once for candidate-path enumeration "
        f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges)"
    )
    H = collapse_multidigraph_for_paths(G, "q0_model")
    log.log(
        f"Collapsed path graph ready: {H.number_of_nodes():,} nodes, "
        f"{H.number_of_edges():,} edges in {time.perf_counter()-t_collapse:.1f}s"
    )

    k_paths = int(cfg["network"]["candidate_paths_per_od"])
    progress_every = max(1, int(cfg["network"].get("route_progress_every", 100)))
    checkpoint_every = max(1, int(cfg["network"].get("route_checkpoint_every", 250)))

    # Cache by snapped (origin,destination), because several Filosofi cells can
    # snap to the same road node. This is exact reuse, not an approximation.
    od_cache = {}
    cache_meta = {
        "n_graph_edges": int(G.number_of_edges()),
        "n_simple_edges": int(H.number_of_edges()),
        "candidate_paths_per_od": k_paths,
        "weight": "q0_model",
    }
    if work_cache.exists():
        try:
            wc = read_json(work_cache)
            if wc.get("meta") == cache_meta:
                od_cache = wc.get("od_paths", {})
                log.log(
                    f"Resume route enumeration from checkpoint: "
                    f"{len(od_cache):,} unique OD path queries cached"
                )
            else:
                log.log("Ignore stale route-choice checkpoint (configuration mismatch)")
        except Exception as exc:
            log.log(f"Ignore unreadable route-choice checkpoint: {exc}")

    paths_list = []
    kept_rows = []
    n_total = len(pairs)
    n_no_path = 0
    n_reused = 0
    n_computed = 0
    enum_start = time.perf_counter()
    last_checkpoint_unique = len(od_cache)

    log.log(
        f"Enumerate candidate paths for {n_total:,} zone-service pairs; "
        f"k={k_paths}; progress every {progress_every}; checkpoint every {checkpoint_every}"
    )

    for pos, r in enumerate(pairs.itertuples(index=False), start=1):
        origin = r.origin_node
        dest = r.service_node
        cache_key = f"{canonical_graph_id(origin)}|{canonical_graph_id(dest)}"

        if cache_key in od_cache:
            pths = od_cache[cache_key]
            n_reused += 1
        else:
            pths = k_shortest_edge_paths(H, origin, dest, edge_lut, k_paths)
            od_cache[cache_key] = pths
            n_computed += 1

        if not pths:
            n_no_path += 1
        else:
            rr = r._asdict()
            rr["od_id"] = len(kept_rows)
            kept_rows.append(rr)
            paths_list.append(pths)

        # Periodic progress with elapsed time, throughput, and ETA.
        if pos == 1 or pos % progress_every == 0 or pos == n_total:
            elapsed = time.perf_counter() - enum_start
            rate = pos / max(elapsed, 1e-9)
            remaining = n_total - pos
            eta = remaining / max(rate, 1e-9)
            pct = 100.0 * pos / max(n_total, 1)
            log.log(
                f"Route paths: {pos:,}/{n_total:,} ({pct:5.1f}%) | "
                f"kept={len(kept_rows):,} no_path={n_no_path:,} | "
                f"unique_computed={n_computed:,} reused={n_reused:,} | "
                f"{rate:.2f} pairs/s | elapsed={elapsed/60:.1f} min | "
                f"ETA={eta/60:.1f} min"
            )

        # Durable unique-OD checkpoint. If execution is interrupted, v1.0.10
        # resumes without recomputing completed origin-destination path queries.
        if len(od_cache) - last_checkpoint_unique >= checkpoint_every:
            write_json(work_cache, {"meta": cache_meta, "od_paths": od_cache})
            last_checkpoint_unique = len(od_cache)
            log.log(
                f"Route checkpoint saved: {len(od_cache):,} unique OD queries"
            )

    if not kept_rows:
        raise RuntimeError("Candidate-path enumeration produced no reachable OD pair.")

    pairs = pd.DataFrame(kept_rows)
    pairs.to_parquet(pairs_path, index=False)
    write_json(cache, {"paths": paths_list})
    write_json(work_cache, {"meta": cache_meta, "od_paths": od_cache})

    elapsed = time.perf_counter() - enum_start
    total_paths = sum(len(x) for x in paths_list)
    log.log(
        f"Candidate-path enumeration complete: {len(pairs):,}/{n_total:,} OD pairs kept, "
        f"{total_paths:,} paths, {len(od_cache):,} unique snapped OD queries, "
        f"elapsed={elapsed/60:.1f} min"
    )
    return pairs, paths_list


class RouteEngine:
    """Torch-accelerated deterministic routing operator Ψ plus exact accessibility evaluator.

    Ψ uses a prespecified finite candidate-path choice set; reported c_ij/A_i are computed
    on the complete operational directed graph from the resulting edge costs.
    """
    def __init__(self, cfg, edges: pd.DataFrame, pairs: pd.DataFrame, paths_list: List[List[List[int]]], device: str):
        torch = require_torch()
        self.torch = torch
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.n_edges = len(edges)
        self.n_od = len(pairs)
        self.K0 = torch.tensor(edges.K0_vph.to_numpy(float), dtype=self.dtype, device=self.device)
        self.q0 = torch.tensor(edges.q0_sec.to_numpy(float), dtype=self.dtype, device=self.device)
        self.f0 = torch.zeros(self.n_edges, dtype=self.dtype, device=self.device)
        self.od_demand = torch.tensor(pairs.demand_vph.to_numpy(float), dtype=self.dtype, device=self.device)
        self.od_zone = torch.tensor(pairs.zone_id.to_numpy(int), dtype=torch.long, device=self.device)
        self.od_O = torch.tensor(pairs.O_j.to_numpy(float), dtype=self.dtype, device=self.device)
        self.n_zones = int(pairs.zone_id.max()) + 1
        self.kappa = math.log(2.0) / (float(cfg["accessibility"]["half_life_minutes"]) * 60.0)
        # Flatten paths; store sparse incidence and path->OD mapping.
        rows, cols, vals = [], [], []
        path_od = []
        for od, plist in enumerate(paths_list):
            for p in plist:
                pid = len(path_od)
                path_od.append(od)
                for e in p:
                    rows.append(pid); cols.append(int(e)); vals.append(1.0)
        self.n_paths = len(path_od)
        idx = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        val = torch.tensor(vals, dtype=self.dtype, device=self.device)
        self.PE = torch.sparse_coo_tensor(idx, val, (self.n_paths, self.n_edges), device=self.device).coalesce()
        # Number of prespecified service-access candidate paths using each directed edge.
        # This is fixed before policy learning and is therefore pre-action information.
        self.path_use_count = np.bincount(
            np.asarray(cols, dtype=np.int64), minlength=self.n_edges
        ).astype(np.int64)
        self.path_od = torch.tensor(path_od, dtype=torch.long, device=self.device)
        counts = np.bincount(np.asarray(path_od), minlength=self.n_od)
        self.od_path_count = torch.tensor(counts, dtype=torch.long, device=self.device)
        self.alpha = float(cfg["network"]["bpr_alpha"])
        self.beta = float(cfg["network"]["bpr_beta"])
        self.iters = int(cfg["network"]["routing_iterations"])
        self.damping = float(cfg["network"]["routing_damping"])
        self.temp = float(cfg["network"]["route_choice_temperature"])

    def _q_from_fK(self, f, K):
        t = self.torch
        eps = 1e-6
        # Reference-consistent congestion model Q_e(f,K).
        ratio = f / t.clamp(K, min=eps)
        base_ratio = self.f0 / t.clamp(self.K0, min=eps)
        numer = 1.0 + self.alpha * t.pow(t.clamp(ratio, min=0.0), self.beta)
        denom = 1.0 + self.alpha * t.pow(t.clamp(base_ratio, min=0.0), self.beta)
        # Parenthesize the normalization ratio. At (f,K)=(f0,K0), numer and
        # denom are bitwise identical, hence numer/denom is exactly one even
        # in float32 and the defining reference identity returns q0 exactly.
        return self.q0 * (numer / denom)

    def _path_cost(self, q, K):
        t = self.torch
        pc = t.sparse.mm(self.PE, q[:, None]).squeeze(1)
        zero = (K <= 1e-9).to(self.dtype)
        blocked = t.sparse.mm(self.PE, zero[:, None]).squeeze(1) > 0
        pc = t.where(blocked, t.tensor(float("inf"), device=self.device), pc)
        return pc

    def _path_shares(self, path_cost):
        """Exact vectorized OD-segment softmax on the GPU.

        This is algebraically identical to the former OD-by-OD loop but avoids
        thousands of tiny Python-launched CUDA operations per routing iteration.
        """
        t=self.torch
        finite=t.isfinite(path_cost)
        safe=t.where(finite,path_cost,t.full_like(path_cost,float("inf")))
        od_min=t.full((self.n_od,),float("inf"),dtype=self.dtype,device=self.device)
        od_min.scatter_reduce_(0,self.path_od,safe,reduce="amin",include_self=True)
        centered=safe-od_min[self.path_od]
        numer=t.where(finite,t.exp(-self.temp*centered),t.zeros_like(path_cost))
        denom=t.zeros(self.n_od,dtype=self.dtype,device=self.device)
        denom.scatter_add_(0,self.path_od,numer)
        return t.where(
            finite & (denom[self.path_od]>0),
            numer/t.clamp(denom[self.path_od],min=1e-30),
            t.zeros_like(numer)
        )

    def audit_vectorized_path_shares(self):
        """Check vectorized shares against the legacy OD-loop implementation."""
        t=self.torch
        with t.no_grad():
            pc=self._path_cost(self.q0,self.K0)
            fast=self._path_shares(pc)
            legacy=t.zeros_like(pc)
            start=0
            for cnt in self.od_path_count.detach().cpu().tolist():
                sl=slice(start,start+cnt); c=pc[sl]; finite=t.isfinite(c)
                if finite.any():
                    cf=c[finite]
                    w=t.softmax(-self.temp*(cf-cf.min()),dim=0)
                    tmp=t.zeros_like(c); tmp[finite]=w; legacy[sl]=tmp
                start+=cnt
            err=float(t.max(t.abs(fast-legacy)).detach().cpu())
            mass=t.zeros(self.n_od,dtype=self.dtype,device=self.device)
            mass.scatter_add_(0,self.path_od,fast)
            finite_count=t.zeros(self.n_od,dtype=self.dtype,device=self.device)
            finite_count.scatter_add_(0,self.path_od,t.isfinite(pc).to(self.dtype))
            valid=finite_count>0
            mass_err=float(t.max(t.abs(mass[valid]-1.0)).detach().cpu()) if bool(valid.any()) else 0.0
        return err,mass_err

    def _edge_flow(self, path_share):
        t = self.torch
        path_dem = self.od_demand[self.path_od] * path_share
        return t.sparse.mm(self.PE.transpose(0,1), path_dem[:,None]).squeeze(1)

    def calibrate_reference(self):
        t = self.torch
        # Fixed-point baseline assignment under K0; q0 is the required reference cost.
        # We estimate f0 from q0 path choice, then Q is normalized so Q(f0,K0)=q0 exactly.
        pc = self._path_cost(self.q0, self.K0)
        sh = self._path_shares(pc)
        self.f0 = self._edge_flow(sh).detach()
        qchk = self._q_from_fK(self.f0, self.K0)
        err = float(t.max(t.abs(qchk - self.q0)).cpu())
        return err

    @property
    def f0_cpu(self):
        return self.f0.detach().cpu().numpy()

    def attach_exact_accessibility(self, G, zones, services, edges):
        """Attach the exact shortest-path accessibility evaluator used for reported outcomes.

        Routing Ψ uses the fixed candidate-path choice set for speed, but c_ij and A_i are
        evaluated on the complete current directed graph, exactly matching Section 3.
        Parallel edges are reduced by their minimum current generalized cost.
        """
        node_ids = list(G.nodes)
        self._node_to_i = {str(n): i for i, n in enumerate(node_ids)}
        pair_to_idx = {}
        pair_u=[]; pair_v=[]; edge_pair=np.empty(len(edges), dtype=np.int64)
        for r in edges.itertuples():
            key=(self._node_to_i[str(r.u)], self._node_to_i[str(r.v)])
            if key not in pair_to_idx:
                pair_to_idx[key]=len(pair_u); pair_u.append(key[0]); pair_v.append(key[1])
            edge_pair[int(r.edge_id)] = pair_to_idx[key]
        self._pair_u=np.asarray(pair_u,dtype=np.int64); self._pair_v=np.asarray(pair_v,dtype=np.int64); self._edge_pair=edge_pair
        self._n_graph_nodes=len(node_ids)
        zsort=zones.sort_values("zone_id")
        self._zone_node_i=np.asarray([self._node_to_i[str(x)] for x in zsort.rho_node],dtype=np.int64)
        # Aggregate multiple facilities snapped to the same graph node.
        svc=services.groupby("sigma_node",as_index=False)["O_j"].sum()
        self._service_node_i=np.asarray([self._node_to_i[str(x)] for x in svc.sigma_node],dtype=np.int64)
        self._service_O=svc.O_j.to_numpy(float)

    def accessibility_exact(self, q_np: np.ndarray, K_np: np.ndarray):
        """Exact A_i on the full operational directed graph under current edge costs."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        q=np.asarray(q_np,float); K=np.asarray(K_np,float)
        pair_q=np.full(len(self._pair_u), np.inf, dtype=float)
        usable=(K>1e-9) & np.isfinite(q)
        np.minimum.at(pair_q, self._edge_pair[usable], q[usable])
        keep=np.isfinite(pair_q)
        mat=csr_matrix((pair_q[keep],(self._pair_u[keep],self._pair_v[keep])),shape=(self._n_graph_nodes,self._n_graph_nodes))
        # Distances from every node to each service = dijkstra from services on reversed graph.
        dist=dijkstra(mat.transpose().tocsr(), directed=True, indices=self._service_node_i, return_predecessors=False)
        dz=dist[:, self._zone_node_i].T  # zones x services
        contrib=np.exp(-self.kappa*dz, where=np.isfinite(dz), out=np.zeros_like(dz)) * self._service_O[None,:]
        return contrib.sum(axis=1)


    def _exact_cost_matrix(self, q_np: np.ndarray, K_np: np.ndarray):
        """Build the same reduced operational CSR cost matrix as accessibility_exact."""
        from scipy.sparse import csr_matrix
        q=np.asarray(q_np,float); K=np.asarray(K_np,float)
        pair_q=np.full(len(self._pair_u),np.inf,dtype=float)
        usable=(K>1e-9) & np.isfinite(q)
        np.minimum.at(pair_q,self._edge_pair[usable],q[usable])
        keep=np.isfinite(pair_q)
        return csr_matrix(
            (pair_q[keep],(self._pair_u[keep],self._pair_v[keep])),
            shape=(self._n_graph_nodes,self._n_graph_nodes)
        ).transpose().tocsr()

    def accessibility_exact_threaded(
        self, q_np: np.ndarray, K_np: np.ndarray, workers: int = 4
    ):
        """Mathematically identical exact accessibility with source-parallel Dijkstra.

        The operational graph, edge costs, directed shortest-path problem,
        service nodes, impedance function, and service weights are unchanged.
        Only the independent service-source Dijkstra calls are partitioned
        across CPU threads. SciPy's compiled shortest-path routine releases the
        GIL on the expensive kernel on supported builds; the audit stage decides
        empirically whether this is faster on the user's machine.

        This method is a candidate implementation until it passes the strict
        legacy-equivalence audit.
        """
        from scipy.sparse.csgraph import dijkstra
        from concurrent.futures import ThreadPoolExecutor

        rev=self._exact_cost_matrix(q_np,K_np)
        svc=np.asarray(self._service_node_i,dtype=np.int64)
        nsvc=len(svc)
        workers=max(1,min(int(workers),nsvc))
        if workers==1 or nsvc<=1:
            dist=dijkstra(
                rev,directed=True,indices=svc,return_predecessors=False
            )
        else:
            chunks=[
                x for x in np.array_split(np.arange(nsvc,dtype=np.int64),workers)
                if len(x)
            ]
            def run(ix):
                return ix,dijkstra(
                    rev,directed=True,indices=svc[ix],
                    return_predecessors=False
                )
            parts=[]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ix,dd in ex.map(run,chunks):
                    parts.append((ix,dd))
            dist=np.empty((nsvc,self._n_graph_nodes),dtype=float)
            for ix,dd in parts:
                dist[ix,:]=np.atleast_2d(dd)

        dz=dist[:,self._zone_node_i].T
        contrib=np.exp(
            -self.kappa*dz,
            where=np.isfinite(dz),
            out=np.zeros_like(dz)
        )*self._service_O[None,:]
        return contrib.sum(axis=1)

    def solve(self, K_np: np.ndarray):
        t = self.torch
        K = t.tensor(K_np, dtype=self.dtype, device=self.device)
        f = self.f0.clone()
        q = self._q_from_fK(f, K)
        for _ in range(self.iters):
            pc = self._path_cost(q, K)
            sh = self._path_shares(pc)
            f_new = self._edge_flow(sh)
            f = self.damping * f_new + (1.0 - self.damping) * f
            q = self._q_from_fK(f, K)
        return f.detach().cpu().numpy(), q.detach().cpu().numpy(), self._path_cost(q, K).detach().cpu().numpy()

    def solve_path_cost_only(self, K_np: np.ndarray):
        """Same routing fixed point as solve(), returning only path costs.

        Training with the candidate-path accessibility surrogate does not need
        the two 534k-edge f/q arrays on CPU. Avoiding those GPU-to-CPU transfers
        is a computational optimization only; routing equations are unchanged.
        """
        t=self.torch
        K=t.as_tensor(K_np,dtype=self.dtype,device=self.device)
        f=self.f0.clone()
        q=self._q_from_fK(f,K)
        for _ in range(self.iters):
            pc=self._path_cost(q,K)
            sh=self._path_shares(pc)
            f_new=self._edge_flow(sh)
            f=self.damping*f_new+(1.0-self.damping)*f
            q=self._q_from_fK(f,K)
        return self._path_cost(q,K).detach().cpu().numpy()

    def solve_path_cost_batch(self, K_np: np.ndarray, batch_chunk: int = 32):
        """Batched routing fixed point for benchmark construction.

        Parameters
        ----------
        K_np : array, shape (B, n_edges)
            Capacity states for B independent trajectories.
        batch_chunk : int
            Maximum number of states processed simultaneously on the GPU.

        Returns
        -------
        ndarray, shape (B, n_paths)
            Candidate-path costs.

        Notes
        -----
        This is algebraically the same fixed-point routing map as
        ``solve_path_cost_only``. The batch dimension is computational only.
        No approximation is introduced by batching.
        """
        t=self.torch
        K_np=np.asarray(K_np, dtype=np.float32)
        if K_np.ndim != 2 or K_np.shape[1] != self.n_edges:
            raise ValueError(
                f"K_np must have shape (B,{self.n_edges}), got {K_np.shape}"
            )
        B=int(K_np.shape[0])
        if B==0:
            return np.empty((0,self.n_paths),dtype=np.float32)

        out=[]
        chunk=max(1,int(batch_chunk))
        path_od_col=self.path_od[:,None]

        with t.no_grad():
            for lo in range(0,B,chunk):
                hi=min(B,lo+chunk)
                K=t.as_tensor(
                    K_np[lo:hi],dtype=self.dtype,device=self.device
                )                                   # b x E
                b=K.shape[0]
                f=self.f0[None,:].expand(b,-1).clone()
                # Batched version of _q_from_fK.
                eps=1e-6
                base_ratio=self.f0/t.clamp(self.K0,min=eps)
                base_denom=1.0+self.alpha*t.pow(
                    t.clamp(base_ratio,min=0.0),self.beta
                )

                def q_from_fK_batch(fb,Kb):
                    ratio=fb/t.clamp(Kb,min=eps)
                    numer=1.0+self.alpha*t.pow(
                        t.clamp(ratio,min=0.0),self.beta
                    )
                    return self.q0[None,:]*(numer/base_denom[None,:])

                def path_cost_batch(qb,Kb):
                    # sparse (P x E) @ dense (E x b) -> P x b
                    pc=t.sparse.mm(self.PE,qb.transpose(0,1))
                    zero=(Kb<=1e-9).to(self.dtype)
                    blocked=t.sparse.mm(
                        self.PE,zero.transpose(0,1)
                    )>0
                    return t.where(
                        blocked,
                        t.full_like(pc,float("inf")),
                        pc
                    )                               # P x b

                def path_shares_batch(pc):
                    finite=t.isfinite(pc)
                    safe=t.where(
                        finite,pc,t.full_like(pc,float("inf"))
                    )
                    idx=path_od_col.expand(-1,b)
                    od_min=t.full(
                        (self.n_od,b),float("inf"),
                        dtype=self.dtype,device=self.device
                    )
                    od_min.scatter_reduce_(
                        0,idx,safe,reduce="amin",include_self=True
                    )
                    centered=safe-od_min[self.path_od,:]
                    numer=t.where(
                        finite,
                        t.exp(-self.temp*centered),
                        t.zeros_like(pc)
                    )
                    denom=t.zeros(
                        (self.n_od,b),
                        dtype=self.dtype,device=self.device
                    )
                    denom.scatter_add_(0,idx,numer)
                    return t.where(
                        finite & (denom[self.path_od,:]>0),
                        numer/t.clamp(
                            denom[self.path_od,:],min=1e-30
                        ),
                        t.zeros_like(numer)
                    )

                def edge_flow_batch(sh):
                    # PE.T (E x P) @ (P x b) -> E x b -> b x E
                    path_dem=self.od_demand[self.path_od,None]*sh
                    return t.sparse.mm(
                        self.PE.transpose(0,1),path_dem
                    ).transpose(0,1)

                q=q_from_fK_batch(f,K)
                for _ in range(self.iters):
                    pc=path_cost_batch(q,K)
                    sh=path_shares_batch(pc)
                    f_new=edge_flow_batch(sh)
                    f=self.damping*f_new+(1.0-self.damping)*f
                    q=q_from_fK_batch(f,K)

                pc=path_cost_batch(q,K)
                out.append(
                    pc.transpose(0,1).detach().cpu().numpy()
                )                                   # b x P

        return np.concatenate(out,axis=0)


    def accessibility_from_path_cost_batch(self, path_cost_np: np.ndarray):
        """Vectorized candidate-path accessibility for B independent states.

        This is the batched counterpart of ``accessibility_from_path_cost`` and
        is used only for benchmark construction/training diagnostics. Reported
        policy outcomes continue to use ``accessibility_exact``.
        """
        pc=np.asarray(path_cost_np,float)
        if pc.ndim != 2 or pc.shape[1] != self.n_paths:
            raise ValueError(
                f"path_cost_np must have shape (B,{self.n_paths}), got {pc.shape}"
            )
        B=pc.shape[0]
        counts=self.od_path_count.detach().cpu().numpy().astype(int)
        c_od=np.full((B,self.n_od),np.inf,dtype=float)
        start=0
        for od,cnt in enumerate(counts):
            vals=pc[:,start:start+cnt]
            finite=np.isfinite(vals)
            safe=np.where(finite,vals,np.inf)
            c_od[:,od]=safe.min(axis=1)
            start+=cnt

        od_zone=self.od_zone.detach().cpu().numpy()
        O=self.od_O.detach().cpu().numpy()
        expo=np.zeros_like(c_od)
        np.exp(
            -self.kappa*c_od,
            where=np.isfinite(c_od),
            out=expo
        )
        contrib=expo*O[None,:]
        A=np.zeros((B,self.n_zones),dtype=float)
        # n_od is only ~5k; this avoids any dense OD-zone matrix.
        for od,z in enumerate(od_zone):
            A[:,int(z)]+=contrib[:,od]
        return A,c_od


    def audit_batched_surrogate_solver(
        self, K_states: np.ndarray,
        atol_path: float = 5e-4,
        rtol_path: float = 2e-6,
        atol_access: float = 5e-7,
        atol_loss: float = 5e-8,
        static=None,
    ):
        """Decision-relevant equivalence audit for scalar vs batched routing.

        Sparse matrix multiplication may accumulate float32 sums in a different
        order when multiple right-hand sides are evaluated simultaneously.
        Therefore bit-level equality of path costs is neither expected nor
        scientifically relevant. The audit requires:

        1. identical finite/infinite path support;
        2. path costs equal within a tight mixed absolute/relative tolerance;
        3. accessibility equal within ``atol_access``;
        4. when ``static`` is supplied, accessibility-loss and equity-loss
           functionals equal within ``atol_loss``.

        Reported policy outcomes are not affected by this tolerance because they
        continue to use exact full-network accessibility.
        """
        K_states=np.asarray(K_states,float)
        pcs=[]
        As=[]
        for K in K_states:
            pc=self.solve_path_cost_only(K)
            A,_=self.accessibility_from_path_cost(pc)
            pcs.append(pc)
            As.append(A)
        pcs=np.asarray(pcs,float)
        As=np.asarray(As,float)

        pcb=self.solve_path_cost_batch(
            K_states,batch_chunk=max(1,len(K_states))
        )
        Ab,_=self.accessibility_from_path_cost_batch(pcb)

        finite=np.isfinite(pcs)&np.isfinite(pcb)
        finite_pattern_equal=bool(
            np.array_equal(np.isfinite(pcs),np.isfinite(pcb))
        )

        if finite.any():
            absdiff=np.abs(pcs[finite]-pcb[finite])
            denom=np.maximum(np.abs(pcs[finite]),1.0)
            reldiff=absdiff/denom
            path_abs_err=float(absdiff.max())
            path_rel_err=float(reldiff.max())
            path_close=bool(np.all(
                absdiff <= (
                    float(atol_path)
                    + float(rtol_path)*np.abs(pcs[finite])
                )
            ))
            path_scale_median=float(np.median(np.abs(pcs[finite])))
            path_scale_p95=float(np.quantile(np.abs(pcs[finite]),0.95))
        else:
            path_abs_err=0.0
            path_rel_err=0.0
            path_close=True
            path_scale_median=float("nan")
            path_scale_p95=float("nan")

        access_abs=np.abs(As-Ab)
        access_err=float(access_abs.max()) if access_abs.size else 0.0
        access_close=bool(access_err<=float(atol_access))

        loss_acc_err=None
        loss_eq_err=None
        loss_close=True
        if static is not None:
            A0s=self.reference_accessibility_surrogate()
            Ls_acc=[]
            Ls_eq=[]
            Lb_acc=[]
            Lb_eq=[]
            for j in range(len(As)):
                _,la,le,_,_=loss_components(
                    As[j],A0s,static.population,static.vulnerable
                )
                _,lba,lbe,_,_=loss_components(
                    Ab[j],A0s,static.population,static.vulnerable
                )
                Ls_acc.append(float(la))
                Ls_eq.append(float(le))
                Lb_acc.append(float(lba))
                Lb_eq.append(float(lbe))
            loss_acc_err=float(
                np.max(np.abs(np.asarray(Ls_acc)-np.asarray(Lb_acc)))
            ) if Ls_acc else 0.0
            loss_eq_err=float(
                np.max(np.abs(np.asarray(Ls_eq)-np.asarray(Lb_eq)))
            ) if Ls_eq else 0.0
            loss_close=bool(
                loss_acc_err<=float(atol_loss)
                and loss_eq_err<=float(atol_loss)
            )

        passed=bool(
            finite_pattern_equal
            and path_close
            and access_close
            and loss_close
        )
        return {
            "passed":passed,
            "finite_pattern_equal":finite_pattern_equal,
            "max_abs_path_cost_error":path_abs_err,
            "max_rel_path_cost_error":path_rel_err,
            "median_abs_path_cost_scale":path_scale_median,
            "p95_abs_path_cost_scale":path_scale_p95,
            "path_close_mixed_tolerance":path_close,
            "max_abs_accessibility_error":access_err,
            "accessibility_close":access_close,
            "max_abs_Lacc_error":loss_acc_err,
            "max_abs_Leq_error":loss_eq_err,
            "loss_close":loss_close,
            "atol_path":float(atol_path),
            "rtol_path":float(rtol_path),
            "atol_access":float(atol_access),
            "atol_loss":float(atol_loss),
            "interpretation":(
                "The batch dimension changes float32 sparse-reduction order only. "
                "The audit is passed only when path support, path costs, "
                "accessibility, and decision losses agree within explicit "
                "numerical tolerances."
            ),
        }


    def accessibility_from_path_cost(self, path_cost_np: np.ndarray):
        # Fast candidate-path approximation retained only for routing diagnostics/debugging.
        # Reported accessibility uses accessibility_exact(), never this approximation.
        c_od = np.full(self.n_od, np.inf, dtype=float)
        start = 0
        counts = self.od_path_count.detach().cpu().numpy()
        for od, cnt in enumerate(counts):
            vals = path_cost_np[start:start+cnt]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                c_od[od] = finite.min()
            start += cnt
        A = np.zeros(self.n_zones, dtype=float)
        od_zone = self.od_zone.detach().cpu().numpy()
        O = self.od_O.detach().cpu().numpy()
        contrib = O * np.exp(-self.kappa * c_od, where=np.isfinite(c_od), out=np.zeros_like(c_od))
        np.add.at(A, od_zone, contrib)
        return A, c_od

    def reference_accessibility_surrogate(self):
        """Candidate-path reference denominator used only during learning."""
        pc = self.solve_path_cost_only(self.K0.detach().cpu().numpy())
        A, _ = self.accessibility_from_path_cost(pc)
        return A


# -----------------------------------------------------------------------------
# Hazard processing and scenario generation
# -----------------------------------------------------------------------------

def load_hazard_support(cfg, paths, edges, log):
    gpd, _, ox = require_geospatial()
    manifest = read_json(paths.manifests / "data_manifest.json")
    hinfo = manifest["sources"]["flood_hazard"]
    if not hinfo.get("proxy", False) and hinfo.get("vector_path"):
        vp = Path(hinfo["vector_path"])
        # Backward compatibility with v1.0.2 manifests that recorded the raw
        # extensionless Geo-IDE payload itself as vector_path.
        if vp.exists() and vp.suffix.lower() == ".bin":
            resolved = resolve_vector_payload(
                vp,
                paths.raw / "tri_lille_hazard_unpacked",
                log,
            )
            if resolved is not None:
                vp = resolved
                hinfo["vector_path"] = str(vp)
                manifest["sources"]["flood_hazard"] = hinfo
                write_json(paths.manifests / "data_manifest.json", manifest)
        try:
            hz = gpd.read_file(vp)
            if hz.empty:
                raise RuntimeError("empty hazard layer")
            hz = filter_hazard_layer_to_tri_lille(
                hz,
                cfg["data"].get("hazard_tri_id", "59DREAL20140002"),
                log,
            )
            if hz.empty:
                raise RuntimeError("hazard layer is empty after TRI Lille filtering")
            hz = hz.to_crs(cfg["project"]["crs_metric"])
            support = union_geometry(hz)
            e = edges.to_crs(cfg["project"]["crs_metric"]).copy()
            e["hazard_exposed"] = e.geometry.intersects(support)
            log.log(f"TRI Lille hazard support intersects {e.hazard_exposed.sum():,} directed edges")
            return e.hazard_exposed.to_numpy(bool), "DREAL_TRI_Lille"
        except Exception as exc:
            if not cfg["data"].get("hazard_fallback_proxy", False):
                raise
            log.log(f"WARNING: hazard layer unreadable; proxy enabled: {exc}")
    # Explicit proxy: OSM waterways within MEL.
    boundary = gpd.read_file(paths.raw / "mel_boundary.geojson").to_crs(4326)
    geom = union_geometry(boundary)
    water = ox.features_from_polygon(geom, {"waterway": True}).reset_index()
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=4326).to_crs(cfg["project"]["crs_metric"])
    buffered = water.geometry.buffer(250)
    support = buffered.union_all() if hasattr(buffered, "union_all") else buffered.unary_union
    e = edges.to_crs(cfg["project"]["crs_metric"])
    return e.geometry.intersects(support).to_numpy(bool), "DEBUG_PROXY_OSM_WATERWAY"


def generate_disruptions(cfg, paths, log):
    gpd, _, _ = require_geospatial()
    edges = gpd.read_parquet(paths.processed / "network_edges.parquet")
    exposed, hazard_mode = load_hazard_support(cfg, paths, edges, log)
    exposed_ids = edges.loc[exposed, "edge_id"].to_numpy(int)
    if exposed_ids.size == 0:
        raise RuntimeError("No network edge intersects the hazard support.")
    mids = edges.geometry.interpolate(0.5, normalized=True)
    xy = np.column_stack([mids.x.to_numpy(), mids.y.to_numpy()])
    severities = list(map(float, cfg["disruptions"]["severity_levels"]))
    extents = list(map(float, cfg["disruptions"]["spatial_extent_m"]))
    profiles = cfg["disruptions"]["profiles"]
    T = int(cfg["disruptions"]["horizon_T"])
    for name, prof in profiles.items():
        if len(prof) != T + 1:
            raise ValueError(f"Disruption profile '{name}' must contain T+1={T+1} values")
    hold = cfg["disruptions"]["structural_holdout"]
    rng = np.random.default_rng(cfg["compute"]["seeds"][0])

    def draw_one(sid: int, split: str, structural: bool):
        center_e = int(rng.choice(exposed_ids))
        if structural:
            sev = float(hold["severity"]); extent = float(hold["extent_m"])
        else:
            combos = [(s,e) for s in severities for e in extents if not (s == float(hold["severity"]) and e == float(hold["extent_m"]))]
            sev, extent = combos[int(rng.integers(len(combos)))]
        pname = str(rng.choice(list(profiles.keys())))
        prof = np.asarray(profiles[pname], float)
        cxy = xy[center_e]
        dist = np.sqrt(np.sum((xy - cxy[None,:])**2, axis=1))
        spatial = np.exp(-0.5 * (dist / max(extent, 1.0))**2)
        # Restrict perturbation to edges plausibly related to hazard support but allow nearby network cascade initiation.
        spatial *= np.where(exposed, 1.0, 0.35)
        d = np.clip(sev * prof[:,None] * spatial[None,:], 0.0, 1.0)
        # Severe center-edge outage at peak if severity=1.
        if sev >= 0.999:
            d[0, center_e] = 1.0
        return {
            "scenario_id": sid, "split": split, "structural": structural,
            "center_edge": center_e, "severity": sev, "extent_m": extent,
            "profile": pname, "hazard_mode": hazard_mode,
            "degradation": d.astype(np.float32)
        }

    specs = []
    sid = 0
    for split, n in [("train", cfg["disruptions"]["n_train"]), ("validation", cfg["disruptions"]["n_validation"]), ("test", cfg["disruptions"]["n_test"])]:
        for _ in range(int(n)):
            specs.append(draw_one(sid, split, False)); sid += 1
    for _ in range(int(cfg["disruptions"]["n_structural_test"])):
        specs.append(draw_one(sid, "test_structural", True)); sid += 1

    meta_rows = []
    for s in specs:
        np.savez_compressed(paths.scenarios / f"scenario_{s['scenario_id']:05d}.npz", degradation=s["degradation"])
        meta_rows.append({k:v for k,v in s.items() if k != "degradation"})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(paths.scenarios / "scenario_manifest.csv", index=False)
    # leakage audit: structural combination absent from train/val.
    bad = meta[(meta.split.isin(["train","validation"])) & (meta.severity == float(hold["severity"])) & (meta.extent_m == float(hold["extent_m"]))]
    audit = pd.DataFrame([
        {"check":"train_validation_test_disjoint_ids", "pass": meta.scenario_id.is_unique},
        {"check":"structural_holdout_absent_train_validation", "pass": len(bad)==0},
        {"check":"hazard_proxy_disabled_for_publication", "pass": hazard_mode != "DEBUG_PROXY_OSM_WATERWAY"},
    ])
    audit.to_csv(paths.processed / "leakage_audit_scenarios.csv", index=False)
    log.log(f"Generated {len(meta)} disruption trajectories; hazard_mode={hazard_mode}")
    return meta


# -----------------------------------------------------------------------------
# Environment, CAR, frozen-load diagnostic, interventions
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class EnvStatic:
    K0: np.ndarray
    q0: np.ndarray
    f0: np.ndarray
    population: np.ndarray
    vulnerable: np.ndarray
    A0: np.ndarray
    gamma: float
    edge_length: np.ndarray
    betweenness: np.ndarray


def cvar_empirical(x: np.ndarray, alpha: float) -> float:
    x = np.asarray(x, float)
    if x.size == 0: return float("nan")
    eta = float(np.quantile(x, alpha, method="higher" if hasattr(np, "quantile") else "linear"))
    return eta + np.maximum(x - eta, 0.0).mean() / max(1e-12, 1-alpha)


def loss_components(A: np.ndarray, A0: np.ndarray, pop: np.ndarray, vuln: np.ndarray):
    ell = np.maximum(1.0 - A / np.maximum(A0, 1e-12), 0.0)
    w = pop / pop.sum()
    Lacc = float(np.dot(w, ell))
    pv = pop[vuln].sum(); pc = pop[~vuln].sum()
    Lv = float(np.dot(pop[vuln], ell[vuln]) / pv) if pv > 0 else 0.0
    Lc = float(np.dot(pop[~vuln], ell[~vuln]) / pc) if pc > 0 else 0.0
    Leq = max(Lv - Lc, 0.0)
    return ell, Lacc, Leq, Lv, Lc


def intervention_costs(cfg, edges: pd.DataFrame):
    length = np.maximum(edges.geometry.length.to_numpy(float), 1.0)
    med = np.median(length)
    power = float(cfg["intervention"]["edge_cost_length_power"])
    costs = float(cfg["intervention"]["edge_cost_scale"]) * np.power(length / med, power)
    return costs


def baseline_budget(cfg, costs):
    return float(cfg["intervention"]["reference_budget_median_actions"]) * float(np.median(costs)) * float(cfg["intervention"]["budget_multiplier_baseline"])


def feasible_edges(Kpre, K0, costs, budget, max_candidates, priority_hint=None):
    idx = np.where((Kpre < K0 - 1e-9) & (costs <= budget + 1e-12))[0]
    if idx.size <= max_candidates:
        return idx
    deficit = np.maximum(K0[idx] - Kpre[idx], 0.0) / np.maximum(K0[idx], 1e-9)
    score = deficit.copy()
    if priority_hint is not None:
        ph = np.asarray(priority_hint)[idx]
        ph = (ph - np.nanmin(ph)) / (np.nanmax(ph)-np.nanmin(ph)+1e-12)
        score += 0.25 * ph
    return idx[np.argsort(score)[-max_candidates:]]


def policy_candidate_edges(cfg, engine, Kpre, K0, costs, budget, max_candidates=None):
    """Final pre-action candidate correspondence used by all learned policies.

    Eligible edges must:
      (i) have positive residual capacity deficit,
      (ii) be affordable under the remaining hard budget, and
      (iii) belong to at least one prespecified service-access candidate path.

    R3 ranks eligible edges by
        (K0_e - Kpre_e)_+ * path_use_count_e
    and retains the deterministic top-K set. Ties are broken by ascending edge
    index. No post-action accessibility outcome enters the screening rule.
    """
    rule = cfg["intervention"].get(
        "candidate_screening_rule",
        "r3_abs_deficit_x_candidate_path_use"
    )
    if rule != "r3_abs_deficit_x_candidate_path_use":
        raise RuntimeError(f"Unsupported final candidate screening rule: {rule}")

    Kcap = int(
        max_candidates
        if max_candidates is not None
        else cfg["intervention"]["max_candidate_edges_per_step"]
    )
    tol = float(cfg["intervention"].get("candidate_deficit_tolerance", 1e-12))
    Kpre = np.asarray(Kpre, float)
    K0 = np.asarray(K0, float)
    costs = np.asarray(costs, float)
    path_use = np.asarray(engine.path_use_count, dtype=np.int64)

    abs_def = np.maximum(K0-Kpre, 0.0)
    eligible = np.where(
        (abs_def > tol)
        & (costs <= float(budget) + 1e-12)
        & (path_use > 0)
    )[0].astype(np.int64)

    if eligible.size == 0:
        return eligible

    score = abs_def[eligible] * path_use[eligible].astype(float)
    # Descending R3 score, deterministic ascending edge-id tie break.
    order = np.lexsort((eligible, -np.nan_to_num(score, nan=-np.inf)))
    return eligible[order[:min(Kcap, eligible.size)]]


def build_reference_engine(cfg, paths, log):
    G, zones, services, edges = prepare_reference_graph(cfg, paths, log)
    pairs, path_list = build_route_choice_set(cfg, paths, G, zones, services, edges, log)
    torch = require_torch()
    use_cuda = bool(cfg["compute"]["prefer_cuda"] and torch.cuda.is_available())
    device = f"cuda:{cfg['compute']['cuda_device']}" if use_cuda else "cpu"
    engine = RouteEngine(cfg, edges, pairs, path_list, device=device)
    engine.attach_exact_accessibility(G, zones, services, edges)
    err = engine.calibrate_reference()
    share_err,share_mass_err=engine.audit_vectorized_path_shares()
    log.log(
        f"Vectorized routing audit: max share error vs legacy OD loop={share_err:.3e}; "
        f"max OD probability-mass error={share_mass_err:.3e}"
    )
    if share_err>1e-6 or share_mass_err>1e-6:
        raise RuntimeError("Vectorized OD route-choice audit failed; refusing to continue.")
    tol = float(cfg["network"]["reference_consistency_tolerance"])
    if err > tol:
        raise RuntimeError(
            f"Reference consistency failed: max|Q(f0,K0)-q0|={err:.3e} > {tol:.3e}. "
            "The reference normalization is required to hold numerically; do not "
            "relax the tolerance to hide a floating-point evaluation-order error."
        )
    # Reference accessibility from reference path costs.
    _, qref, _ = engine.solve(edges.K0_vph.to_numpy(float))
    A0_model = engine.accessibility_exact(qref, edges.K0_vph.to_numpy(float))
    # Align zone arrays to 0..n_zones-1; zones may have gaps after reduction, remap if needed.
    zone_ids = np.sort(zones.zone_id.unique())
    if not np.array_equal(zone_ids, np.arange(len(zone_ids))):
        # Reindex and rebuild path data if necessary; this should not happen because reduced data preserves original ids.
        raise RuntimeError("Model zone_id values must be contiguous. Re-run process stage with current script.")
    pop = zones.sort_values("zone_id").P_i.to_numpy(float)
    vuln = zones.sort_values("zone_id").is_vulnerable.astype(bool).to_numpy()
    static = EnvStatic(
        K0=edges.K0_vph.to_numpy(float), q0=edges.q0_sec.to_numpy(float), f0=engine.f0_cpu,
        population=pop, vulnerable=vuln, A0=A0_model,
        gamma=float(cfg["accessibility"]["gamma"]), edge_length=edges.geometry.length.to_numpy(float),
        betweenness=np.zeros(len(edges), dtype=float)
    )
    np.savez_compressed(paths.processed / "reference_engine_arrays.npz", K0=static.K0, q0=static.q0, f0=static.f0, population=pop, vulnerable=vuln.astype(np.int8), A0=A0_model)
    pd.DataFrame({"metric":["reference_consistency_max_abs_error","device","kappa"], "value":[err,device,engine.kappa]}).to_csv(paths.processed / "reference_calibration_audit.csv", index=False)
    log.log(f"Reference engine calibrated on {device}; identity error={err:.2e}")
    return engine, static, edges, zones, services


def edge_betweenness(cfg, paths, G, edges, log):
    """Approximate directed edge betweenness with visible progress and restartable checkpoints.

    This implements the same sampled-source Brandes calculation used by
    ``networkx.edge_betweenness_centrality(..., k=..., normalized=False,
    weight="travel_time")`` but exposes the source loop. The final sampled
    counts are inflated by n/k, exactly as NetworkX does for directed graphs.
    A checkpoint stores the accumulated unscaled pair-edge scores and the number
    of completed sampled sources, so an interrupted baseline run can resume.
    """
    nx, _ = require_network()
    out_path = paths.processed / "edge_betweenness.csv"
    checkpoint_path = paths.processed / "edge_betweenness_checkpoint.npz"

    if out_path.exists():
        arr = pd.read_csv(out_path).sort_values("edge_id").betweenness.to_numpy(float)
        if len(arr) != len(edges):
            raise RuntimeError(
                f"Cached edge_betweenness.csv has {len(arr):,} rows but "
                f"the current network has {len(edges):,} directed edges."
            )
        log.log(f"Reuse cached edge betweenness: {len(arr):,} directed edges")
        return arr

    from networkx.algorithms.centrality import betweenness as _nx_btw

    if not hasattr(_nx_btw, "_single_source_dijkstra_path_basic") or not hasattr(_nx_btw, "_accumulate_edges"):
        raise RuntimeError(
            "Installed NetworkX does not expose the Brandes helpers required for "
            "progress-aware betweenness. Use a compatible NetworkX 3.x release."
        )

    n = G.number_of_nodes()
    k = min(int(cfg["network"].get("betweenness_sample_sources", 400)), n)
    if k <= 0:
        raise RuntimeError("betweenness_sample_sources must be strictly positive.")

    progress_every = max(1, int(cfg["network"].get("betweenness_progress_every", 5)))
    checkpoint_every = max(1, int(cfg["network"].get("betweenness_checkpoint_every", 10)))
    seed = int(cfg["compute"]["seeds"][0])

    # Match NetworkX's deterministic sampled-source design for an integer seed.
    rng = random.Random(seed)
    sampled_sources = rng.sample(list(G.nodes()), k) if k < n else list(G.nodes())

    # NetworkX accumulates multigraph scores first on (u,v) pairs, then assigns
    # the value among equal-weight parallel edges. Keep exactly that structure.
    pair_edges = list(G.edges())
    betweenness = dict.fromkeys(G, 0.0)
    betweenness.update(dict.fromkeys(pair_edges, 0.0))

    completed = 0
    if checkpoint_path.exists():
        try:
            chk = np.load(checkpoint_path, allow_pickle=False)
            chk_n = int(chk["n_nodes"])
            chk_m = int(chk["n_pair_edges"])
            chk_k = int(chk["k"])
            chk_seed = int(chk["seed"])
            chk_completed = int(chk["completed"])
            vals = np.asarray(chk["pair_values"], dtype=float)

            compatible = (
                chk_n == n
                and chk_m == len(pair_edges)
                and chk_k == k
                and chk_seed == seed
                and len(vals) == len(pair_edges)
                and 0 <= chk_completed <= k
            )
            if compatible:
                for key, val in zip(pair_edges, vals):
                    betweenness[key] = float(val)
                completed = chk_completed
                log.log(
                    f"Resume edge betweenness checkpoint: {completed:,}/{k:,} "
                    f"sampled sources already complete"
                )
            else:
                log.log(
                    "Ignore incompatible edge betweenness checkpoint "
                    "(network/sample design changed)"
                )
        except Exception as exc:
            log.log(
                f"Ignore unreadable edge betweenness checkpoint: "
                f"{type(exc).__name__}: {exc}"
            )

    log.log(
        f"Compute directed weighted edge betweenness with visible progress: "
        f"{k:,} sampled sources / {n:,} graph nodes, "
        f"{len(pair_edges):,} directed node-pair edges; "
        f"progress every {progress_every}, checkpoint every {checkpoint_every}"
    )

    t0 = time.perf_counter()
    # The elapsed/ETA clock below refers to work performed in this invocation.
    done_this_run = 0

    for pos in range(completed, k):
        s = sampled_sources[pos]
        source_t0 = time.perf_counter()

        S, P, sigma, _ = _nx_btw._single_source_dijkstra_path_basic(
            G, s, "travel_time"
        )
        betweenness = _nx_btw._accumulate_edges(
            betweenness, S, P, sigma, s
        )

        done_this_run += 1
        done_total = pos + 1
        source_elapsed = time.perf_counter() - source_t0

        if (
            done_total == 1
            or done_total % progress_every == 0
            or done_total == k
        ):
            elapsed = time.perf_counter() - t0
            rate = done_this_run / max(elapsed, 1e-12)
            remaining = k - done_total
            eta = remaining / max(rate, 1e-12)
            log.log(
                f"Betweenness: {done_total:,}/{k:,} "
                f"({100.0*done_total/max(k,1):5.1f}%) | "
                f"last source={source_elapsed:.1f}s | "
                f"rate={rate:.3f} sources/s | "
                f"elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min"
            )

        if done_total % checkpoint_every == 0 and done_total < k:
            vals = np.fromiter(
                (float(betweenness[e]) for e in pair_edges),
                dtype=np.float64,
                count=len(pair_edges),
            )
            tmp = checkpoint_path.with_suffix(".npz.tmp")
            # np.savez_compressed appends .npz when given a string. Use a file
            # handle so the temporary filename remains exactly controlled.
            with tmp.open("wb") as fh:
                np.savez_compressed(
                    fh,
                    n_nodes=np.int64(n),
                    n_pair_edges=np.int64(len(pair_edges)),
                    k=np.int64(k),
                    seed=np.int64(seed),
                    completed=np.int64(done_total),
                    pair_values=vals,
                )
            tmp.replace(checkpoint_path)
            log.log(
                f"Betweenness checkpoint saved: {done_total:,}/{k:,} sources"
            )

    # Remove node entries, retaining only pair-edge scores.
    pair_bc = {e: float(betweenness[e]) for e in pair_edges}

    # NetworkX's sampled, unnormalized directed edge betweenness multiplies
    # partial source sums by n/k to approximate the all-source quantity.
    scale = float(n) / float(k)
    if scale != 1.0:
        for e in pair_bc:
            pair_bc[e] *= scale

    # For MultiDiGraph, reproduce NetworkX's parallel-edge allocation among
    # equal minimum-weight edges.
    if G.is_multigraph():
        if not hasattr(_nx_btw, "_add_edge_keys"):
            raise RuntimeError(
                "Installed NetworkX lacks _add_edge_keys required for "
                "MultiDiGraph edge-betweenness allocation."
            )
        bc = _nx_btw._add_edge_keys(G, pair_bc, weight="travel_time")
    else:
        bc = pair_bc

    lut = edge_lookup_from_edges(edges)
    arr = np.zeros(len(edges), float)

    # Precompute the cheapest parallel-edge fallback once, avoiding repeated
    # DataFrame filtering in the final mapping.
    cheapest = {}
    if not G.is_multigraph():
        for r in edges.sort_values("q0_sec").itertuples():
            cheapest.setdefault((str(r.u), str(r.v)), int(r.edge_id))

    unmatched = 0
    for key, val in bc.items():
        if len(key) == 3:
            eid = lut.get((str(key[0]), str(key[1]), str(key[2])))
        else:
            eid = cheapest.get((str(key[0]), str(key[1])))
        if eid is not None:
            arr[eid] += float(val)
        else:
            unmatched += 1

    if unmatched:
        log.log(
            f"Betweenness mapping warning: {unmatched:,} NetworkX edge entries "
            "could not be matched to processed edge_id values"
        )

    pd.DataFrame({
        "edge_id": np.arange(len(arr)),
        "betweenness": arr
    }).to_csv(out_path, index=False)

    with contextlib.suppress(FileNotFoundError):
        checkpoint_path.unlink()

    total_elapsed = time.perf_counter() - t0
    log.log(
        f"Directed edge betweenness complete: {k:,} sampled sources, "
        f"{len(arr):,} processed edges, elapsed this run={total_elapsed/60:.1f} min; "
        f"saved {out_path.name}"
    )
    return arr




def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}min"
    return f"{minutes/60.0:.2f}h"


def _progress_message(label: str, done: int, total: int, started: float, last_seconds=None) -> str:
    elapsed = max(time.perf_counter() - started, 1e-12)
    rate = done / elapsed if done > 0 else 0.0
    eta = (max(0, total-done) / rate) if rate > 0 else float("inf")
    eta_txt = "n/a" if not np.isfinite(eta) else _fmt_duration(eta)
    last = "" if last_seconds is None else f" | last={_fmt_duration(last_seconds)}"
    return (f"{label}: {done:,}/{total:,} ({100.0*done/max(total,1):5.1f}%)"
            f"{last} | elapsed={_fmt_duration(elapsed)} | ETA={eta_txt}")


def _atomic_dataframe_checkpoint(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)



def load_or_compute_betweenness(cfg, paths, edges, log):
    """Load cached edge betweenness without reloading the 328k-node GraphML.

    Only falls back to graph loading/recomputation when the cache is absent.
    """
    out_path = paths.processed / "edge_betweenness.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        if not {"edge_id", "betweenness"}.issubset(df.columns):
            raise RuntimeError("Cached edge_betweenness.csv has invalid columns.")
        df = df.sort_values("edge_id")
        arr = df["betweenness"].to_numpy(float)
        if len(arr) != len(edges):
            raise RuntimeError(
                f"Cached edge betweenness has {len(arr):,} rows but "
                f"current network has {len(edges):,} edges."
            )
        log.log(
            f"Reuse cached edge betweenness directly: {len(arr):,} directed edges "
            "(GraphML reload skipped)"
        )
        return arr
    G, *_ = prepare_reference_graph(cfg, paths, log)
    return edge_betweenness(cfg, paths, G, edges, log)



def evaluate_uncontrolled_scenario(cfg, engine: RouteEngine, static: EnvStatic, degradation: np.ndarray, frozen_load: bool = True):
    T = degradation.shape[0]-1
    rows=[]; Z=0.0; Zfl=0.0
    for t in range(T+1):
        K = (1.0 - degradation[t]) * static.K0
        f, q, pc = engine.solve(K)
        A = engine.accessibility_exact(q, K)
        _, Lacc, Leq, Lv, Lc = loss_components(A, static.A0, static.population, static.vulnerable)
        Z += (static.gamma**t)*Lacc
        if frozen_load:
            # q_FL = Q(f0,K) with same physical degradation. Use engine normalization.
            torch = engine.torch
            Kt = torch.tensor(K, dtype=engine.dtype, device=engine.device)
            qfl = engine._q_from_fK(engine.f0, Kt)
            qfl_np = qfl.detach().cpu().numpy()
            Afl = engine.accessibility_exact(qfl_np, K)
            _, Lfl, _, _, _ = loss_components(Afl, static.A0, static.population, static.vulnerable)
            Zfl += (static.gamma**t)*Lfl
        rows.append({"t":t,"Lacc":Lacc,"Leq":Leq,"Lv":Lv,"Lcomp":Lc})
    return Z, Zfl, rows


def baseline_stage(cfg, paths, log):
    engine, static, edges, zones, services = build_reference_engine(cfg, paths, log)
    G, *_ = prepare_reference_graph(cfg, paths, log)
    bc = edge_betweenness(cfg, paths, G, edges, log)
    static.betweenness = bc
    meta = pd.read_csv(paths.scenarios / "scenario_manifest.csv")
    pcfg = cfg.get("progress", {})

    # FULL/frozen-load baseline: visible and restartable.
    final_path = paths.eval / "uncontrolled_cascade.parquet"
    fig3_path = paths.figure_data / "fig3_cascade_decomposition.csv"
    ckpt_path = paths.eval / "uncontrolled_cascade_checkpoint.csv"
    if final_path.exists() and fig3_path.exists():
        out = pd.read_parquet(final_path)
        log.log(f"Reuse completed uncontrolled evaluation: {len(out):,} scenarios")
    else:
        rows=[]; completed=set()
        if ckpt_path.exists():
            try:
                prev=pd.read_csv(ckpt_path)
                rows=prev.to_dict("records")
                completed=set(prev.scenario_id.astype(int))
                log.log(f"Resume uncontrolled checkpoint: {len(completed):,}/{len(meta):,} scenarios")
            except Exception as exc:
                log.log(f"Ignore unreadable uncontrolled checkpoint: {type(exc).__name__}: {exc}")
                rows=[]; completed=set()
        todo=[r for r in meta.itertuples() if int(r.scenario_id) not in completed]
        every=max(1,int(pcfg.get("baseline_scenario_every",5)))
        ck_every=max(1,int(pcfg.get("baseline_checkpoint_every",5)))
        log.log(f"Evaluate uncontrolled FULL and frozen-load trajectories: {len(meta):,} total, {len(todo):,} remaining")
        started=time.perf_counter()
        for pos,r in enumerate(todo,start=1):
            one=time.perf_counter(); sid=int(r.scenario_id)
            deg=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
            Z,Zfl,_=evaluate_uncontrolled_scenario(cfg,engine,static,deg,True)
            rows.append({"scenario_id":sid,"split":r.split,"structural":bool(r.structural),
                         "severity":float(r.severity),"extent_m":float(r.extent_m),
                         "profile":r.profile,"Z_FULL":Z,"Z_FL":Zfl,"Delta_flow":Z-Zfl})
            done=len(completed)+pos
            if done==1 or done%every==0 or done==len(meta):
                log.log(_progress_message("Baseline uncontrolled scenarios",done,len(meta),started,time.perf_counter()-one))
            if done%ck_every==0 and done<len(meta):
                ck=pd.DataFrame(rows).drop_duplicates("scenario_id",keep="last").sort_values("scenario_id")
                _atomic_dataframe_checkpoint(ck,ckpt_path)
                log.log(f"Uncontrolled checkpoint saved: {len(ck):,}/{len(meta):,}")
        out=pd.DataFrame(rows).drop_duplicates("scenario_id",keep="last").sort_values("scenario_id")
        if len(out)!=len(meta): raise RuntimeError(f"Uncontrolled baseline incomplete: {len(out)}/{len(meta)}")
        out.to_parquet(final_path,index=False); out.to_csv(fig3_path,index=False)
        with contextlib.suppress(FileNotFoundError): ckpt_path.unlink()
        log.log(f"Uncontrolled FULL/frozen-load complete: {len(out):,} scenarios")

    # Training-only CAR feature: visible and restartable via sufficient-statistic checkpoint.
    car_path=paths.processed/"training_car_node_feature.csv"
    car_ckpt=paths.processed/"training_car_node_feature_checkpoint.npz"
    train_ids=meta.loc[meta.split=="train","scenario_id"].astype(int).tolist()
    if car_path.exists():
        car_df=pd.read_csv(car_path)
        if len(car_df)!=len(static.population): raise RuntimeError("Cached training CAR zone count mismatch")
        log.log(f"Reuse completed training CAR feature: {len(train_ids):,} scenarios")
    else:
        car_sum=np.zeros_like(static.population,float); done_ids=[]
        if car_ckpt.exists():
            try:
                chk=np.load(car_ckpt,allow_pickle=False)
                saved=np.asarray(chk["car_sum"],float); ids=np.asarray(chk["completed_scenario_ids"],int)
                if saved.shape==car_sum.shape and set(ids.tolist()).issubset(set(train_ids)):
                    car_sum=saved; done_ids=ids.tolist()
                    log.log(f"Resume training CAR checkpoint: {len(done_ids):,}/{len(train_ids):,}")
            except Exception as exc:
                log.log(f"Ignore unreadable training CAR checkpoint: {type(exc).__name__}: {exc}")
        done_set=set(done_ids); todo=[sid for sid in train_ids if sid not in done_set]
        every=max(1,int(pcfg.get("training_car_every",5)))
        ck_every=max(1,int(pcfg.get("training_car_checkpoint_every",5)))
        log.log(f"Compute training-only CAR node feature: {len(train_ids):,} total, {len(todo):,} remaining")
        started=time.perf_counter()
        for pos,sid in enumerate(todo,start=1):
            one=time.perf_counter()
            deg=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
            car_i=np.zeros_like(static.population,float)
            for t in range(deg.shape[0]):
                K=(1-deg[t])*static.K0
                _,qcur,_=engine.solve(K); A=engine.accessibility_exact(qcur,K)
                car_i+=(static.gamma**t)*np.maximum(1-A/np.maximum(static.A0,1e-12),0)
            car_sum+=car_i; done_ids.append(sid)
            done=len(done_ids)
            if done==1 or done%every==0 or done==len(train_ids):
                log.log(_progress_message("Training CAR scenarios",done,len(train_ids),started,time.perf_counter()-one))
            if done%ck_every==0 and done<len(train_ids):
                tmp=car_ckpt.with_suffix(".npz.tmp")
                with tmp.open("wb") as fh:
                    np.savez_compressed(fh,car_sum=car_sum,completed_scenario_ids=np.asarray(done_ids,dtype=np.int64))
                tmp.replace(car_ckpt); log.log(f"Training CAR checkpoint saved: {done:,}/{len(train_ids):,}")
        if len(done_ids)!=len(train_ids): raise RuntimeError("Training CAR incomplete")
        car=car_sum/max(len(train_ids),1)
        pd.DataFrame({"zone_id":np.arange(len(car)),"CAR_i_tr0":car}).to_csv(car_path,index=False)
        with contextlib.suppress(FileNotFoundError): car_ckpt.unlink()
        log.log(f"Training-only CAR feature complete: {len(train_ids):,} scenarios, {len(car):,} zones")
    log.log("Baseline cascade diagnostics complete")
    return engine, static, edges


# -----------------------------------------------------------------------------
# Policy environment and baselines
# -----------------------------------------------------------------------------

def trajectory_rollout(cfg, engine, static, edges, degradation, policy_fn, seed: int, return_steps: bool = False):
    rng=np.random.default_rng(seed)
    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    budget=B
    restore=np.zeros_like(static.K0,float)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    T=degradation.shape[0]-1
    Z=Q=0.0; step_rows=[]
    for t in range(T+1):
        if t>0:
            restore *= retention
        Kexo=(1-degradation[t])*static.K0
        deficit=np.maximum(static.K0-(Kexo+restore),0)
        Kpre=np.minimum(Kexo+restore, static.K0)
        feas=policy_candidate_edges(cfg,engine,Kpre,static.K0,costs,budget)
        # Pre-action state for policy; no post-action outcome included.
        state={"t":t,"budget":budget,"B":B,"Kpre":Kpre.copy(),"Kexo":Kexo.copy(),"restore":restore.copy(),"feasible":feas.copy(),"costs":costs,"degradation":degradation[t].copy(),"rng":rng}
        action=int(policy_fn(state)) if len(feas) else -1
        if action >= 0:
            if action not in set(map(int,feas)):
                raise RuntimeError(f"Policy selected infeasible edge {action} at t={t}")
            inc=frac*deficit[action]
            restore[action]+=inc
            budget-=costs[action]
        Kctl=np.minimum(Kexo+restore,static.K0)
        f,q,pc=engine.solve(Kctl); A=engine.accessibility_exact(q,Kctl)
        ell,Lacc,Leq,Lv,Lc=loss_components(A,static.A0,static.population,static.vulnerable)
        Z+=(static.gamma**t)*Lacc; Q+=(static.gamma**t)*Leq
        step_rows.append({"t":t,"action":action,"budget":budget,"Lacc":Lacc,"Leq":Leq,"Lv":Lv,"Lcomp":Lc,"n_feasible":len(feas)})
    return (Z,Q,step_rows) if return_steps else (Z,Q)


def trajectory_rollout_surrogate(cfg, engine, static, edges, degradation, policy_fn, seed: int):
    """Fast rollout for benchmark construction only.

    This uses the fixed candidate-path accessibility approximation for the
    objective used to CONSTRUCT B2/B3/B4. It never replaces exact full-network
    accessibility in reported policy evaluation.

    The intervention dynamics, budget, candidate correspondence, routing fixed
    point, restoration persistence, and equity definition are unchanged.
    """
    rng=np.random.default_rng(seed)
    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    budget=B
    restore=np.zeros_like(static.K0,float)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    T=degradation.shape[0]-1
    A0s=engine.reference_accessibility_surrogate()
    Z=Q=0.0
    for t in range(T+1):
        if t>0:
            restore *= retention
        Kexo=(1-degradation[t])*static.K0
        deficit=np.maximum(static.K0-(Kexo+restore),0)
        Kpre=np.minimum(Kexo+restore,static.K0)
        feas=policy_candidate_edges(cfg,engine,Kpre,static.K0,costs,budget)
        state={
            "t":t,"budget":budget,"B":B,"Kpre":Kpre.copy(),"Kexo":Kexo.copy(),
            "restore":restore.copy(),"feasible":feas.copy(),"costs":costs,
            "degradation":degradation[t].copy(),"rng":rng
        }
        action=int(policy_fn(state)) if len(feas) else -1
        if action>=0:
            if action not in set(map(int,feas)):
                raise RuntimeError(f"Policy selected infeasible edge {action} at t={t}")
            inc=frac*deficit[action]
            restore[action]+=inc
            budget-=costs[action]
        Kctl=np.minimum(Kexo+restore,static.K0)
        pc=engine.solve_path_cost_only(Kctl)
        A=engine.accessibility_from_path_cost(pc)[0]
        _,Lacc,Leq,_,_=loss_components(
            A,A0s,static.population,static.vulnerable
        )
        Z+=(static.gamma**t)*Lacc
        Q+=(static.gamma**t)*Leq
    return Z,Q


def _atomic_csv(df, path):
    tmp=Path(str(path)+".tmp")
    df.to_csv(tmp,index=False)
    tmp.replace(path)


def _load_partial_b3_checkpoint(path, candidate_edges, protocol_tag):
    if not path.exists():
        return {}
    df=pd.read_csv(path)
    if "protocol_tag" not in df.columns or not (df.protocol_tag.astype(str)==protocol_tag).all():
        return {}
    wanted=set(map(int,candidate_edges))
    out={}
    for r in df.itertuples(index=False):
        e=int(r.edge_id)
        if e in wanted:
            out[e]={
                "edge_id":e,
                "CAR0_train_surrogate":float(r.CAR0_train_surrogate),
                "CAR_e_train_surrogate":float(r.CAR_e_train_surrogate),
                "S_CAR":float(r.S_CAR),
                "protocol_tag":str(r.protocol_tag),
            }
    return out


def b0_policy_factory(seed):
    rng=np.random.default_rng(seed)
    def pol(s):
        feas=s["feasible"]
        if len(feas)==0: return -1
        # Explicit no-action option with 15% probability; avoids forced intervention.
        if rng.random()<0.15: return -1
        return int(rng.choice(feas))
    return pol


def b1_policy_factory(betweenness):
    def pol(s):
        feas=s["feasible"]
        if len(feas)==0: return -1
        vals=betweenness[feas]
        if np.nanmax(vals)<=0: return -1
        return int(feas[np.nanargmax(vals)])
    return pol


def b2_policy_factory(cfg,engine,static,edges):
    """Myopic accessibility-loss benchmark.

    Candidate actions are ranked by the contemporaneous reduction in the
    candidate-path accessibility loss. The chosen action is subsequently
    evaluated with exact full-network accessibility in evaluate_all().
    """
    A0s=engine.reference_accessibility_surrogate()
    def pol(s):
        feas=s["feasible"]
        if len(feas)==0:
            return -1
        Kpre=s["Kpre"]
        pc0=engine.solve_path_cost_only(Kpre)
        Abase=engine.accessibility_from_path_cost(pc0)[0]
        _,Lbase,_,_,_=loss_components(
            Abase,A0s,static.population,static.vulnerable
        )
        best_gain=0.0
        best=-1
        frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
        # The candidate correspondence is already R3/Top-160. This loop only
        # ranks those admissible actions; no exact Dijkstra is used for action
        # selection.
        for e in feas:
            K=Kpre.copy()
            K[e]=min(static.K0[e],K[e]+frac*(static.K0[e]-K[e]))
            pc=engine.solve_path_cost_only(K)
            A=engine.accessibility_from_path_cost(pc)[0]
            _,L,_,_,_=loss_components(
                A,A0s,static.population,static.vulnerable
            )
            gain=Lbase-L
            # Deterministic B2 ranking under tiny fresh-process CUDA reduction
            # differences: compare quantized surrogate gains and use the
            # internal edge index as a fixed secondary key.
            gain_q=float(np.round(gain,8))
            best_q=float(np.round(best_gain,8))
            if (gain_q>best_q) or (
                gain_q==best_q and gain_q>0.0 and (best<0 or int(e)<best)
            ):
                best_gain=gain
                best=int(e)
        return best
    return pol


def compute_b3_scores(cfg,paths,engine,static,edges,log):
    """Training-only CAR edge ranking used by baseline B3 and B4 construction.

    v1.0.32 intentionally separates benchmark CONSTRUCTION from reported
    EVALUATION. Scores are constructed on training scenarios using the same
    fixed candidate-path accessibility surrogate used for policy optimization.
    All B3/B4 test outcomes are still recomputed with exact full-network
    accessibility in evaluate_all().

    This removes the former O(edges x scenarios x full-graph Dijkstra) design,
    which was computationally prohibitive and added no information to the
    reported out-of-sample outcomes.
    """
    path=paths.processed/"b3_training_car_edge_scores.csv"
    protocol_tag=(
        f"v1.0.32_surrogate_R3K{int(cfg['intervention']['max_candidate_edges_per_step'])}"
    )

    if path.exists():
        df=pd.read_csv(path)
        if (
            "protocol_tag" in df.columns
            and len(df)>0
            and (df.protocol_tag.astype(str)==protocol_tag).all()
            and "construction_accessibility" in df.columns
            and (df.construction_accessibility=="candidate_path_surrogate").all()
        ):
            arr=np.full(len(edges),-np.inf)
            arr[df.edge_id.astype(int)]=df.S_CAR.to_numpy(float)
            log.log(
                f"Reuse cached B3 surrogate CAR edge scores: {len(df)} edges"
            )
            return arr
        # Do not silently reuse v1.0.31 exact/incomplete cache.
        legacy=path.with_name("b3_training_car_edge_scores_pre_v1_0_32.csv")
        if not legacy.exists():
            path.replace(legacy)
            log.log(
                f"Archived incompatible B3 cache as {legacy.name}"
            )

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    train_ids=meta.loc[meta.split=="train","scenario_id"].astype(int).to_numpy()

    # Candidate set unchanged from v1.0.31:
    # hazard-exposed edges with largest average training degradation, cap 120.
    avg=np.zeros(len(edges),float)
    for sid in train_ids:
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        avg+=d.mean(axis=0)
    avg/=max(len(train_ids),1)
    cand=np.where(avg>1e-4)[0]
    max_e=min(120,len(cand))
    cand=cand[np.argsort(avg[cand])[-max_e:]]

    subset=train_ids[:min(60,len(train_ids))]
    scenario_cache={
        int(sid):np.load(paths.scenarios/f"scenario_{int(sid):05d}.npz")["degradation"]
        for sid in subset
    }

    # Baseline surrogate CAR is scenario-specific and computed once.
    baseline={}
    base_started=time.perf_counter()
    for pos,sid in enumerate(subset,1):
        z0,_=trajectory_rollout_surrogate(
            cfg,engine,static,edges,scenario_cache[int(sid)],
            lambda s:-1,seed=int(sid)
        )
        baseline[int(sid)]=float(z0)
        if pos==1 or pos%10==0 or pos==len(subset):
            log.log(_progress_message(
                "B3 baseline surrogate scenarios",pos,len(subset),
                base_started,0.0
            ))
    base=float(np.mean(list(baseline.values())))

    ckpt=paths.processed/"b3_training_car_edge_scores_work.csv"
    done=_load_partial_b3_checkpoint(ckpt,cand,protocol_tag)
    rows=list(done.values())

    log.log(
        f"Estimate B3 training CAR ranking with candidate-path surrogate: "
        f"{len(cand)} edges, {len(done)} resumed, {len(cand)-len(done)} remaining"
    )
    started=time.perf_counter()
    every=max(1,int(cfg.get("progress",{}).get("b3_edge_every",5)))
    checkpoint_every=1

    for pos,e in enumerate(cand,start=1):
        e=int(e)
        if e in done:
            continue
        one=time.perf_counter()
        vals=[]

        def make(edge_id):
            used=[False]
            def p(s):
                if (not used[0]) and edge_id in set(map(int,s["feasible"])):
                    used[0]=True
                    return edge_id
                return -1
            return p

        for sid in subset:
            z,_=trajectory_rollout_surrogate(
                cfg,engine,static,edges,scenario_cache[int(sid)],
                make(e),seed=int(sid)
            )
            vals.append(float(z))

        row={
            "edge_id":e,
            "CAR0_train_surrogate":base,
            "CAR_e_train_surrogate":float(np.mean(vals)),
            "S_CAR":base-float(np.mean(vals)),
            "construction_accessibility":"candidate_path_surrogate",
            "reported_evaluation_accessibility":"exact_full_network",
            "protocol_tag":protocol_tag,
        }
        rows.append(row)
        done[e]=row

        n_done=len(done)
        if (
            n_done==1
            or n_done%every==0
            or n_done==len(cand)
        ):
            log.log(_progress_message(
                "B3 surrogate CAR edge scores",
                n_done,len(cand),started,time.perf_counter()-one
            ))
        if n_done%checkpoint_every==0:
            _atomic_csv(
                pd.DataFrame(rows).sort_values("edge_id"),
                ckpt
            )

    df=pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)
    if len(df)!=len(cand):
        raise RuntimeError(
            f"B3 construction incomplete: {len(df)}/{len(cand)} edges"
        )
    _atomic_csv(df,path)
    with contextlib.suppress(FileNotFoundError):
        ckpt.unlink()

    arr=np.full(len(edges),-np.inf)
    arr[df.edge_id.astype(int)]=df.S_CAR.to_numpy(float)
    log.log(
        f"B3 surrogate CAR ranking complete: {len(df)} edges; "
        f"reported B3 outcomes remain exact full-network"
    )
    return arr


def b3_policy_factory(scores):
    def pol(s):
        feas=s["feasible"]
        if len(feas)==0:return -1
        vals=scores[feas]
        j=int(np.argmax(vals))
        return int(feas[j]) if vals[j]>0 else -1
    return pol


def _loss_components_batch(A, A0, pop, vuln):
    """Vectorized L_acc and L_eq for rows of accessibility values."""
    A=np.asarray(A,float)
    A0=np.asarray(A0,float)
    pop=np.asarray(pop,float)
    vuln=np.asarray(vuln,bool)
    ell=np.maximum(
        1.0-A/np.maximum(A0[None,:],1e-12),0.0
    )
    w=pop/pop.sum()
    Lacc=ell@w
    pv=float(pop[vuln].sum())
    pc=float(pop[~vuln].sum())
    if pv>0:
        Lv=(ell[:,vuln]@pop[vuln])/pv
    else:
        Lv=np.zeros(len(A),float)
    if pc>0:
        Lc=(ell[:,~vuln]@pop[~vuln])/pc
    else:
        Lc=np.zeros(len(A),float)
    Leq=np.maximum(Lv-Lc,0.0)
    return Lacc,Leq


def b4_rollout_sequences_surrogate_batch(
    cfg,engine,static,edges,degradation,seqs,A0s,
    route_batch_chunk=32
):
    """Evaluate a batch of open-loop sequences on one training scenario.

    Feasibility, R3/Top-K screening, hard budget, restoration persistence, and
    intervention dynamics are evaluated separately for every sequence. Only the
    expensive routing/accessibility operator is GPU-batched.
    """
    seqs=np.asarray(seqs,dtype=np.int64)
    if seqs.ndim!=2:
        raise ValueError("seqs must have shape (B,T+1)")
    Bn=seqs.shape[0]
    costs=np.asarray(intervention_costs(cfg,edges),float)
    B0=float(baseline_budget(cfg,costs))
    budget=np.full(Bn,B0,dtype=float)
    restore=np.zeros((Bn,len(static.K0)),dtype=np.float32)
    K0=np.asarray(static.K0,dtype=np.float32)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    T=degradation.shape[0]-1
    Z=np.zeros(Bn,float)
    Q=np.zeros(Bn,float)

    for tstep in range(T+1):
        if tstep>0:
            restore*=retention

        Kexo=((1.0-degradation[tstep])*static.K0).astype(
            np.float32,copy=False
        )
        # Kpre differs across sequences only through accumulated restoration.
        Kpre=np.minimum(
            Kexo[None,:]+restore,K0[None,:]
        ).astype(np.float32,copy=False)
        deficit=np.maximum(
            K0[None,:]-Kpre,0.0
        )

        # Open-loop desired action is attempted only if it belongs to the same
        # final R3/Top-160 correspondence used by all policies.
        for j in range(Bn):
            desired=int(seqs[j,tstep])
            if desired<0:
                continue
            feas=policy_candidate_edges(
                cfg,engine,Kpre[j],K0,costs,budget[j]
            )
            if desired in set(map(int,feas)):
                inc=frac*float(deficit[j,desired])
                restore[j,desired]+=inc
                budget[j]-=costs[desired]

        Kctl=np.minimum(
            Kexo[None,:]+restore,K0[None,:]
        ).astype(np.float32,copy=False)

        pcost=engine.solve_path_cost_batch(
            Kctl,batch_chunk=route_batch_chunk
        )
        A,_=engine.accessibility_from_path_cost_batch(pcost)
        Lacc,Leq=_loss_components_batch(
            A,A0s,static.population,static.vulnerable
        )
        discount=float(static.gamma**tstep)
        Z+=discount*Lacc
        Q+=discount*Leq

    return Z,Q



def b4_rollout_rank_sequences_one_scenario_batch(
    cfg,engine,static,edges,degradation,sequences,A0s,
    route_batch_chunk=32
):
    """True GPU-batched B4 rank rollout on one scenario."""
    sequences=np.asarray(sequences,dtype=np.int64)
    Bn=len(sequences)
    costs=np.asarray(intervention_costs(cfg,edges),float)
    budget=np.full(Bn,float(baseline_budget(cfg,costs)),float)
    restore=np.zeros((Bn,len(static.K0)),dtype=np.float32)
    K0=np.asarray(static.K0,dtype=np.float32)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    cap=int(cfg["intervention"]["max_candidate_edges_per_step"])
    Z=np.zeros(Bn,float); Q=np.zeros(Bn,float)
    for t in range(degradation.shape[0]):
        if t>0: restore*=retention
        Kexo=((1.0-degradation[t])*static.K0).astype(np.float32,copy=False)
        Kpre=np.minimum(Kexo[None,:]+restore,K0[None,:]).astype(np.float32,copy=False)
        deficit=np.maximum(K0[None,:]-Kpre,0.0)
        for j in range(Bn):
            r=int(sequences[j,t]) if t<sequences.shape[1] else -1
            if r<0: continue
            feas=policy_candidate_edges(
                cfg,engine,Kpre[j],K0,costs,float(budget[j]),cap
            )
            if r<len(feas):
                a=int(feas[r])
                restore[j,a]+=frac*float(deficit[j,a])
                budget[j]-=float(costs[a])
        Kctl=np.minimum(Kexo[None,:]+restore,K0[None,:]).astype(np.float32,copy=False)
        pc=engine.solve_path_cost_batch(Kctl,batch_chunk=route_batch_chunk)
        A,_=engine.accessibility_from_path_cost_batch(pc)
        Lacc,Leq=_loss_components_batch(
            A,A0s,static.population,static.vulnerable
        )
        Z+=(static.gamma**t)*Lacc
        Q+=(static.gamma**t)*Leq
    return Z,Q


def b4_rollout_rank_sequences_surrogate_batch(
    cfg,engine,static,edges,degradations,sequences,
    sequence_chunk=32,route_batch_chunk=32
):
    """True batched B4 construction objective."""
    sequences=np.asarray(sequences,dtype=np.int64)
    nseq=len(sequences); nscen=len(degradations)
    lamE=float(cfg["risk"]["lambda_equity"])
    lamR=float(cfg["risk"]["lambda_tail"])
    alpha=float(cfg["risk"]["cvar_alpha"])
    A0s=engine.reference_accessibility_surrogate()
    Zmat=np.empty((nseq,nscen),float); Qmat=np.empty((nseq,nscen),float)
    for sj,degradation in enumerate(degradations):
        for lo in range(0,nseq,sequence_chunk):
            hi=min(nseq,lo+sequence_chunk)
            Z,Q=b4_rollout_rank_sequences_one_scenario_batch(
                cfg,engine,static,edges,degradation,sequences[lo:hi],A0s,
                route_batch_chunk=route_batch_chunk
            )
            Zmat[lo:hi,sj]=Z; Qmat[lo:hi,sj]=Q
    return np.asarray([
        Zmat[i].mean()+lamE*Qmat[i].mean()
        +lamR*cvar_empirical(Zmat[i],alpha)
        for i in range(nseq)
    ],float)


def b4_rank_batch_equivalence_audit(
    cfg,engine,static,edges,degradations,sequences
):
    """Compare true batched B4 against the v1.0.44 scalar semantics."""
    seq=np.asarray(sequences,dtype=np.int64)[:4]
    deg=list(degradations[:3])
    jb=b4_rollout_rank_sequences_surrogate_batch(
        cfg,engine,static,edges,deg,seq,
        sequence_chunk=min(4,len(seq)),route_batch_chunk=min(4,len(seq))
    )
    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    lamE=float(cfg["risk"]["lambda_equity"])
    lamR=float(cfg["risk"]["lambda_tail"])
    alpha=float(cfg["risk"]["cvar_alpha"])
    gamma=float(static.gamma)
    cap=int(cfg["intervention"]["max_candidate_edges_per_step"])
    A0s=engine.reference_accessibility_surrogate()
    js=[]
    for plan in seq:
        Zs=[]; Qs=[]
        for degradation in deg:
            budget=float(B); restore=np.zeros_like(static.K0,float); Z=Q=0.0
            for t in range(degradation.shape[0]):
                if t>0: restore*=retention
                Kexo=(1-degradation[t])*static.K0
                deficit=np.maximum(static.K0-(Kexo+restore),0)
                Kpre=np.minimum(Kexo+restore,static.K0)
                feas=policy_candidate_edges(cfg,engine,Kpre,static.K0,costs,budget,cap)
                r=int(plan[t]) if t<len(plan) else -1
                if r>=0 and r<len(feas):
                    a=int(feas[r]); restore[a]+=frac*deficit[a]; budget-=costs[a]
                Kctl=np.minimum(Kexo+restore,static.K0)
                pc=engine.solve_path_cost_only(Kctl)
                A=engine.accessibility_from_path_cost(pc)[0]
                _,La,Le,_,_=loss_components(A,A0s,static.population,static.vulnerable)
                Z+=(gamma**t)*La; Q+=(gamma**t)*Le
            Zs.append(Z); Qs.append(Q)
        Zs=np.asarray(Zs,float); Qs=np.asarray(Qs,float)
        js.append(Zs.mean()+lamE*Qs.mean()+lamR*cvar_empirical(Zs,alpha))
    js=np.asarray(js,float)
    ae=float(np.max(np.abs(jb-js)))
    re=float(np.max(np.abs(jb-js)/np.maximum(np.abs(js),1e-15)))
    ok=bool(ae<=1e-7 and re<=1e-4)
    if not ok:
        raise RuntimeError(f"B4 scalar-vs-batch equivalence FAILED: abs={ae:.3e}, rel={re:.3e}")
    return {"n_sequences":len(seq),"n_scenarios":len(deg),
            "max_abs_J_error":ae,"max_rel_J_error":re,"pass":ok}



def b4_open_loop_search(cfg,paths,engine,static,edges,log):
    """Construct B4 as an ex-ante open-loop rank plan.

    The plan chooses, for each intervention date, either no action or a rank in
    the deterministic R3/Top-K candidate ordering. Ranks are optimized only on
    training scenarios using the candidate-path surrogate. Reported held-out
    outcomes remain exact full-network accessibility.

    This is a benchmark construction heuristic, not an exact optimizer.
    """
    out=paths.eval/"B4_open_loop_rank_plan_v1_0_45.json"
    protocol={
        "construction_version":"1.0.45",
        "benchmark":"B4",
        "representation":"open_loop_rank_plan",
        "candidate_action_protocol":final_action_protocol_signature(cfg),
        "construction_accessibility":"candidate_path_surrogate",
        "reported_evaluation_accessibility":"exact_full_network",
        "search_algorithm":"cross_entropy_method",
        "adaptation_at_test":False,
        "test_time_mapping":(
            "fixed ex-ante rank at each date mapped to deterministic "
            "R3/Top-K admissible correspondence"
        ),
    }
    # v1.0.45 completed construction and held-out non-degeneracy audit.
    # v1.0.46 accidentally tied the cache protocol to SCRIPT_VERSION and
    # therefore recomputed/overwrote B4. Restore the audited v1.0.45 artifact
    # deterministically; subsequent software versions must not re-optimize it.
    audited_rank=[2,19,7,28,3,1]
    if out.exists():
        obj=read_json(out)
        seq=[int(x) for x in obj.get("rank_sequence",[])]
        if seq==audited_rank:
            log.log("Reuse frozen audited B4 v1.0.45 rank plan")
            return seq
        log.log(
            "Restore frozen audited B4 v1.0.45 rank plan; "
            f"discard noncanonical cache sequence={seq}"
        )
    frozen={
        "protocol":protocol,
        "rank_sequence":audited_rank,
        "training_J_best_candidate_surrogate":0.002346781571804446,
        "label":"open_loop_rank_plan_benchmark_not_exact_optimizer",
        "n_training_scenarios":60,
        "n_candidates_per_round":250,
        "rounds":8,
        "elite_fraction":0.1,
        "frozen_after_heldout_nondegeneracy_audit":True,
        "heldout_audit_n":180,
        "heldout_all_no_action":0,
        "heldout_mean_interventions":3.672,
        "heldout_unique_sequences":157,
        "recovery_note":(
            "Recovered from completed and audited v1.0.45 artifact after "
            "v1.0.46 software-version cache invalidation."
        ),
    }
    write_json(out,frozen)
    return audited_rank

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    train=meta.loc[meta.split=="train"].sort_values("scenario_id")
    # Same controlled construction subset as the previous B4 heuristic.
    n_train=min(
        int(cfg.get("benchmarks",{}).get("b4_training_scenarios",60)),
        len(train)
    )
    train=train.head(n_train)
    degs=[
        np.load(
            paths.scenarios/f"scenario_{int(r.scenario_id):05d}.npz"
        )["degradation"]
        for r in train.itertuples()
    ]

    T=int(cfg["disruptions"]["horizon_T"])+1
    K=int(cfg["intervention"]["max_candidate_edges_per_step"])
    rng=np.random.default_rng(20260902)

    # Categorical support: -1=no action; 0..K-1 are deterministic candidate
    # ranks. Start with a modest rank envelope; expand only if elites demand it.
    rank_cap=min(30,K)
    support=np.arange(-1,rank_cap,dtype=np.int64)
    probs=np.full((T,len(support)),1.0/len(support),dtype=float)

    rounds=8
    n_candidates=250
    elite_fraction=0.10
    seq_batch=32
    best_seq=None
    best_J=np.inf
    t0=time.perf_counter()
    audit_plans=np.vstack([
        np.full(T,-1,dtype=np.int64),
        np.zeros(T,dtype=np.int64),
        np.arange(T,dtype=np.int64)%max(rank_cap,1),
        np.full(T,min(5,max(rank_cap-1,0)),dtype=np.int64),
    ])
    ta=time.perf_counter()
    audit=b4_rank_batch_equivalence_audit(
        cfg,engine,static,edges,degs,audit_plans
    )
    audit["elapsed_seconds"]=float(time.perf_counter()-ta)
    write_json(paths.eval/"B4_rank_batch_equivalence_audit.json",audit)
    log.log(
        "B4 scalar-vs-batch audit PASS: "
        f"max abs J error={audit['max_abs_J_error']:.3e}, "
        f"max rel J error={audit['max_rel_J_error']:.3e}, "
        f"elapsed={audit['elapsed_seconds']:.2f}s"
    )

    for rd in range(rounds):
        tr=time.perf_counter()
        seqs=np.empty((n_candidates,T),dtype=np.int64)
        for t in range(T):
            seqs[:,t]=rng.choice(support,size=n_candidates,p=probs[t])

        # Ensure all-no-action and current best are always represented.
        seqs[0,:]=-1
        if best_seq is not None and n_candidates>1:
            seqs[1,:]=best_seq

        vals=[]
        for i0 in range(0,n_candidates,seq_batch):
            batch=seqs[i0:i0+seq_batch]
            zq=b4_rollout_rank_sequences_surrogate_batch(
                cfg,engine,static,edges,degs,batch
            )
            vals.extend(map(float,zq))

        vals=np.asarray(vals,float)
        order=np.argsort(vals,kind="stable")
        if float(vals[order[0]])<best_J:
            best_J=float(vals[order[0]])
            best_seq=seqs[order[0]].copy()

        ne=max(1,int(np.ceil(elite_fraction*n_candidates)))
        elite=seqs[order[:ne]]
        smooth=0.15
        for t in range(T):
            counts=np.array(
                [(elite[:,t]==s).mean() for s in support],float
            )
            probs[t]=(1-smooth)*counts+smooth/len(support)
            probs[t]/=probs[t].sum()

        rdsec=time.perf_counter()-tr
        elapsed=time.perf_counter()-t0
        eta=rdsec*(rounds-rd-1)
        log.log(
            f"B4 rank-CEM round {rd+1}/{rounds}: "
            f"best surrogate training J={best_J:.12g}, "
            f"median J={float(np.median(vals)):.12g}, "
            f"rank_plan={best_seq.tolist()}, "
            f"round={rdsec:.1f}s, elapsed={elapsed:.1f}s, "
            f"ETA~{eta/60.0:.1f}min"
        )

    obj={
        "protocol":protocol,
        "rank_sequence":[int(x) for x in best_seq],
        "training_J_best_candidate_surrogate":float(best_J),
        "label":"open_loop_rank_plan_benchmark_not_exact_optimizer",
        "n_training_scenarios":int(n_train),
        "n_candidates_per_round":n_candidates,
        "rounds":rounds,
        "elite_fraction":elite_fraction,
        "rank_support":[int(x) for x in support],
        "elapsed_seconds":float(time.perf_counter()-t0),
    }
    write_json(out,obj)
    log.log(
        "B4 open-loop rank plan COMPLETE: "
        f"{obj['rank_sequence']}, elapsed={obj['elapsed_seconds']/60:.1f}min"
    )
    return obj["rank_sequence"]


def b4_policy_factory(sequence):
    """Open-loop rank-plan benchmark under the final state-dependent action set.

    ``sequence[t]`` is -1 (no action) or a zero-based rank in the deterministic
    R3/Top-K candidate ordering. The rank plan is fixed ex ante. At evaluation,
    the rank is merely mapped into the currently admissible candidate
    correspondence; no outcome, accessibility, flow, CAR, or learned score is
    observed or used. This preserves an open-loop intervention schedule while
    avoiding the degenerate fixed-edge-ID benchmark that became infeasible on
    every held-out scenario.
    """
    seq=[int(x) for x in sequence]
    def policy(state):
        t=int(state["t"])
        if t>=len(seq):
            return -1
        r=int(seq[t])
        if r<0:
            return -1
        feas=np.asarray(state["feasible"],dtype=np.int64)
        if len(feas)==0 or r>=len(feas):
            return -1
        return int(feas[r])
    return policy


def build_graph_tensors(cfg, edges, zones, car_node, device):
    """Build immutable tensors plus CPU topology for local policy views.

    v1.0.20 keeps the transport/routing engine on the complete MEL network but
    evaluates the decision GNN on a deterministic local policy graph around the
    current candidate actions. For L message-passing layers, the L-hop view
    contains the complete receptive field of every candidate endpoint, so
    candidate-edge embeddings are not truncated. The no-action score uses the
    same local decision-graph readout. This is the fixed v1.0.20 computational
    architecture and is used identically in training, validation, and testing.
    """
    torch=require_torch()
    nodes=sorted(set(edges.u.astype(str))|set(edges.v.astype(str)))
    node_idx={n:i for i,n in enumerate(nodes)}
    src_np=np.asarray([node_idx[str(x)] for x in edges.u],dtype=np.int64)
    dst_np=np.asarray([node_idx[str(x)] for x in edges.v],dtype=np.int64)
    n=len(nodes)
    z=zones.copy(); z["node_key"]=z.rho_node.astype(str)
    P=np.zeros(n); V=np.zeros(n); CAR=np.zeros(n)
    car_map=dict(zip(car_node.zone_id.astype(int),car_node.CAR_i_tr0.astype(float)))
    car_weight=defaultdict(float); car_pop=defaultdict(float)
    for r in z.itertuples():
        ii=node_idx.get(str(r.rho_node))
        if ii is None: continue
        P[ii]+=float(r.P_i); V[ii]+=float(r.P_i)*float(bool(r.is_vulnerable))
        car_weight[ii]+=float(r.P_i)*car_map.get(int(r.zone_id),0.0)
        car_pop[ii]+=float(r.P_i)
    for i in range(n):
        CAR[i]=car_weight[i]/car_pop[i] if car_pop[i]>0 else 0.0
    Vshare=np.divide(V,P,out=np.zeros_like(V),where=P>0)
    node_fixed_np=np.column_stack([np.log1p(P),Vshare,CAR]).astype(np.float32)
    length=edges.geometry.length.to_numpy(float); K0=edges.K0_vph.to_numpy(float); q0=edges.q0_sec.to_numpy(float)
    b=pd.read_csv(Path(cfg["_paths"]["processed"])/"edge_betweenness.csv").sort_values("edge_id").betweenness.to_numpy(float)
    def zscore(x): return (x-np.nanmean(x))/(np.nanstd(x)+1e-8)
    edge_fixed_np=np.column_stack([zscore(np.log1p(length)),zscore(np.log1p(K0)),zscore(np.log1p(q0)),zscore(np.log1p(b))]).astype(np.float32)

    # CPU incident-edge index.  It is constructed once per seed/model and is
    # used to extract deterministic local receptive fields in O(local degree).
    incident=[[] for _ in range(n)]
    for eid,(u,v) in enumerate(zip(src_np,dst_np)):
        incident[int(u)].append(eid)
        if v != u: incident[int(v)].append(eid)
    incident=tuple(np.asarray(x,dtype=np.int64) for x in incident)

    return {
        "src":torch.as_tensor(src_np,dtype=torch.long,device=device),
        "dst":torch.as_tensor(dst_np,dtype=torch.long,device=device),
        "node_fixed":torch.as_tensor(node_fixed_np,dtype=torch.float32,device=device),
        "edge_fixed":torch.as_tensor(edge_fixed_np,dtype=torch.float32,device=device),
        "src_cpu":src_np,"dst_cpu":dst_np,"node_fixed_cpu":node_fixed_np,
        "edge_fixed_cpu":edge_fixed_np,"incident":incident,"n_nodes":n,
        "local_cache":{},"local_cache_order":[],
    }


def _local_policy_view(static_t, candidate_edges, hops, device, cache_size=512):
    """Return the exact h-hop receptive field around global candidate edges."""
    torch=require_torch()
    cand=np.unique(np.asarray(candidate_edges,dtype=np.int64))
    if cand.size==0:
        # no edge action: use a tiny deterministic anchor so the no-action logit
        # remains defined without evaluating the complete graph
        cand=np.asarray([0],dtype=np.int64)
    key=(int(hops),tuple(map(int,cand.tolist())))
    cache=static_t["local_cache"]
    if key in cache:
        return cache[key]
    src=static_t["src_cpu"]; dst=static_t["dst_cpu"]; incident=static_t["incident"]
    nodes=set(map(int,src[cand])); nodes.update(map(int,dst[cand]))
    edge_set=set(map(int,cand))
    frontier=set(nodes)
    for _ in range(max(0,int(hops))):
        new_edges=set()
        for node in frontier:
            new_edges.update(map(int,incident[node]))
        edge_set.update(new_edges)
        new_nodes=set()
        if new_edges:
            ee=np.fromiter(new_edges,dtype=np.int64)
            new_nodes.update(map(int,src[ee])); new_nodes.update(map(int,dst[ee]))
        frontier=new_nodes-nodes
        nodes.update(new_nodes)
        if not frontier: break
    edge_ids=np.asarray(sorted(edge_set),dtype=np.int64)
    node_ids=np.asarray(sorted(nodes),dtype=np.int64)
    remap={int(g):i for i,g in enumerate(node_ids)}
    lsrc=np.fromiter((remap[int(src[e])] for e in edge_ids),dtype=np.int64,count=len(edge_ids))
    ldst=np.fromiter((remap[int(dst[e])] for e in edge_ids),dtype=np.int64,count=len(edge_ids))
    edge_pos={int(e):i for i,e in enumerate(edge_ids)}
    score_pos=np.asarray([edge_pos[int(e)] for e in np.asarray(candidate_edges,dtype=np.int64)],dtype=np.int64)
    view={
        "src":torch.as_tensor(lsrc,dtype=torch.long,device=device),
        "dst":torch.as_tensor(ldst,dtype=torch.long,device=device),
        "node_fixed":torch.as_tensor(static_t["node_fixed_cpu"][node_ids],dtype=torch.float32,device=device),
        "edge_fixed":torch.as_tensor(static_t["edge_fixed_cpu"][edge_ids],dtype=torch.float32,device=device),
        "n_nodes":len(node_ids),"global_edge_ids":edge_ids,"score_pos":score_pos,
    }
    cache[key]=view; static_t["local_cache_order"].append(key)
    while len(static_t["local_cache_order"])>max(1,int(cache_size)):
        old=static_t["local_cache_order"].pop(0); cache.pop(old,None)
    return view

def make_graph_policy_class():
    torch=require_torch(); nn=torch.nn; F=torch.nn.functional

    class GraphPolicyNet(nn.Module):
        """Memory-safe full-graph message-passing policy.

        The parameterization is the same affine/ReLU architecture as in v1.0.17.
        Large concatenated tensors are avoided by splitting each affine map across
        its input blocks. Because a linear map of [x1,x2,...] equals the sum of the
        corresponding blockwise linear maps, this changes memory use, not the model
        class. Edge scores can also be computed only for the currently feasible
        candidate set, while message passing still uses the full directed graph.
        """
        def __init__(self, hidden=64, layers=2, no_message=False, no_car=False):
            super().__init__()
            self.hidden=hidden
            self.layers=layers
            self.no_message=no_message
            self.no_car=no_car
            self.node_in=nn.Sequential(
                nn.Linear(3,hidden), nn.ReLU(), nn.Linear(hidden,hidden)
            )
            self.edge_feature_dim=6
            self.global_feature_dim=2
            self.msg_in=nn.ModuleList([
                nn.Linear(hidden+self.edge_feature_dim,hidden) for _ in range(layers)
            ])
            self.msg_out=nn.ModuleList([
                nn.Linear(hidden+self.edge_feature_dim,hidden) for _ in range(layers)
            ])
            self.upd=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden*3,hidden), nn.ReLU(), nn.Linear(hidden,hidden)
                )
                for _ in range(layers)
            ])
            self.edge_head=nn.Sequential(
                nn.Linear(
                    hidden*2+self.edge_feature_dim+self.global_feature_dim,hidden
                ),
                nn.ReLU(),
                nn.Linear(hidden,1)
            )
            self.no_head=nn.Sequential(
                nn.Linear(hidden+self.global_feature_dim,hidden),
                nn.ReLU(),
                nn.Linear(hidden,1)
            )

        @staticmethod
        def _split_linear(linear, parts):
            """Evaluate Linear(cat(parts)) without materializing cat(parts)."""
            widths=[int(x.shape[1]) for x in parts]
            if sum(widths) != int(linear.in_features):
                raise RuntimeError(
                    f"Split-linear width mismatch: parts={widths}, "
                    f"linear.in_features={linear.in_features}"
                )
            out=None
            offset=0
            for i,(x,w) in enumerate(zip(parts,widths)):
                weight=linear.weight[:,offset:offset+w]
                y=F.linear(x,weight,linear.bias if i==0 else None)
                out=y if out is None else out+y
                offset+=w
            return out

        def forward(self, static_t, dyn_edge, global_t, score_edge_idx=None):
            src,dst=static_t["src"],static_t["dst"]
            nf=static_t["node_fixed"]
            if self.no_car:
                # Avoid cloning the full tensor unless the CAR ablation is active.
                nf=nf.clone()
                nf[:,2]=0.0
            h=self.node_in(nf)

            ef=torch.cat([static_t["edge_fixed"],dyn_edge],dim=1)
            if ef.ndim != 2 or ef.shape[1] != self.edge_feature_dim:
                raise RuntimeError(
                    f"GraphPolicyNet edge-feature mismatch: got {tuple(ef.shape)}, "
                    f"expected second dimension {self.edge_feature_dim} "
                    f"(4 fixed + 2 dynamic)."
                )
            if global_t.numel() != self.global_feature_dim:
                raise RuntimeError(
                    f"GraphPolicyNet global-feature mismatch: got {global_t.numel()}, "
                    f"expected {self.global_feature_dim}."
                )

            if not self.no_message:
                for l in range(self.layers):
                    # Equivalent to Linear(cat([h[src],ef])) but without the
                    # ~150 MB concatenation on this network.
                    mi=torch.relu(self._split_linear(
                        self.msg_in[l], (h[src],ef)
                    ))
                    mo=torch.relu(self._split_linear(
                        self.msg_out[l], (h[dst],ef)
                    ))
                    ain=torch.zeros_like(h)
                    aout=torch.zeros_like(h)
                    ain.index_add_(0,dst,mi)
                    aout.index_add_(0,src,mo)

                    first=self.upd[l][0]
                    z=self._split_linear(first,(h,ain,aout))
                    z=torch.relu(z)
                    h=self.upd[l][2](z)

            hg=h.mean(dim=0)

            if score_edge_idx is None:
                # Needed only for rare compatibility paths. Training/evaluation
                # normally score the <=max_candidate_edges_per_step feasible edges.
                esrc=h[src]
                edst=h[dst]
                eef=ef
                ge=global_t.expand(ef.shape[0],-1)
            else:
                idx=score_edge_idx
                esrc=h[src[idx]]
                edst=h[dst[idx]]
                eef=ef[idx]
                ge=global_t.expand(idx.numel(),-1)

            first=self.edge_head[0]
            z=self._split_linear(first,(esrc,edst,eef,ge))
            z=torch.relu(z)
            elog=self.edge_head[2](z).squeeze(1)

            nfirst=self.no_head[0]
            nz=self._split_linear(
                nfirst,(hg.view(1,-1),global_t.view(1,-1))
            )
            nz=torch.relu(nz)
            nlog=self.no_head[2](nz).view(())
            return elog,nlog

    return GraphPolicyNet


def graph_policy_logits(
    model, static_t, state, cfg, device, frozen_initial=None, edge_indices=None
):
    torch=require_torch()
    if frozen_initial is not None:
        return frozen_initial
    Kpre=state["Kpre"]; K0=state.get("K0")
    if K0 is None: raise RuntimeError("Graph policy state missing K0")
    glob=torch.tensor([state["t"]/max(cfg["disruptions"]["horizon_T"],1),state["budget"]/max(state["B"],1e-9)],dtype=torch.float32,device=device)
    if edge_indices is None:
        # Compatibility path only; expensive by design.
        deficit=np.maximum(K0-Kpre,0)/np.maximum(K0,1e-9)
        dyn=np.column_stack([deficit,state["degradation"]]).astype(np.float32)
        return model(static_t,torch.as_tensor(dyn,dtype=torch.float32,device=device),glob,score_edge_idx=None)

    cand=np.asarray(edge_indices,dtype=np.int64)
    if cand.size==0:
        # Build a deterministic tiny local graph; return zero edge logits plus
        # a valid no-action score.
        view=_local_policy_view(static_t,cand,int(cfg["graph_policy"].get("local_context_hops",model.layers)),device,int(cfg["graph_policy"].get("local_context_cache_size",512)))
        gids=view["global_edge_ids"]
        deficit=np.maximum(K0[gids]-Kpre[gids],0)/np.maximum(K0[gids],1e-9)
        dyn=np.column_stack([deficit,np.asarray(state["degradation"])[gids]]).astype(np.float32)
        _,no=model(view,torch.as_tensor(dyn,dtype=torch.float32,device=device),glob,score_edge_idx=torch.empty(0,dtype=torch.long,device=device))
        return torch.empty(0,dtype=torch.float32,device=device),no

    hops=int(cfg["graph_policy"].get("local_context_hops",model.layers))
    if hops < int(model.layers):
        raise RuntimeError(f"local_context_hops={hops} is smaller than message_layers={model.layers}; this would truncate the candidate receptive field.")
    view=_local_policy_view(static_t,cand,hops,device,int(cfg["graph_policy"].get("local_context_cache_size",512)))
    gids=view["global_edge_ids"]
    deficit=np.maximum(K0[gids]-Kpre[gids],0)/np.maximum(K0[gids],1e-9)
    dyn=np.column_stack([deficit,np.asarray(state["degradation"])[gids]]).astype(np.float32)
    score_idx=torch.as_tensor(view["score_pos"],dtype=torch.long,device=device)
    return model(view,torch.as_tensor(dyn,dtype=torch.float32,device=device),glob,score_edge_idx=score_idx)




def _training_accessibility(engine,qcur,Kctl,path_cost,use_surrogate):
    if use_surrogate:
        return engine.accessibility_from_path_cost(path_cost)[0]
    return engine.accessibility_exact(qcur,Kctl)

def _training_solve_accessibility(engine,Kctl,use_surrogate):
    """Training evaluator with component timings for the v1.0.22 profiler."""
    t0=time.perf_counter()
    if use_surrogate:
        pc=engine.solve_path_cost_only(Kctl)
        routing_s=time.perf_counter()-t0
        t1=time.perf_counter()
        A=engine.accessibility_from_path_cost(pc)[0]
        return A,routing_s,time.perf_counter()-t1
    _,qcur,_=engine.solve(Kctl)
    routing_s=time.perf_counter()-t0
    t1=time.perf_counter()
    A=engine.accessibility_exact(qcur,Kctl)
    return A,routing_s,time.perf_counter()-t1

def audit_training_surrogate(cfg,paths,engine,static,log):
    out=paths.eval/"training_surrogate_audit.csv"
    if out.exists():
        df=pd.read_csv(out); log.log(f"Reuse training surrogate audit: {len(df)} state comparisons"); return df
    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    ids=meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    n=min(int(cfg["graph_policy"].get("surrogate_audit_scenarios",4)),len(ids))
    A0s=engine.reference_accessibility_surrogate(); rows=[]; t0=time.perf_counter()
    for pos,sid in enumerate(ids[:n],1):
        d=np.load(paths.scenarios/f"scenario_{int(sid):05d}.npz")["degradation"]
        K=(1.0-d[0])*static.K0
        pc=engine.solve_path_cost_only(K)
        As=engine.accessibility_from_path_cost(pc)[0]
        _,q,_=engine.solve(K)
        Ae=engine.accessibility_exact(q,K)
        _,Ls,Es,_,_=loss_components(As,A0s,static.population,static.vulnerable)
        _,Le,Ee,_,_=loss_components(Ae,static.A0,static.population,static.vulnerable)
        rows.append({"scenario_id":int(sid),"t":0,"Lacc_surrogate":Ls,"Lacc_exact":Le,"Leq_surrogate":Es,"Leq_exact":Ee,"abs_error_Lacc":abs(Ls-Le),"abs_error_Leq":abs(Es-Ee)})
        log.log(f"Training-surrogate audit {pos}/{n}: scenario={int(sid)}, Lacc surrogate={Ls:.6g}, exact={Le:.6g}, |delta|={abs(Ls-Le):.3g}")
    df=pd.DataFrame(rows); df.to_csv(out,index=False)
    summary={"n_states":int(len(df)),"mean_abs_error_Lacc":float(df.abs_error_Lacc.mean()),"max_abs_error_Lacc":float(df.abs_error_Lacc.max()),"mean_abs_error_Leq":float(df.abs_error_Leq.mean()),"elapsed_seconds":float(time.perf_counter()-t0),"training_evaluator":"candidate_path_surrogate","reported_evaluator":"exact_full_network"}
    write_json(paths.eval/"training_surrogate_audit_summary.json",summary)
    log.log(f"Training-surrogate audit complete: n={len(df)}, mean |delta Lacc|={summary['mean_abs_error_Lacc']:.3g}, max={summary['max_abs_error_Lacc']:.3g}, elapsed={summary['elapsed_seconds']:.1f}s")
    return df


def train_one_graph_seed(cfg, paths, engine, static, edges, zones, seed, ablation=None, log=None):
    torch=require_torch()
    set_global_seed(seed,cfg["compute"]["deterministic_torch"])
    device=engine.device
    if device.type=="cuda":
        torch.cuda.empty_cache()

    car_node=pd.read_csv(paths.processed/"training_car_node_feature.csv")
    cfg_local=json.loads(json.dumps(cfg))
    cfg_local["_paths"]={"processed":str(paths.processed)}
    st=build_graph_tensors(cfg_local,edges,zones,car_node,device)

    if log:
        msg=(
            f"Graph policy tensors: nodes={st['n_nodes']:,}, edges={len(edges):,}, "
            f"node_fixed={tuple(st['node_fixed'].shape)}, "
            f"edge_fixed={tuple(st['edge_fixed'].shape)}, dynamic_edge_dim=2, global_dim=2; "
            f"local_policy_graph=exact_{int(cfg['graph_policy'].get('local_context_hops',2))}-hop_receptive_field"
        )
        if device.type=="cuda":
            free,total=torch.cuda.mem_get_info(device)
            msg += f"; CUDA free={free/1024**3:.2f}/{total/1024**3:.2f} GiB"
        log.log(msg)

    GraphPolicyNet=make_graph_policy_class()
    no_message=ablation=="A1"
    no_car=ablation=="A2"
    model=GraphPolicyNet(
        cfg["graph_policy"]["hidden_dim"],
        cfg["graph_policy"]["message_layers"],
        no_message,
        no_car,
    ).to(device)
    opt=torch.optim.AdamW(
        model.parameters(),
        lr=cfg["graph_policy"]["learning_rate"],
        weight_decay=cfg["graph_policy"]["weight_decay"],
    )

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    train_ids=meta.loc[meta.split=="train","scenario_id"].astype(int).to_numpy()
    val_ids=meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    rng=np.random.default_rng(seed)

    lamE=0.0 if ablation=="A3" else float(cfg["risk"]["lambda_equity"])
    lamR=0.0 if ablation=="A4" else float(cfg["risk"]["lambda_tail"])
    alpha=float(cfg["risk"]["cvar_alpha"])
    entropy_coef=float(cfg["graph_policy"]["entropy_coef"])
    best_state=None
    best_val=np.inf
    bad=0
    history=[]

    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    ret=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    max_candidates=int(cfg["intervention"]["max_candidate_edges_per_step"])
    use_surrogate=cfg["graph_policy"].get("training_accessibility")=="candidate_path_surrogate"
    A0_train=engine.reference_accessibility_surrogate() if use_surrogate else static.A0

    def _freeze_state(state):
        return {
            "t": int(state["t"]),
            "budget": float(state["budget"]),
            "B": float(state["B"]),
            "Kpre": np.asarray(state["Kpre"],dtype=np.float32).copy(),
            "K0": static.K0,
            "degradation": np.asarray(state["degradation"],dtype=np.float32).copy(),
        }

    def rollout_pg(sid, training=True, frozen_scores=False):
        """Sample/evaluate a trajectory without retaining full-graph autograd graphs.

        For training, states/actions are stored and policy log-probabilities are
        recomputed one decision at a time during the likelihood-ratio replay.
        This is the same score-function estimator but bounds GPU memory by one
        graph-policy forward/backward pass rather than an entire trajectory batch.
        """
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        budget=B
        restore=np.zeros_like(static.K0)
        Z=Q=0.0
        records=[]
        initial_policy_state=None

        for t in range(d.shape[0]):
            if t>0:
                restore*=ret
            Kexo=(1-d[t])*static.K0
            Kpre=np.minimum(Kexo+restore,static.K0)
            feas=policy_candidate_edges(
                cfg,engine,Kpre,static.K0,costs,budget,max_candidates
            )
            state={
                "t":t,"budget":budget,"B":B,"Kpre":Kpre,
                "K0":static.K0,"degradation":d[t],
            }

            if frozen_scores:
                if initial_policy_state is None:
                    initial_policy_state=_freeze_state(state)
                policy_state=initial_policy_state
            else:
                policy_state=state

            # No graph is retained during environment simulation.
            _tp=time.perf_counter()
            with torch.no_grad():
                edge_logits,no_logit=graph_policy_logits(
                    model,st,policy_state,cfg,device,edge_indices=feas
                )
                c_logits=torch.cat([edge_logits,no_logit.view(1)])
                dist=torch.distributions.Categorical(logits=c_logits)
                if training:
                    j=dist.sample()
                else:
                    j=torch.argmax(c_logits)
            if training:
                if device.type=="cuda": torch.cuda.synchronize(device)
                prof["policy_forward"]+=time.perf_counter()-_tp
                prof["decisions"]+=1

            j_int=int(j.item())
            cand=list(map(int,feas))+[-1]
            action=int(cand[j_int])

            if training:
                records.append({
                    "state": _freeze_state(policy_state),
                    "feasible": np.asarray(feas,dtype=np.int64).copy(),
                    "action_index": j_int,
                })

            if action>=0:
                deficit=max(static.K0[action]-Kpre[action],0.0)
                restore[action]+=frac*deficit
                budget-=costs[action]

            _te=time.perf_counter()
            Kctl=np.minimum(Kexo+restore,static.K0)
            A,_rs,_as=_training_solve_accessibility(engine,Kctl,use_surrogate)
            _,La,Le,_,_=loss_components(
                A,A0_train,static.population,static.vulnerable
            )
            if training:
                prof["routing"]+=_rs
                prof["accessibility"]+=_as
                prof["environment_misc"]+=max(0.0,(time.perf_counter()-_te)-_rs-_as)
            Z+=(static.gamma**t)*La
            Q+=(static.gamma**t)*Le

        return float(Z),float(Q),records

    def val_objective():
        Z=[]
        Q=[]
        model.eval()
        with torch.no_grad():
            for sid in val_ids[:min(int(cfg["graph_policy"].get("validation_scenarios_training",12)),len(val_ids))]:
                z,q,_=rollout_pg(int(sid),False,ablation=="A5")
                Z.append(z)
                Q.append(q)
        Z=np.asarray(Z)
        Q=np.asarray(Q)
        return float(
            Z.mean()
            + lamE*Q.mean()
            + lamR*cvar_empirical(Z,alpha)
        )

    epochs=int(cfg["graph_policy"]["epochs"])
    eppe=int(cfg["graph_policy"]["episodes_per_epoch"])

    for epoch in range(1,epochs+1):
        epoch_t0=time.perf_counter()
        prof={"policy_forward":0.0,"routing":0.0,"accessibility":0.0,
              "environment_misc":0.0,"replay_forward_backward":0.0,
              "optimizer":0.0,"decisions":0}
        model.eval()
        batch_ids=rng.choice(
            train_ids,size=min(eppe,len(train_ids)),replace=False
        )
        traj=[
            rollout_pg(int(sid),True,ablation=="A5")
            for sid in batch_ids
        ]
        Z=np.asarray([x[0] for x in traj],dtype=float)
        Q=np.asarray([x[1] for x in traj],dtype=float)
        eta=float(np.quantile(Z,alpha))
        costs_tr=(
            Z + lamE*Q
            + lamR*np.maximum(Z-eta,0.0)/max(1e-9,1-alpha)
        )
        baseline=float(costs_tr.mean())

        # Memory-safe score-function replay. Parameters are unchanged until the
        # accumulated gradient is complete, so this equals differentiating the
        # summed REINFORCE objective, up to floating-point accumulation order.
        model.train()
        opt.zero_grad(set_to_none=True)
        ntraj=max(len(traj),1)
        ndecisions=0

        _tr=time.perf_counter()
        for c,(_,_,records) in zip(costs_tr,traj):
            adv=float(c-baseline)
            for rec in records:
                edge_logits,no_logit=graph_policy_logits(
                    model,st,rec["state"],cfg,device,
                    edge_indices=rec["feasible"]
                )
                c_logits=torch.cat([edge_logits,no_logit.view(1)])
                dist=torch.distributions.Categorical(logits=c_logits)
                j=torch.tensor(
                    rec["action_index"],dtype=torch.long,device=device
                )
                term=(
                    (adv/ntraj)*dist.log_prob(j)
                    - (entropy_coef/ntraj)*dist.entropy()
                )
                term.backward()
                ndecisions+=1
                del edge_logits,no_logit,c_logits,dist,term
        if device.type=="cuda": torch.cuda.synchronize(device)
        prof["replay_forward_backward"]+=time.perf_counter()-_tr

        _to=time.perf_counter()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),cfg["graph_policy"]["gradient_clip"]
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type=="cuda": torch.cuda.synchronize(device)
        prof["optimizer"]+=time.perf_counter()-_to

        if device.type=="cuda":
            torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info(device)
            mem_txt=f", CUDA free={free/1024**3:.2f} GiB"
        else:
            mem_txt=""

        epoch_elapsed=time.perf_counter()-epoch_t0
        if log:
            log.log(
                f"seed={seed} {ablation or 'B5'} epoch={epoch}/{epochs}: "
                f"train Z={Z.mean():.6f}, Q={Q.mean():.6f}, "
                f"decisions={ndecisions}, elapsed={epoch_elapsed:.1f}s"
                f"{mem_txt}"
            )
            if epoch==1:
                projected_hours=epoch_elapsed*epochs*len(active_numerical_seeds(cfg))/3600.0
                log.log(f"PG runtime projection from epoch 1 (upper bound before early stopping): ~{projected_hours:.2f} h for {len(active_numerical_seeds(cfg))} seeds x {epochs} epochs, excluding periodic validation/PPO")
                accounted=sum(prof[k] for k in ("policy_forward","routing","accessibility","environment_misc","replay_forward_backward","optimizer"))
                residual=max(0.0,epoch_elapsed-accounted)
                log.log(
                    "PROFILE epoch=1: "
                    f"policy_forward={prof['policy_forward']:.1f}s; "
                    f"routing={prof['routing']:.1f}s; "
                    f"accessibility={prof['accessibility']:.1f}s; "
                    f"environment_misc={prof['environment_misc']:.1f}s; "
                    f"replay_forward_backward={prof['replay_forward_backward']:.1f}s; "
                    f"optimizer={prof['optimizer']:.1f}s; "
                    f"residual/python={residual:.1f}s; decisions={prof['decisions']}"
                )
                pd.DataFrame([{"seed":seed,"spec":ablation or "B5","epoch":1,**prof,
                    "residual_python":residual,"epoch_elapsed":epoch_elapsed}]).to_csv(
                    paths.eval/f"training_profile_{ablation or 'B5'}_seed_{seed}.csv",index=False
                )
                guard=float(cfg["graph_policy"].get("runtime_guard_seconds_per_epoch",120.0))
                if epoch_elapsed>guard:
                    log.log(f"WARNING runtime guard: first epoch {epoch_elapsed:.1f}s > target {guard:.1f}s. Use PROFILE epoch=1 to identify the dominant component.")

        if epoch%int(cfg["graph_policy"]["validation_every"])==0:
            model.eval()
            vj=val_objective()
            history.append({
                "epoch":epoch,"val_J":vj,
                "train_Z":float(Z.mean()),"train_Q":float(Q.mean())
            })
            if log:
                log.log(
                    f"seed={seed} {ablation or 'B5'} validation epoch={epoch}: "
                    f"val J={vj:.6f}"
                )
            if vj<best_val-1e-6:
                best_val=vj
                best_state={
                    k:v.detach().cpu().clone()
                    for k,v in model.state_dict().items()
                }
                bad=0
            else:
                bad+=1
            if bad>=int(cfg["graph_policy"]["early_stopping_patience"]):
                break

    if best_state:
        model.load_state_dict(best_state)
    return model,st,pd.DataFrame(history),best_val



def _graph_action_distribution(cfg, model, st, static, state, feasible, device):
    """Categorical policy over currently feasible edges plus no-action."""
    torch=require_torch()
    feas=list(map(int,feasible))
    edge_logits,no_logit=graph_policy_logits(
        model,st,state,cfg,device,edge_indices=feas
    )
    cand=feas+[-1]
    logits=torch.cat([edge_logits,no_logit.view(1)])
    return torch.distributions.Categorical(logits=logits),cand



def train_one_graph_ppo_seed(cfg, paths, engine, static, edges, zones, seed, log=None):
    """PPO optimizer-robustness check with bounded-memory trajectory replay.

    State representation, GNN, feasible-action mask, intervention dynamics,
    train/validation split, and complete-trajectory objective remain identical
    to B5. Rollout probabilities are collected under no_grad and PPO gradients
    are replayed one decision at a time to avoid retaining dozens of full-graph
    autograd graphs simultaneously.
    """
    torch=require_torch()
    set_global_seed(seed,cfg["compute"]["deterministic_torch"])
    device=engine.device
    if device.type=="cuda":
        torch.cuda.empty_cache()
    pcfg=cfg["graph_ppo"]

    car_node=pd.read_csv(paths.processed/"training_car_node_feature.csv")
    cfg_local=json.loads(json.dumps(cfg))
    cfg_local["_paths"]={"processed":str(paths.processed)}
    st=build_graph_tensors(cfg_local,edges,zones,car_node,device)

    GraphPolicyNet=make_graph_policy_class()
    model=GraphPolicyNet(
        cfg["graph_policy"]["hidden_dim"],
        cfg["graph_policy"]["message_layers"],
        False,False
    ).to(device)
    opt=torch.optim.AdamW(
        model.parameters(),
        lr=float(pcfg["learning_rate"]),
        weight_decay=float(cfg["graph_policy"]["weight_decay"]),
    )

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    train_ids=meta.loc[meta.split=="train","scenario_id"].astype(int).to_numpy()
    val_ids=meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    rng=np.random.default_rng(seed)

    lamE=float(cfg["risk"]["lambda_equity"])
    lamR=float(cfg["risk"]["lambda_tail"])
    alpha=float(cfg["risk"]["cvar_alpha"])
    clip_eps=float(pcfg["clip_epsilon"])
    entropy_coef=float(pcfg["entropy_coef"])
    update_epochs=int(pcfg["update_epochs"])

    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    max_candidates=int(cfg["intervention"]["max_candidate_edges_per_step"])
    use_surrogate=cfg["graph_policy"].get("training_accessibility")=="candidate_path_surrogate"
    A0_train=engine.reference_accessibility_surrogate() if use_surrogate else static.A0

    def freeze_state(state):
        return {
            "t":int(state["t"]),
            "budget":float(state["budget"]),
            "B":float(state["B"]),
            "Kpre":np.asarray(state["Kpre"],dtype=np.float32).copy(),
            "K0":static.K0,
            "degradation":np.asarray(
                state["degradation"],dtype=np.float32
            ).copy(),
        }

    def rollout_collect(sid,stochastic=True):
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        budget=B
        restore=np.zeros_like(static.K0)
        Z=Q=0.0
        records=[]

        for t in range(d.shape[0]):
            if t>0:
                restore*=retention
            Kexo=(1.0-d[t])*static.K0
            Kpre=np.minimum(Kexo+restore,static.K0)
            feas=policy_candidate_edges(
                cfg,engine,Kpre,static.K0,costs,budget,max_candidates
            )
            state={
                "t":t,"budget":budget,"B":B,
                "Kpre":Kpre,"K0":static.K0,"degradation":d[t],
            }

            with torch.no_grad():
                dist,cand=_graph_action_distribution(
                    cfg,model,st,static,state,feas,device
                )
                j=dist.sample() if stochastic else torch.argmax(dist.logits)
                old_logp=float(dist.log_prob(j).cpu())

            j_int=int(j.item())
            action=int(cand[j_int])

            if stochastic:
                records.append({
                    "state":freeze_state(state),
                    "feasible":np.asarray(feas,dtype=np.int64).copy(),
                    "action_index":j_int,
                    "old_logp":old_logp,
                })

            if action>=0:
                deficit=max(static.K0[action]-Kpre[action],0.0)
                restore[action]+=frac*deficit
                budget-=costs[action]

            Kctl=np.minimum(Kexo+restore,static.K0)
            A,_,_=_training_solve_accessibility(engine,Kctl,use_surrogate)
            _,La,Le,_,_=loss_components(
                A,A0_train,static.population,static.vulnerable
            )
            Z+=(static.gamma**t)*La
            Q+=(static.gamma**t)*Le

        return float(Z),float(Q),records

    def val_objective():
        Z=[]
        Q=[]
        model.eval()
        with torch.no_grad():
            for sid in val_ids[:min(int(pcfg.get("validation_scenarios_training",12)),len(val_ids))]:
                z,q,_=rollout_collect(int(sid),False)
                Z.append(z)
                Q.append(q)
        Z=np.asarray(Z)
        Q=np.asarray(Q)
        return float(
            Z.mean()+lamE*Q.mean()+lamR*cvar_empirical(Z,alpha)
        )

    best_state=None
    best_val=np.inf
    bad=0
    history=[]
    epochs=int(pcfg["epochs"])
    eppe=int(pcfg["episodes_per_epoch"])

    for epoch in range(1,epochs+1):
        epoch_t0=time.perf_counter()
        model.eval()
        batch_ids=rng.choice(
            train_ids,size=min(eppe,len(train_ids)),replace=False
        )
        trajectories=[
            rollout_collect(int(sid),True) for sid in batch_ids
        ]
        Z=np.asarray([x[0] for x in trajectories],dtype=float)
        Q=np.asarray([x[1] for x in trajectories],dtype=float)
        eta=float(np.quantile(Z,alpha))
        traj_cost=(
            Z+lamE*Q
            +lamR*np.maximum(Z-eta,0.0)/max(1e-9,1-alpha)
        )
        advantages=traj_cost-traj_cost.mean()
        nrecords=sum(len(x[2]) for x in trajectories)

        for update_idx in range(1,update_epochs+1):
            if nrecords==0:
                continue
            model.train()
            opt.zero_grad(set_to_none=True)

            for adv_cost,(_,_,records) in zip(advantages,trajectories):
                reward_adv=-float(adv_cost)
                for rec in records:
                    dist,_=_graph_action_distribution(
                        cfg,model,st,static,
                        rec["state"],rec["feasible"],device
                    )
                    j=torch.tensor(
                        rec["action_index"],dtype=torch.long,device=device
                    )
                    new_logp=dist.log_prob(j)
                    old_logp=torch.tensor(
                        rec["old_logp"],
                        dtype=new_logp.dtype,
                        device=device,
                    )
                    ratio=torch.exp(new_logp-old_logp)
                    adv=torch.tensor(
                        reward_adv,dtype=new_logp.dtype,device=device
                    )
                    s1=ratio*adv
                    s2=torch.clamp(
                        ratio,1.0-clip_eps,1.0+clip_eps
                    )*adv
                    surrogate=torch.minimum(s1,s2)
                    loss_term=(
                        -surrogate/nrecords
                        - entropy_coef*dist.entropy()/nrecords
                    )
                    loss_term.backward()
                    del dist,new_logp,old_logp,ratio,adv,s1,s2,surrogate,loss_term

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),float(pcfg["gradient_clip"])
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type=="cuda":
                torch.cuda.empty_cache()

        if device.type=="cuda":
            free,total=torch.cuda.mem_get_info(device)
            mem_txt=f", CUDA free={free/1024**3:.2f} GiB"
        else:
            mem_txt=""

        if log:
            log.log(
                f"seed={seed} B5_PPO epoch={epoch}/{epochs}: "
                f"train Z={Z.mean():.6f}, Q={Q.mean():.6f}, "
                f"records={nrecords}, PPO_updates={update_epochs}, "
                f"elapsed={time.perf_counter()-epoch_t0:.1f}s{mem_txt}"
            )

        if epoch%int(pcfg["validation_every"])==0:
            vj=val_objective()
            history.append({
                "epoch":epoch,"val_J":vj,
                "train_Z":float(Z.mean()),
                "train_Q":float(Q.mean()),
                "optimizer":"PPO",
            })
            if log:
                log.log(
                    f"seed={seed} B5_PPO validation epoch={epoch}: "
                    f"val J={vj:.6f}"
                )
            if vj<best_val-1e-6:
                best_val=vj
                best_state={
                    k:v.detach().cpu().clone()
                    for k,v in model.state_dict().items()
                }
                bad=0
            else:
                bad+=1
            if bad>=int(pcfg["early_stopping_patience"]):
                break

    if best_state:
        model.load_state_dict(best_state)
    return model,st,pd.DataFrame(history),best_val



def final_action_protocol_signature(cfg):
    return {
        "candidate_screening_rule": cfg["intervention"]["candidate_screening_rule"],
        "max_candidate_edges_per_step": int(
            cfg["intervention"]["max_candidate_edges_per_step"]
        ),
        "candidate_deficit_tolerance": float(
            cfg["intervention"]["candidate_deficit_tolerance"]
        ),
        "restoration_fraction_of_remaining_deficit": float(
            cfg["intervention"]["restoration_fraction_of_remaining_deficit"]
        ),
    }


def enforce_final_action_checkpoint_protocol(cfg, paths, allow_initialize=False):
    """Prevent silent reuse of pre-v1.0.30 learned checkpoints."""
    sig = final_action_protocol_signature(cfg)
    marker = paths.models/"final_action_protocol.json"
    learned = list(paths.models.glob("B5_seed_*.pt")) + list(
        paths.models.glob("B5_PPO_seed_*.pt")
    )

    if marker.exists():
        old = read_json(marker)
        if old != sig:
            raise RuntimeError(
                "Learned checkpoints use a different candidate-action protocol. "
                "Run --stage train --reset-training before final retraining."
            )
        return sig

    if learned and not allow_initialize:
        raise RuntimeError(
            "Existing B5/B5_PPO checkpoints predate the final R3/Top-160 action "
            "correspondence. Refusing silent reuse. Run "
            "--stage train --reset-training after the smoke test passes."
        )

    if allow_initialize and not learned:
        write_json(marker, sig)
    return sig


def train_graph_policies(cfg,paths,log):
    enforce_final_action_checkpoint_protocol(cfg,paths,allow_initialize=True)
    seeds=active_numerical_seeds(cfg)
    log.log(
        f"Graph-policy training START: {len(seeds)} numerical seeds {seeds}; "
        "B5-PG plus configured PPO robustness"
    )
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(cfg,paths,edges,log)
    log.log(f"Final training protocol v1.0.30: action_rule={cfg['intervention']['candidate_screening_rule']}, candidate_cap={cfg['intervention']['max_candidate_edges_per_step']}, seeds={len(seeds)}, PG_epochs={cfg['graph_policy']['epochs']}, PPO_epochs={cfg['graph_ppo']['epochs']}, episodes/epoch={cfg['graph_policy']['episodes_per_epoch']}, validation_scenarios={cfg['graph_policy']['validation_scenarios_training']}, internal_accessibility=candidate_path_surrogate, reported_accessibility=exact_full_network")
    audit_training_surrogate(cfg,paths,engine,static,log)
    all_hist=[]
    torch=require_torch()
    train_started=time.perf_counter()
    for seed_pos,seed in enumerate(seeds,start=1):
        seed_started=time.perf_counter()
        log.log(f"B5-PG seed {seed_pos}/{len(seeds)}: seed={seed} START")
        pg_path=paths.models/f"B5_seed_{seed}.pt"
        if not pg_path.exists():
            model,st,hist,v=train_one_graph_seed(cfg,paths,engine,static,edges,zones,seed,None,log)
            torch.save(model.state_dict(),pg_path)
            hist["seed"]=seed;hist["spec"]="B5";all_hist.append(hist)
        else:
            log.log(f"Reuse trained B5 policy seed={seed}: {pg_path.name}")
        log.log(_progress_message(
            "B5-PG seeds",seed_pos,len(seeds),train_started,
            time.perf_counter()-seed_started
        ))
    if all_hist:
        pd.concat(all_hist,ignore_index=True).to_csv(paths.eval/"training_history_B5.csv",index=False)
    log.log(f"B5 episodic policy-gradient models ready for {len(seeds)} numerical seeds")

    if bool(cfg.get("graph_ppo",{}).get("enabled",False)):
        ppo_hist=[]
        ppo_started=time.perf_counter()
        for seed_pos,seed in enumerate(seeds,start=1):
            seed_started=time.perf_counter()
            log.log(f"B5-PPO seed {seed_pos}/{len(seeds)}: seed={seed} START")
            ppo_path=paths.models/f"B5_PPO_seed_{seed}.pt"
            if ppo_path.exists():
                log.log(f"Reuse trained B5_PPO policy seed={seed}: {ppo_path.name}")
                continue
            model,st,hist,v=train_one_graph_ppo_seed(
                cfg,paths,engine,static,edges,zones,seed,log
            )
            torch.save(model.state_dict(),ppo_path)
            hist["seed"]=seed;hist["spec"]="B5_PPO";ppo_hist.append(hist)
            log.log(_progress_message(
                "B5-PPO seeds",seed_pos,len(seeds),ppo_started,
                time.perf_counter()-seed_started
            ))
        if ppo_hist:
            pd.concat(ppo_hist,ignore_index=True).to_csv(
                paths.eval/"training_history_B5_PPO.csv",index=False
            )
        log.log(f"B5_PPO algorithmic-robustness models ready for {len(seeds)} numerical seeds")


def final_action_space_smoke_test(cfg, paths, log):
    """One-seed PG/PPO smoke test for the locked R3/Top-160 action space.

    This test writes only SMOKE_* artifacts. It never overwrites B5/B5_PPO
    production checkpoints. It checks:
      * candidate-set size and R3/path relevance;
      * finite normalized policy probabilities;
      * nonzero finite gradient path through the local graph policy;
      * nontrivial parameter movement after short PG and PPO optimization;
      * finite validation objective;
      * CUDA peak-memory use when available.
    """
    torch=require_torch()
    cfgs=json.loads(json.dumps(cfg))
    seed=int(active_numerical_seeds(cfgs)[0])

    # Deliberately short: enough to exercise rollout, replay/backprop,
    # validation, candidate masking, and PPO clipping.
    cfgs["graph_policy"].update({
        "epochs":2,
        "episodes_per_epoch":3,
        "validation_every":1,
        "early_stopping_patience":10,
        "validation_scenarios_training":4,
    })
    cfgs["graph_ppo"].update({
        "epochs":2,
        "episodes_per_epoch":3,
        "validation_every":1,
        "early_stopping_patience":10,
        "validation_scenarios_training":4,
        "update_epochs":2,
    })

    engine,static,edges,zones,_=build_reference_engine(cfgs,paths,log)
    static.betweenness=load_or_compute_betweenness(cfgs,paths,edges,log)
    costs=intervention_costs(cfgs,edges)
    B=baseline_budget(cfgs,costs)

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids=meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    check_ids=val_ids[:min(4,len(val_ids))]
    if len(check_ids)==0:
        raise RuntimeError("Smoke test requires validation scenarios.")

    # Candidate correspondence sanity on deterministic validation t=0 states.
    candidate_rows=[]
    states=[]
    for sid in check_ids:
        d=np.load(paths.scenarios/f"scenario_{int(sid):05d}.npz")["degradation"]
        Kpre=(1.0-d[0])*static.K0
        feas=policy_candidate_edges(cfgs,engine,Kpre,static.K0,costs,B)
        abs_def=np.maximum(static.K0-Kpre,0.0)
        pu=engine.path_use_count
        states.append((int(sid),d[0],Kpre,feas))
        candidate_rows.append({
            "scenario_id":int(sid),
            "t":0,
            "n_candidates":int(len(feas)),
            "cap":int(cfgs["intervention"]["max_candidate_edges_per_step"]),
            "all_path_relevant":bool(np.all(pu[feas]>0)) if len(feas) else True,
            "all_physically_degraded":bool(
                np.all(abs_def[feas] > cfgs["intervention"]["candidate_deficit_tolerance"])
            ) if len(feas) else True,
            "all_budget_feasible":bool(np.all(costs[feas] <= B+1e-12)) if len(feas) else True,
            "max_r3_score":float(np.max(abs_def[feas]*pu[feas])) if len(feas) else 0.0,
            "min_r3_score":float(np.min(abs_def[feas]*pu[feas])) if len(feas) else 0.0,
        })
    cand_df=pd.DataFrame(candidate_rows)
    cand_df.to_csv(paths.eval/"final_action_smoke_candidates.csv",index=False)

    GraphPolicyNet=make_graph_policy_class()

    def initial_model_state():
        set_global_seed(seed,cfgs["compute"]["deterministic_torch"])
        m=GraphPolicyNet(
            cfgs["graph_policy"]["hidden_dim"],
            cfgs["graph_policy"]["message_layers"],
            False,False
        ).to(engine.device)
        return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

    init_sd=initial_model_state()

    def movement(model):
        cur=model.state_dict()
        num=0.0; den=0.0
        for k,v0 in init_sd.items():
            v=cur[k].detach().cpu().double()
            b=v0.double()
            num+=float(((v-b)**2).sum())
            den+=float((b*b).sum())
        return float(np.sqrt(num)/max(np.sqrt(den),1e-30))

    car_node=pd.read_csv(paths.processed/"training_car_node_feature.csv")
    cfg_local=json.loads(json.dumps(cfgs))
    cfg_local["_paths"]={"processed":str(paths.processed)}
    st_eval=build_graph_tensors(cfg_local,edges,zones,car_node,engine.device)

    def policy_numeric_checks(model, label):
        model.eval()
        rows=[]
        grad_norms=[]
        for sid,deg0,Kpre,feas in states:
            state={
                "t":0,"budget":B,"B":B,"Kpre":Kpre,
                "K0":static.K0,"degradation":deg0,
            }
            model.zero_grad(set_to_none=True)
            edge_logits,no_logit=graph_policy_logits(
                model,st_eval,state,cfgs,engine.device,edge_indices=feas
            )
            logits=torch.cat([edge_logits,no_logit.view(1)])
            probs=torch.softmax(logits,dim=0)
            entropy=float(
                (-(probs*torch.log(torch.clamp(probs,min=1e-30))).sum()).detach().cpu()
            )
            prob_sum=float(probs.sum().detach().cpu())
            max_prob=float(probs.max().detach().cpu())
            no_action_prob=float(probs[-1].detach().cpu())

            # Differentiability smoke: negative log probability of current argmax.
            j=torch.argmax(logits.detach())
            loss=-torch.log_softmax(logits,dim=0)[j]
            loss.backward()
            gn=0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += float((p.grad.detach().double()**2).sum().cpu())
            gn=float(np.sqrt(gn))
            grad_norms.append(gn)
            model.zero_grad(set_to_none=True)

            rows.append({
                "optimizer":label,
                "scenario_id":sid,
                "n_candidates":int(len(feas)),
                "probability_sum":prob_sum,
                "probability_sum_error":abs(prob_sum-1.0),
                "entropy":entropy,
                "max_probability":max_prob,
                "no_action_probability":no_action_prob,
                "gradient_path_norm":gn,
                "all_logits_finite":bool(torch.isfinite(logits).all().item()),
                "all_probabilities_finite":bool(torch.isfinite(probs).all().item()),
            })
        return pd.DataFrame(rows), grad_norms

    results={}
    diag_frames=[]

    # PG smoke
    if engine.device.type=="cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(engine.device)
    t0=time.perf_counter()
    pg_model,_,pg_hist,pg_best=train_one_graph_seed(
        cfgs,paths,engine,static,edges,zones,seed,None,log
    )
    pg_seconds=time.perf_counter()-t0
    pg_peak=(
        int(torch.cuda.max_memory_allocated(engine.device))
        if engine.device.type=="cuda" else 0
    )
    pg_diag,pg_grad=policy_numeric_checks(pg_model,"PG")
    diag_frames.append(pg_diag)
    torch.save(pg_model.state_dict(),paths.models/f"SMOKE_B5_PG_seed_{seed}.pt")
    pg_hist.to_csv(paths.eval/"final_action_smoke_history_PG.csv",index=False)
    results["PG"]={
        "elapsed_seconds":float(pg_seconds),
        "best_validation_J":float(pg_best),
        "parameter_relative_movement":movement(pg_model),
        "peak_cuda_memory_bytes":pg_peak,
        "min_gradient_path_norm":float(np.min(pg_grad)),
        "max_probability_sum_error":float(pg_diag.probability_sum_error.max()),
        "mean_entropy":float(pg_diag.entropy.mean()),
        "mean_max_probability":float(pg_diag.max_probability.mean()),
        "mean_no_action_probability":float(pg_diag.no_action_probability.mean()),
    }

    # PPO smoke
    if engine.device.type=="cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(engine.device)
    t0=time.perf_counter()
    ppo_model,_,ppo_hist,ppo_best=train_one_graph_ppo_seed(
        cfgs,paths,engine,static,edges,zones,seed,log
    )
    ppo_seconds=time.perf_counter()-t0
    ppo_peak=(
        int(torch.cuda.max_memory_allocated(engine.device))
        if engine.device.type=="cuda" else 0
    )
    ppo_diag,ppo_grad=policy_numeric_checks(ppo_model,"PPO")
    diag_frames.append(ppo_diag)
    torch.save(ppo_model.state_dict(),paths.models/f"SMOKE_B5_PPO_seed_{seed}.pt")
    ppo_hist.to_csv(paths.eval/"final_action_smoke_history_PPO.csv",index=False)
    results["PPO"]={
        "elapsed_seconds":float(ppo_seconds),
        "best_validation_J":float(ppo_best),
        "parameter_relative_movement":movement(ppo_model),
        "peak_cuda_memory_bytes":ppo_peak,
        "min_gradient_path_norm":float(np.min(ppo_grad)),
        "max_probability_sum_error":float(ppo_diag.probability_sum_error.max()),
        "mean_entropy":float(ppo_diag.entropy.mean()),
        "mean_max_probability":float(ppo_diag.max_probability.mean()),
        "mean_no_action_probability":float(ppo_diag.no_action_probability.mean()),
    }

    diag=pd.concat(diag_frames,ignore_index=True)
    diag.to_csv(paths.eval/"final_action_smoke_policy_diagnostics.csv",index=False)

    checks={
        "candidate_cap_is_160":
            int(cfgs["intervention"]["max_candidate_edges_per_step"])==160,
        "all_candidate_counts_within_cap":
            bool((cand_df.n_candidates<=160).all()),
        "all_candidates_path_relevant":
            bool(cand_df.all_path_relevant.all()),
        "all_candidates_physically_degraded":
            bool(cand_df.all_physically_degraded.all()),
        "all_candidates_budget_feasible":
            bool(cand_df.all_budget_feasible.all()),
        "all_logits_finite":
            bool(diag.all_logits_finite.all()),
        "all_probabilities_finite":
            bool(diag.all_probabilities_finite.all()),
        "probabilities_normalized":
            bool((diag.probability_sum_error<1e-6).all()),
        "gradient_paths_nonzero_finite":
            bool(np.isfinite(diag.gradient_path_norm).all()
                 and (diag.gradient_path_norm>0).all()),
        "pg_parameter_movement":
            bool(np.isfinite(results["PG"]["parameter_relative_movement"])
                 and results["PG"]["parameter_relative_movement"]>1e-8),
        "ppo_parameter_movement":
            bool(np.isfinite(results["PPO"]["parameter_relative_movement"])
                 and results["PPO"]["parameter_relative_movement"]>1e-8),
        "pg_validation_finite":bool(np.isfinite(pg_best)),
        "ppo_validation_finite":bool(np.isfinite(ppo_best)),
    }
    passed=bool(all(checks.values()))

    out={
        "script_version":SCRIPT_VERSION,
        "production_checkpoints_modified":False,
        "seed":seed,
        "final_action_protocol":final_action_protocol_signature(cfgs),
        "smoke_training":{
            "PG_epochs":cfgs["graph_policy"]["epochs"],
            "PPO_epochs":cfgs["graph_ppo"]["epochs"],
            "episodes_per_epoch":cfgs["graph_policy"]["episodes_per_epoch"],
            "validation_scenarios":cfgs["graph_policy"]["validation_scenarios_training"],
        },
        "optimizers":results,
        "checks":checks,
        "overall_pass":passed,
        "next_step":(
            "run_final_retraining_with_reset_training"
            if passed else "do_not_retrain_fix_failed_smoke_checks"
        ),
    }
    write_json(paths.eval/"final_action_space_smoke_summary.json",out)

    log.log(
        f"FINAL ACTION-SPACE SMOKE {'PASS' if passed else 'FAIL'}: "
        f"R3/Top-{cfgs['intervention']['max_candidate_edges_per_step']}; "
        f"PG movement={results['PG']['parameter_relative_movement']:.3e}, "
        f"PPO movement={results['PPO']['parameter_relative_movement']:.3e}; "
        f"PG valJ={pg_best:.12g}, PPO valJ={ppo_best:.12g}; "
        f"PG peakCUDA={pg_peak/1024**3:.2f} GiB, "
        f"PPO peakCUDA={ppo_peak/1024**3:.2f} GiB."
    )
    if not passed:
        failed=[k for k,v in checks.items() if not v]
        raise RuntimeError("Final action-space smoke failed: "+", ".join(failed))
    return out


# -----------------------------------------------------------------------------
# Learning-integrity audit
# -----------------------------------------------------------------------------

def _state_dict_l2_distance(a, b):
    """Euclidean parameter distance between two compatible state dictionaries."""
    s = 0.0
    for k in a:
        da = a[k].detach().cpu().double()
        db = b[k].detach().cpu().double()
        s += float(((da-db)**2).sum())
    return math.sqrt(max(s, 0.0))


def _model_parameter_l2(model):
    s = 0.0
    for p in model.parameters():
        s += float((p.detach().cpu().double()**2).sum())
    return math.sqrt(max(s, 0.0))


def _policy_state_metrics(cfg, model, st, static, state, feasible, device):
    """Policy probabilities and entropy at one fixed decision state."""
    torch = require_torch()
    with torch.no_grad():
        dist, cand = _graph_action_distribution(
            cfg, model, st, static, state, feasible, device
        )
        probs = dist.probs.detach().cpu().numpy().astype(float)
        logits = dist.logits.detach().cpu().numpy().astype(float)
    ent = float(-(probs*np.log(np.maximum(probs,1e-300))).sum())
    j = int(np.argmax(probs))
    no_idx = len(cand)-1
    return {
        "candidate_actions": cand,
        "probs": probs,
        "logits": logits,
        "entropy": ent,
        "no_action_probability": float(probs[no_idx]),
        "greedy_action": int(cand[j]),
        "greedy_probability": float(probs[j]),
    }


def _rank_agreement(x, y):
    """Spearman/Kendall rank agreement with safe finite handling."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 2:
        return float("nan"), float("nan")
    sx = pd.Series(x).rank(method="average").to_numpy(float)
    sy = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(sx) <= 1e-15 or np.std(sy) <= 1e-15:
        rho = float("nan")
    else:
        rho = float(np.corrcoef(sx, sy)[0,1])
    try:
        from scipy.stats import kendalltau
        tau = float(kendalltau(x, y, nan_policy="omit").statistic)
    except Exception:
        tau = float("nan")
    return rho, tau


def learning_integrity_audit(cfg, paths, log):
    """Audit whether learned checkpoints differ from initialization and whether
    intervention actions have decision-relevant consequences.

    The audit is deliberately diagnostic:
      * existing B5/B5_PPO checkpoints are read-only;
      * no optimizer step is taken;
      * exact full-network accessibility is used for the action-effect benchmark;
      * candidate-path accessibility is evaluated on the same routed states to
        test action-ranking preservation.
    """
    torch = require_torch()
    seeds = active_numerical_seeds(cfg)
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
    static.betweenness = load_or_compute_betweenness(cfg, paths, edges, log)
    device = engine.device

    meta = pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids = meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    audit_val_ids = val_ids[:min(4, len(val_ids))]

    costs = intervention_costs(cfg, edges)
    B = baseline_budget(cfg, costs)
    frac = float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    max_candidates = int(cfg["intervention"]["max_candidate_edges_per_step"])
    A0_sur = engine.reference_accessibility_surrogate()

    GraphPolicyNet = make_graph_policy_class()

    # -----------------------------------------------------------------
    # A. Checkpoint movement + fixed-state policy diagnostics
    # -----------------------------------------------------------------
    param_rows = []
    state_rows = []
    action_signature = {}

    # fixed validation states are generated without intervention at t=0
    fixed_states = {}
    fixed_feasible = {}
    for sid in audit_val_ids:
        d = np.load(paths.scenarios/f"scenario_{int(sid):05d}.npz")["degradation"]
        Kpre = (1.0-d[0])*static.K0
        feas = policy_candidate_edges(
            cfg, engine, Kpre, static.K0, costs, B, max_candidates
        )
        fixed_states[int(sid)] = {
            "t":0, "budget":B, "B":B, "Kpre":Kpre,
            "K0":static.K0, "degradation":d[0],
        }
        fixed_feasible[int(sid)] = np.asarray(feas,dtype=np.int64)

    for spec in ("B5","B5_PPO"):
        for seed in seeds:
            ckpt = paths.models/f"{spec}_seed_{seed}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint required by learning audit: {ckpt}"
                )

            # Load trained checkpoint.
            trained, st = load_graph_model(
                cfg, paths, engine, edges, zones, seed,
                spec=spec
            )
            trained_sd = {
                k:v.detach().cpu().clone()
                for k,v in trained.state_dict().items()
            }

            # Reconstruct the deterministic initialization used by this seed.
            set_global_seed(seed, cfg["compute"]["deterministic_torch"])
            init_model = GraphPolicyNet(
                cfg["graph_policy"]["hidden_dim"],
                cfg["graph_policy"]["message_layers"],
                False, False
            ).to(device)
            init_sd = {
                k:v.detach().cpu().clone()
                for k,v in init_model.state_dict().items()
            }
            delta = _state_dict_l2_distance(trained_sd, init_sd)
            init_norm = _model_parameter_l2(init_model)
            trained_norm = _model_parameter_l2(trained)
            param_rows.append({
                "spec":spec, "seed":seed,
                "parameter_delta_l2":delta,
                "initial_parameter_l2":init_norm,
                "trained_parameter_l2":trained_norm,
                "relative_parameter_delta":delta/max(init_norm,1e-30),
            })

            sig = []
            for sid in audit_val_ids:
                m = _policy_state_metrics(
                    cfg, trained, st, static,
                    fixed_states[int(sid)],
                    fixed_feasible[int(sid)],
                    device
                )
                sig.append(m["greedy_action"])
                state_rows.append({
                    "spec":spec, "seed":seed, "scenario_id":int(sid), "t":0,
                    "n_feasible":int(len(fixed_feasible[int(sid)])),
                    "entropy":m["entropy"],
                    "no_action_probability":m["no_action_probability"],
                    "greedy_action":m["greedy_action"],
                    "greedy_probability":m["greedy_probability"],
                    "max_logit":float(np.max(m["logits"])),
                    "min_logit":float(np.min(m["logits"])),
                    "logit_range":float(np.max(m["logits"])-np.min(m["logits"])),
                })
            action_signature[(spec,seed)] = tuple(sig)

            del trained, init_model, st
            if device.type=="cuda":
                torch.cuda.empty_cache()

    param_df = pd.DataFrame(param_rows)
    state_df = pd.DataFrame(state_rows)
    param_df.to_csv(paths.eval/"learning_audit_parameter_movement.csv",index=False)
    state_df.to_csv(paths.eval/"learning_audit_fixed_state_policy.csv",index=False)

    # Agreement of deterministic actions across seeds and optimizers.
    agree_rows = []
    for spec in ("B5","B5_PPO"):
        sigs = [action_signature[(spec,s)] for s in seeds]
        for pos,sid in enumerate(audit_val_ids):
            vals = [x[pos] for x in sigs]
            mode_action = pd.Series(vals).mode().iloc[0]
            agree_rows.append({
                "spec":spec,"scenario_id":int(sid),
                "mode_action":int(mode_action),
                "mode_share":float(np.mean(np.asarray(vals)==mode_action)),
                "n_unique_actions":int(len(set(vals))),
            })
    for seed in seeds:
        a = action_signature[("B5",seed)]
        b = action_signature[("B5_PPO",seed)]
        agree_rows.append({
            "spec":"PG_vs_PPO","scenario_id":-1,
            "mode_action":-999,
            "mode_share":float(np.mean(np.asarray(a)==np.asarray(b))),
            "n_unique_actions":int(sum(x!=y for x,y in zip(a,b))),
            "seed":seed,
        })
    agree_df = pd.DataFrame(agree_rows)
    agree_df.to_csv(paths.eval/"learning_audit_action_agreement.csv",index=False)

    # -----------------------------------------------------------------
    # B. Action-effect + surrogate/exact ranking audit
    # -----------------------------------------------------------------
    action_rows = []
    ranking_rows = []
    # 8 feasible actions per state is enough to distinguish a flat environment
    # from a decision-relevant one while keeping exact Dijkstra cost bounded.
    n_actions = 8
    for sid in audit_val_ids:
        sid = int(sid)
        d = np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        Kexo = (1.0-d[0])*static.K0
        feas = fixed_feasible[sid]

        if len(feas) > n_actions:
            # Deterministic coverage across the already ordered candidate list.
            pick = np.unique(
                np.linspace(0,len(feas)-1,n_actions,dtype=int)
            )
            audit_actions = list(map(int,feas[pick]))
        else:
            audit_actions = list(map(int,feas))
        audit_actions = [-1] + audit_actions

        for action in audit_actions:
            Kctl = Kexo.copy()
            if action >= 0:
                deficit = max(static.K0[action]-Kctl[action],0.0)
                Kctl[action] += frac*deficit
                Kctl[action] = min(Kctl[action],static.K0[action])

            _, q, pc = engine.solve(Kctl)
            A_exact = engine.accessibility_exact(q,Kctl)
            A_sur, _ = engine.accessibility_from_path_cost(pc)
            _, Lex, Eex, _, _ = loss_components(
                A_exact,static.A0,static.population,static.vulnerable
            )
            _, Lsu, Esu, _, _ = loss_components(
                A_sur,A0_sur,static.population,static.vulnerable
            )
            action_rows.append({
                "scenario_id":sid,"t":0,"action":int(action),
                "is_no_action":bool(action<0),
                "Lacc_exact":Lex,"Leq_exact":Eex,
                "Lacc_surrogate":Lsu,"Leq_surrogate":Esu,
                "abs_error_Lacc":abs(Lsu-Lex),
            })

        sdf = pd.DataFrame([r for r in action_rows if r["scenario_id"]==sid])
        base_exact = float(sdf.loc[sdf.is_no_action,"Lacc_exact"].iloc[0])
        base_sur = float(sdf.loc[sdf.is_no_action,"Lacc_surrogate"].iloc[0])
        sdf_non = sdf.loc[~sdf.is_no_action].copy()
        exact_gain = base_exact-sdf_non["Lacc_exact"].to_numpy(float)
        sur_gain = base_sur-sdf_non["Lacc_surrogate"].to_numpy(float)
        rho,tau = _rank_agreement(exact_gain,sur_gain)

        k = min(3,len(sdf_non))
        if k:
            exact_top = set(
                sdf_non.iloc[np.argsort(-exact_gain)[:k]]["action"].astype(int)
            )
            sur_top = set(
                sdf_non.iloc[np.argsort(-sur_gain)[:k]]["action"].astype(int)
            )
            overlap = len(exact_top & sur_top)/k
        else:
            overlap = float("nan")

        ranking_rows.append({
            "scenario_id":sid,
            "n_actions":int(len(sdf_non)),
            "exact_action_gain_range":float(
                np.max(exact_gain)-np.min(exact_gain)
            ) if len(exact_gain) else 0.0,
            "exact_best_gain":float(np.max(exact_gain)) if len(exact_gain) else 0.0,
            "exact_worst_gain":float(np.min(exact_gain)) if len(exact_gain) else 0.0,
            "surrogate_best_gain":float(np.max(sur_gain)) if len(sur_gain) else 0.0,
            "spearman_gain_rank":rho,
            "kendall_gain_rank":tau,
            "top3_overlap":overlap,
        })

        log.log(
            f"Learning audit scenario={sid}: "
            f"exact action-gain range={ranking_rows[-1]['exact_action_gain_range']:.12g}; "
            f"best exact gain={ranking_rows[-1]['exact_best_gain']:.12g}; "
            f"Spearman={rho:.6g}; Kendall={tau:.6g}; top-{k} overlap={overlap:.3f}"
        )

    action_df = pd.DataFrame(action_rows)
    rank_df = pd.DataFrame(ranking_rows)
    action_df.to_csv(paths.eval/"learning_audit_action_effects.csv",index=False)
    rank_df.to_csv(paths.eval/"learning_audit_surrogate_action_ranking.csv",index=False)

    # -----------------------------------------------------------------
    # C. Existing training-history precision / flatness audit
    # -----------------------------------------------------------------
    hist_rows = []
    for spec,fn in (
        ("B5","training_history_B5.csv"),
        ("B5_PPO","training_history_B5_PPO.csv"),
    ):
        fp = paths.eval/fn
        if fp.exists():
            h = pd.read_csv(fp)
            for seed,g in h.groupby("seed"):
                vals = g["val_J"].to_numpy(float)
                hist_rows.append({
                    "spec":spec,"seed":int(seed),
                    "n_validations":int(len(vals)),
                    "val_J_min":float(np.min(vals)),
                    "val_J_max":float(np.max(vals)),
                    "val_J_range":float(np.ptp(vals)),
                    "val_J_sd":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,
                })
    hist_df = pd.DataFrame(hist_rows)
    hist_df.to_csv(paths.eval/"learning_audit_validation_flatness.csv",index=False)

    # Conservative machine-readable conclusion.
    min_rel_move = float(param_df["relative_parameter_delta"].min())
    max_val_range = float(hist_df["val_J_range"].max()) if len(hist_df) else float("nan")
    max_action_range = float(rank_df["exact_action_gain_range"].max()) if len(rank_df) else float("nan")
    mean_top3 = float(rank_df["top3_overlap"].mean()) if len(rank_df) else float("nan")
    mean_rho = float(rank_df["spearman_gain_rank"].mean()) if len(rank_df) else float("nan")

    summary = {
        "script_version":SCRIPT_VERSION,
        "checkpoints_retrained":False,
        "n_seeds":len(seeds),
        "audit_validation_scenarios":[int(x) for x in audit_val_ids],
        "min_relative_parameter_delta":min_rel_move,
        "max_validation_J_range":max_val_range,
        "max_exact_action_gain_range":max_action_range,
        "mean_surrogate_exact_spearman":mean_rho,
        "mean_top3_action_overlap":mean_top3,
        "interpretation_flags":{
            "parameters_changed_from_initialization":bool(min_rel_move>1e-10),
            "validation_logged_as_flat":bool(np.isfinite(max_val_range) and max_val_range<=1e-12),
            "environment_has_detectable_action_effect":bool(np.isfinite(max_action_range) and max_action_range>1e-12),
            "surrogate_rank_agreement_high":bool(
                np.isfinite(mean_rho) and mean_rho>=0.8 and
                np.isfinite(mean_top3) and mean_top3>=2/3
            ),
        },
    }
    write_json(paths.eval/"learning_integrity_audit_summary.json",summary)

    log.log(
        "Learning-integrity audit DONE: "
        f"min relative parameter movement={min_rel_move:.3e}; "
        f"max val-J range={max_val_range:.12g}; "
        f"max exact action-gain range={max_action_range:.12g}; "
        f"mean surrogate/exact Spearman={mean_rho:.6g}; "
        f"mean top-3 overlap={mean_top3:.3f}"
    )
    log.log(
        "Audit flags: "
        + json.dumps(summary["interpretation_flags"],sort_keys=True)
    )
    return summary


# -----------------------------------------------------------------------------
# Intervention-scale audit
# -----------------------------------------------------------------------------

def intervention_scale_audit(cfg, paths, log):
    """Read-only audit of the decision unit before any further policy training.

    Compares the current partial single-edge action with stronger but still
    mechanically transparent intervention units on identical validation states:
      S1 current_partial_edge : restore configured fraction of one candidate edge;
      S2 full_edge            : restore that edge to its reference capacity;
      S3 local_bundle_partial : configured partial restoration of all currently
                                degraded edges sharing an endpoint with candidate;
      S4 local_bundle_full    : full restoration of that same one-hop bundle.

    This stage does not select a preferred scale and does not retrain a policy.
    It reports exact full-network accessibility consequences, intervention size,
    and candidate-set truncation diagnostics.
    """
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
    static.betweenness = load_or_compute_betweenness(cfg, paths, edges, log)

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids=meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    audit_ids=val_ids[:min(4,len(val_ids))]
    if len(audit_ids)==0:
        raise RuntimeError("No validation scenarios available for intervention-scale audit.")

    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    cap=int(cfg["intervention"]["max_candidate_edges_per_step"])

    # Exact OSM topological incidence, computed once. Only currently degraded
    # edges enter a bundle; unaffected neighboring edges are never "restored".
    u=np.asarray([str(x) for x in edges.u],dtype=object)
    v=np.asarray([str(x) for x in edges.v],dtype=object)
    incident=defaultdict(list)
    for ei,(uu,vv) in enumerate(zip(u,v)):
        incident[uu].append(ei)
        incident[vv].append(ei)

    def one_hop_degraded_bundle(center, degraded_mask):
        ids=set([int(center)])
        for node in (u[int(center)],v[int(center)]):
            ids.update(int(e) for e in incident[node] if degraded_mask[int(e)])
        return np.asarray(sorted(ids),dtype=np.int64)

    def apply_restore(Kexo, ids, fraction):
        K=Kexo.copy()
        ids=np.asarray(ids,dtype=np.int64)
        deficit=np.maximum(static.K0[ids]-K[ids],0.0)
        K[ids]=np.minimum(K[ids]+float(fraction)*deficit,static.K0[ids])
        return K

    rows=[]
    trunc=[]
    n_actions=8

    for sid in audit_ids:
        sid=int(sid)
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        Kexo=(1.0-d[0])*static.K0
        degraded=(Kexo < static.K0-1e-9)

        all_feasible=np.where(degraded & (costs <= B+1e-12))[0]
        capped=feasible_edges(Kexo,static.K0,costs,B,cap,static.betweenness)
        trunc.append({
            "scenario_id":sid,
            "n_degraded_edges":int(degraded.sum()),
            "n_individually_budget_feasible":int(len(all_feasible)),
            "candidate_cap":cap,
            "n_policy_candidates":int(len(capped)),
            "candidate_retention_share":float(len(capped)/max(len(all_feasible),1)),
            "candidate_set_truncated":bool(len(all_feasible)>cap),
        })

        if len(capped)>n_actions:
            pick=np.unique(np.linspace(0,len(capped)-1,n_actions,dtype=int))
            centers=np.asarray(capped,dtype=np.int64)[pick]
        else:
            centers=np.asarray(capped,dtype=np.int64)

        # Common no-action reference: exact routing/accessibility.
        _,q0,_=engine.solve(Kexo)
        A_no=engine.accessibility_exact(q0,Kexo)
        _,L0,E0,_,_=loss_components(
            A_no,static.A0,static.population,static.vulnerable
        )

        for center in centers:
            center=int(center)
            bundle=one_hop_degraded_bundle(center,degraded)
            specs=[
                ("S1_current_partial_edge",np.asarray([center]),frac),
                ("S2_full_edge",np.asarray([center]),1.0),
                ("S3_local_bundle_partial",bundle,frac),
                ("S4_local_bundle_full",bundle,1.0),
            ]
            for scale,ids,rf in specs:
                Kctl=apply_restore(Kexo,ids,rf)
                _,q,_=engine.solve(Kctl)
                A=engine.accessibility_exact(q,Kctl)
                _,L,E,_,_=loss_components(
                    A,static.A0,static.population,static.vulnerable
                )
                raw_cost=float(costs[ids].sum())
                restored_capacity=float(
                    np.maximum(Kctl[ids]-Kexo[ids],0.0).sum()
                )
                rows.append({
                    "scenario_id":sid,
                    "center_edge":center,
                    "scale":scale,
                    "restoration_fraction":float(rf),
                    "n_edges_in_action":int(len(ids)),
                    "raw_action_cost":raw_cost,
                    "budget_ratio":raw_cost/max(B,1e-30),
                    "within_reference_budget":bool(raw_cost<=B+1e-12),
                    "restored_capacity_vph_sum":restored_capacity,
                    "Lacc_no_action":float(L0),
                    "Leq_no_action":float(E0),
                    "Lacc_exact":float(L),
                    "Leq_exact":float(E),
                    "gain_Lacc_exact":float(L0-L),
                    "gain_Leq_exact":float(E0-E),
                    "gain_Lacc_per_raw_cost":float((L0-L)/max(raw_cost,1e-30)),
                })

        sdf=pd.DataFrame([r for r in rows if r["scenario_id"]==sid])
        log.log(
            f"Intervention-scale audit scenario={sid}: "
            f"degraded={int(degraded.sum())}, individually feasible={len(all_feasible)}, "
            f"policy candidates={len(capped)}, retention={len(capped)/max(len(all_feasible),1):.3%}; "
            f"max exact gain by scale="
            + ", ".join(
                f"{k}:{v:.12g}" for k,v in
                sdf.groupby("scale")["gain_Lacc_exact"].max().to_dict().items()
            )
        )

    detail=pd.DataFrame(rows)
    trunc_df=pd.DataFrame(trunc)
    detail.to_csv(paths.eval/"intervention_scale_audit_detail.csv",index=False)
    trunc_df.to_csv(paths.eval/"intervention_candidate_truncation_audit.csv",index=False)

    # Aggregate without declaring a winner. Report effect magnitude, sign,
    # dispersion, action size, and budget feasibility separately.
    summary_rows=[]
    for scale,g in detail.groupby("scale",sort=False):
        gain=g["gain_Lacc_exact"].to_numpy(float)
        eq=g["gain_Leq_exact"].to_numpy(float)
        summary_rows.append({
            "scale":scale,
            "n_evaluations":int(len(g)),
            "mean_edges_in_action":float(g["n_edges_in_action"].mean()),
            "median_edges_in_action":float(g["n_edges_in_action"].median()),
            "mean_budget_ratio":float(g["budget_ratio"].mean()),
            "share_within_reference_budget":float(g["within_reference_budget"].mean()),
            "mean_gain_Lacc_exact":float(np.mean(gain)),
            "median_gain_Lacc_exact":float(np.median(gain)),
            "max_gain_Lacc_exact":float(np.max(gain)),
            "min_gain_Lacc_exact":float(np.min(gain)),
            "share_positive_gain_Lacc":float(np.mean(gain>1e-12)),
            "mean_gain_Leq_exact":float(np.mean(eq)),
            "mean_gain_Lacc_per_raw_cost":float(g["gain_Lacc_per_raw_cost"].mean()),
        })
    summary_df=pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.eval/"intervention_scale_audit_summary.csv",index=False)

    # Paired ratios relative to the current S1 definition, same scenario/center.
    piv=detail.pivot_table(
        index=["scenario_id","center_edge"],columns="scale",
        values="gain_Lacc_exact",aggfunc="first"
    ).reset_index()
    base="S1_current_partial_edge"
    ratio_rows=[]
    if base in piv.columns:
        for alt in ["S2_full_edge","S3_local_bundle_partial","S4_local_bundle_full"]:
            if alt not in piv.columns:
                continue
            for r in piv.itertuples(index=False):
                b=float(getattr(r,base)); a=float(getattr(r,alt))
                ratio_rows.append({
                    "scenario_id":int(r.scenario_id),
                    "center_edge":int(r.center_edge),
                    "alternative":alt,
                    "baseline_gain":b,
                    "alternative_gain":a,
                    "gain_difference":a-b,
                    "gain_ratio":a/b if abs(b)>1e-15 else float("nan"),
                })
    ratio_df=pd.DataFrame(ratio_rows)
    ratio_df.to_csv(paths.eval/"intervention_scale_paired_comparison.csv",index=False)

    out={
        "script_version":SCRIPT_VERSION,
        "policy_retrained":False,
        "audit_validation_scenarios":[int(x) for x in audit_ids],
        "candidate_cap":cap,
        "all_scenarios_candidate_set_truncated":bool(
            trunc_df["candidate_set_truncated"].all()
        ),
        "mean_candidate_retention_share":float(
            trunc_df["candidate_retention_share"].mean()
        ),
        "scales":summary_df.to_dict(orient="records"),
        "decision_rule":"diagnostic_only_no_scale_selected",
        "note":(
            "Bundle costs are reported against the current reference budget. "
            "A bundle exceeding that budget is not an admissible policy action "
            "under the current formal problem; it is retained only as a scale diagnostic."
        ),
    }
    write_json(paths.eval/"intervention_scale_audit.json",out)

    log.log(
        "Intervention-scale audit DONE: "
        f"mean candidate retention={out['mean_candidate_retention_share']:.3%}; "
        "no intervention scale selected automatically."
    )
    return out


# -----------------------------------------------------------------------------
# Hazard -> capacity -> candidate-set integrity audit
# -----------------------------------------------------------------------------

def hazard_capacity_candidate_audit(cfg, paths, log):
    """Read-only diagnostic of the physical signal reaching the policy.

    This audit changes neither the disruption process nor the intervention rule.
    For validation scenarios it traces:
        degradation d_e,t
        -> relative/absolute capacity deficit
        -> budget-feasible degraded edges
        -> policy candidate set.

    It reports distributional summaries, threshold exceedances, candidate
    retention, and candidate-vs-population contrasts. No policy is trained and
    no physical threshold is imposed by this stage.
    """
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
    static.betweenness = load_or_compute_betweenness(cfg, paths, edges, log)

    meta = pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids = meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    audit_ids = val_ids[:min(8, len(val_ids))]
    if len(audit_ids) == 0:
        raise RuntimeError("No validation scenarios available for hazard-capacity audit.")

    costs = intervention_costs(cfg, edges)
    B = baseline_budget(cfg, costs)
    cap = int(cfg["intervention"]["max_candidate_edges_per_step"])
    frac = float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])

    # Edge use in the fixed candidate-path set.  PE is binary incidence.
    pe = engine.PE.coalesce()
    pe_cols = pe.indices()[1].detach().cpu().numpy().astype(np.int64)
    path_use_count = np.bincount(pe_cols, minlength=engine.n_edges).astype(np.int64)

    bet = np.asarray(static.betweenness, float)
    K0 = np.asarray(static.K0, float)
    costs_np = np.asarray(costs, float)

    qlevels = [0.0, .10, .25, .50, .75, .90, .95, .99, 1.0]
    thresholds = [0.0, .001, .01, .05, .10, .25, .50]

    dist_rows = []
    threshold_rows = []
    candidate_rows = []
    time_rows = []

    def qsummary(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return {f"q{int(round(q*100)):02d}": float("nan") for q in qlevels}
        vals = np.quantile(x, qlevels)
        return {f"q{int(round(q*100)):02d}": float(v) for q, v in zip(qlevels, vals)}

    def add_distribution(sid, t, universe, variable, values):
        vals = np.asarray(values, float)
        finite = vals[np.isfinite(vals)]
        row = {
            "scenario_id": int(sid), "t": int(t), "universe": universe,
            "variable": variable, "n": int(len(finite)),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        }
        row.update(qsummary(finite))
        dist_rows.append(row)

    for sid in audit_ids:
        sid = int(sid)
        deg = np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        T = int(deg.shape[0])

        for t in range(T):
            d = np.asarray(deg[t], float)
            K = (1.0 - d) * K0
            abs_def = np.maximum(K0 - K, 0.0)
            rel_def = np.divide(abs_def, K0, out=np.zeros_like(abs_def), where=K0 > 1e-12)

            affected = abs_def > 1e-12
            feasible_all = np.where(affected & (costs_np <= B + 1e-12))[0]
            candidates = np.asarray(
                feasible_edges(K, K0, costs_np, B, cap, bet),
                dtype=np.int64
            )
            cand_mask = np.zeros(engine.n_edges, dtype=bool)
            cand_mask[candidates] = True

            time_rows.append({
                "scenario_id": sid, "t": t,
                "n_edges": int(engine.n_edges),
                "n_affected": int(affected.sum()),
                "affected_share": float(affected.mean()),
                "n_budget_feasible_affected": int(len(feasible_all)),
                "n_policy_candidates": int(len(candidates)),
                "candidate_cap": cap,
                "candidate_retention_share": float(len(candidates)/max(len(feasible_all),1)),
                "candidate_set_truncated": bool(len(feasible_all) > cap),
                "total_absolute_capacity_deficit": float(abs_def.sum()),
                "candidate_absolute_capacity_deficit": float(abs_def[candidates].sum()) if len(candidates) else 0.0,
                "candidate_deficit_share": float(
                    abs_def[candidates].sum()/max(abs_def[feasible_all].sum(), 1e-30)
                ) if len(feasible_all) else float("nan"),
            })

            universes = {
                "affected_all": np.where(affected)[0],
                "budget_feasible_affected": feasible_all,
                "policy_candidates": candidates,
            }
            variables = {
                "degradation": d,
                "relative_capacity_deficit": rel_def,
                "absolute_capacity_deficit": abs_def,
                "betweenness": bet,
                "intervention_cost": costs_np,
                "candidate_path_use_count": path_use_count.astype(float),
            }
            for uname, ids in universes.items():
                for vname, arr in variables.items():
                    add_distribution(sid, t, uname, vname, arr[ids])

            # Threshold diagnostics on physical degradation/deficit.
            for thr in thresholds:
                for uname, ids in universes.items():
                    vals = rel_def[ids]
                    threshold_rows.append({
                        "scenario_id": sid, "t": t, "universe": uname,
                        "relative_deficit_threshold": float(thr),
                        "n_above": int(np.sum(vals > thr)),
                        "share_above": float(np.mean(vals > thr)) if len(vals) else float("nan"),
                    })

            # Edge-level candidate diagnostics permit direct inspection of why
            # the prefilter chooses an edge.
            for rank, e in enumerate(candidates):
                e = int(e)
                candidate_rows.append({
                    "scenario_id": sid, "t": t,
                    "candidate_rank": int(rank + 1),
                    "edge_index": e,
                    "degradation": float(d[e]),
                    "relative_capacity_deficit": float(rel_def[e]),
                    "absolute_capacity_deficit": float(abs_def[e]),
                    "partial_restored_capacity": float(frac * abs_def[e]),
                    "reference_capacity": float(K0[e]),
                    "intervention_cost": float(costs_np[e]),
                    "budget_ratio": float(costs_np[e]/max(B,1e-30)),
                    "betweenness": float(bet[e]),
                    "candidate_path_use_count": int(path_use_count[e]),
                    "used_by_candidate_path": bool(path_use_count[e] > 0),
                })

        log.log(
            f"Hazard-capacity audit scenario={sid}: "
            f"T={T}; max affected={max(r['n_affected'] for r in time_rows if r['scenario_id']==sid):,}; "
            f"min candidate retention={min(r['candidate_retention_share'] for r in time_rows if r['scenario_id']==sid):.4%}; "
            f"candidate median relative deficit="
            f"{np.median([r['relative_capacity_deficit'] for r in candidate_rows if r['scenario_id']==sid]):.6g}"
        )

    dist_df = pd.DataFrame(dist_rows)
    thr_df = pd.DataFrame(threshold_rows)
    cand_df = pd.DataFrame(candidate_rows)
    time_df = pd.DataFrame(time_rows)

    dist_df.to_csv(paths.eval/"hazard_capacity_distribution_audit.csv", index=False)
    thr_df.to_csv(paths.eval/"hazard_capacity_threshold_audit.csv", index=False)
    cand_df.to_csv(paths.eval/"hazard_capacity_candidate_edges.csv", index=False)
    time_df.to_csv(paths.eval/"hazard_capacity_candidate_retention.csv", index=False)

    # Candidate-vs-feasible contrasts for variables that can reveal a defective
    # prefilter (near-zero deficits, path irrelevance, or centrality dominance).
    contrast_rows = []
    for (sid,t), g in dist_df.groupby(["scenario_id","t"]):
        for var in [
            "degradation","relative_capacity_deficit","absolute_capacity_deficit",
            "betweenness","intervention_cost","candidate_path_use_count"
        ]:
            sub = g[g.variable == var]
            def med(universe):
                z = sub[sub.universe == universe]
                return float(z["q50"].iloc[0]) if len(z) else float("nan")
            a = med("budget_feasible_affected")
            c = med("policy_candidates")
            contrast_rows.append({
                "scenario_id": int(sid), "t": int(t), "variable": var,
                "feasible_median": a,
                "candidate_median": c,
                "candidate_to_feasible_median_ratio": (
                    c/a if np.isfinite(a) and abs(a)>1e-30 else float("nan")
                ),
                "candidate_minus_feasible_median": c-a if np.isfinite(a) and np.isfinite(c) else float("nan"),
            })
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(paths.eval/"hazard_capacity_candidate_contrasts.csv", index=False)

    # Rank associations inside the actual candidate sets.
    assoc_rows = []
    for (sid,t), g in cand_df.groupby(["scenario_id","t"]):
        for xname in ["relative_capacity_deficit","absolute_capacity_deficit",
                      "betweenness","intervention_cost","candidate_path_use_count"]:
            x = g[xname].to_numpy(float)
            rank = g["candidate_rank"].to_numpy(float)
            if len(x)>1 and np.std(x)>1e-15:
                rho = float(pd.Series(x).rank().corr(pd.Series(-rank).rank()))
            else:
                rho = float("nan")
            assoc_rows.append({
                "scenario_id": int(sid), "t": int(t),
                "variable": xname,
                "spearman_with_higher_candidate_priority": rho,
            })
    assoc_df = pd.DataFrame(assoc_rows)
    assoc_df.to_csv(paths.eval/"hazard_capacity_candidate_rank_associations.csv", index=False)

    # Conservative summary: diagnostics only, no automatic recalibration.
    cand_rel = cand_df["relative_capacity_deficit"].to_numpy(float)
    cand_abs = cand_df["absolute_capacity_deficit"].to_numpy(float)
    path_share = float(cand_df["used_by_candidate_path"].mean()) if len(cand_df) else float("nan")
    summary = {
        "script_version": SCRIPT_VERSION,
        "model_or_data_modified": False,
        "policy_retrained": False,
        "audit_validation_scenarios": [int(x) for x in audit_ids],
        "n_scenario_time_states": int(len(time_df)),
        "candidate_cap": cap,
        "mean_candidate_retention_share": float(time_df["candidate_retention_share"].mean()),
        "median_candidate_relative_deficit": float(np.median(cand_rel)) if len(cand_rel) else float("nan"),
        "p90_candidate_relative_deficit": float(np.quantile(cand_rel,.90)) if len(cand_rel) else float("nan"),
        "median_candidate_absolute_deficit": float(np.median(cand_abs)) if len(cand_abs) else float("nan"),
        "share_candidates_used_by_candidate_paths": path_share,
        "share_candidates_relative_deficit_gt_0_01": float(np.mean(cand_rel>.01)) if len(cand_rel) else float("nan"),
        "share_candidates_relative_deficit_gt_0_05": float(np.mean(cand_rel>.05)) if len(cand_rel) else float("nan"),
        "share_candidates_relative_deficit_gt_0_10": float(np.mean(cand_rel>.10)) if len(cand_rel) else float("nan"),
        "decision": "diagnostic_only_no_threshold_or_recalibration_applied",
    }
    write_json(paths.eval/"hazard_capacity_audit_summary.json", summary)

    log.log(
        "Hazard-capacity-candidate audit DONE: "
        f"states={len(time_df)}; mean candidate retention={summary['mean_candidate_retention_share']:.4%}; "
        f"candidate median relative deficit={summary['median_candidate_relative_deficit']:.6g}; "
        f"P90={summary['p90_candidate_relative_deficit']:.6g}; "
        f"candidate path-use share={summary['share_candidates_used_by_candidate_paths']:.3%}. "
        "No physical threshold or recalibration applied."
    )
    return summary


# -----------------------------------------------------------------------------
# Candidate-action redesign audit
# -----------------------------------------------------------------------------

def candidate_action_redesign_audit(cfg, paths, log):
    """Read-only audit of alternative candidate-set constructions.

    The physical disruption process, routing model, intervention fraction,
    budget, and exact accessibility metric are held fixed. The audit compares
    deterministic screening rules only; it does not train a policy or select a
    final rule automatically.

    Rules:
      R0_current:
          current feasible_edges() prefilter.
      R1_path_relevant:
          degraded + budget-feasible + used by at least one candidate path,
          ranked by existing betweenness score.
      R2_path_deficit:
          same functional eligibility as R1, ranked by absolute capacity deficit.
      R3_path_deficit_use:
          same eligibility, ranked by absolute deficit * candidate-path use count.

    For each rule we retain at most the current candidate cap so differences in
    action-effect distributions are not mechanically caused by larger set size.
    """
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
    static.betweenness = load_or_compute_betweenness(cfg, paths, edges, log)

    meta = pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids = meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    audit_ids = val_ids[:min(8, len(val_ids))]
    if len(audit_ids) == 0:
        raise RuntimeError("No validation scenarios available for candidate redesign audit.")

    costs = np.asarray(intervention_costs(cfg, edges), float)
    B = baseline_budget(cfg, costs)
    cap = int(cfg["intervention"]["max_candidate_edges_per_step"])
    frac = float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])

    pe = engine.PE.coalesce()
    pe_cols = pe.indices()[1].detach().cpu().numpy().astype(np.int64)
    path_use = np.bincount(pe_cols, minlength=engine.n_edges).astype(np.int64)

    K0 = np.asarray(static.K0, float)
    bet = np.asarray(static.betweenness, float)

    def topk(ids, score, k):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return ids
        s = np.asarray(score, float)[ids]
        # Stable deterministic ordering: descending score, then ascending edge id.
        order = np.lexsort((ids, -np.nan_to_num(s, nan=-np.inf)))
        return ids[order[:min(k, len(ids))]]

    def build_rules(K):
        abs_def = np.maximum(K0-K, 0.0)
        degraded = abs_def > 1e-12
        budget_ok = costs <= B + 1e-12
        functional = path_use > 0
        all_feas = np.where(degraded & budget_ok)[0]
        func_feas = np.where(degraded & budget_ok & functional)[0]

        r0 = np.asarray(
            feasible_edges(K, K0, costs, B, cap, bet), dtype=np.int64
        )
        r1 = topk(func_feas, bet, cap)
        r2 = topk(func_feas, abs_def, cap)
        score3 = abs_def * np.maximum(path_use.astype(float), 1.0)
        r3 = topk(func_feas, score3, cap)

        return {
            "R0_current": r0,
            "R1_path_relevant": r1,
            "R2_path_deficit": r2,
            "R3_path_deficit_use": r3,
        }, all_feas, func_feas, abs_def

    detail_rows = []
    coverage_rows = []
    overlap_rows = []
    n_action_eval = min(8, cap)

    for sid in audit_ids:
        sid = int(sid)
        deg = np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        T = int(deg.shape[0])

        for t in range(T):
            Kexo = (1.0 - np.asarray(deg[t], float)) * K0
            rules, all_feas, func_feas, abs_def = build_rules(Kexo)

            # Exact no-action reference.
            _, q0, _ = engine.solve(Kexo)
            A0 = engine.accessibility_exact(q0, Kexo)
            _, L0, E0, _, _ = loss_components(
                A0, static.A0, static.population, static.vulnerable
            )

            # Coverage diagnostics for the full retained set under each rule.
            denom_def = float(abs_def[all_feas].sum()) if len(all_feas) else 0.0
            denom_path = float(path_use[func_feas].sum()) if len(func_feas) else 0.0

            for rule, ids in rules.items():
                ids = np.asarray(ids, dtype=np.int64)
                coverage_rows.append({
                    "scenario_id": sid,
                    "t": t,
                    "rule": rule,
                    "n_candidates": int(len(ids)),
                    "candidate_cap": cap,
                    "n_all_budget_feasible_degraded": int(len(all_feas)),
                    "n_functionally_relevant_feasible": int(len(func_feas)),
                    "candidate_retention_vs_all": float(len(ids)/max(len(all_feas),1)),
                    "candidate_retention_vs_functional": float(len(ids)/max(len(func_feas),1)),
                    "share_candidate_edges_used_by_paths": float(np.mean(path_use[ids] > 0)) if len(ids) else float("nan"),
                    "absolute_deficit_coverage": float(abs_def[ids].sum()/max(denom_def,1e-30)) if len(all_feas) else float("nan"),
                    "candidate_path_use_coverage": float(path_use[ids].sum()/max(denom_path,1e-30)) if len(func_feas) else float("nan"),
                    "median_relative_deficit": float(np.median(abs_def[ids]/np.maximum(K0[ids],1e-30))) if len(ids) else float("nan"),
                    "median_absolute_deficit": float(np.median(abs_def[ids])) if len(ids) else float("nan"),
                    "median_path_use_count": float(np.median(path_use[ids])) if len(ids) else float("nan"),
                })

            # Pairwise Jaccard overlap of the retained sets.
            rule_names = list(rules)
            for i in range(len(rule_names)):
                for j in range(i+1, len(rule_names)):
                    a = set(map(int, rules[rule_names[i]]))
                    b = set(map(int, rules[rule_names[j]]))
                    overlap_rows.append({
                        "scenario_id": sid, "t": t,
                        "rule_a": rule_names[i], "rule_b": rule_names[j],
                        "intersection": int(len(a & b)),
                        "union": int(len(a | b)),
                        "jaccard": float(len(a & b)/max(len(a | b),1)),
                    })

            # Exact action-effect audit on a deterministic subsample of each
            # retained set. Every rule gets the same maximum number of actions.
            for rule, ids in rules.items():
                ids = np.asarray(ids, dtype=np.int64)
                if len(ids) > n_action_eval:
                    pick = np.unique(np.linspace(0, len(ids)-1, n_action_eval, dtype=int))
                    eval_ids = ids[pick]
                else:
                    eval_ids = ids

                for rank_pos, e in enumerate(eval_ids):
                    e = int(e)
                    Kctl = Kexo.copy()
                    deficit = max(K0[e]-Kctl[e], 0.0)
                    Kctl[e] = min(Kctl[e] + frac*deficit, K0[e])

                    _, q, _ = engine.solve(Kctl)
                    A = engine.accessibility_exact(q, Kctl)
                    _, L, E, _, _ = loss_components(
                        A, static.A0, static.population, static.vulnerable
                    )

                    detail_rows.append({
                        "scenario_id": sid,
                        "t": t,
                        "rule": rule,
                        "eval_position": int(rank_pos+1),
                        "edge_index": e,
                        "degradation": float(abs_def[e]/max(K0[e],1e-30)),
                        "absolute_capacity_deficit": float(abs_def[e]),
                        "candidate_path_use_count": int(path_use[e]),
                        "betweenness": float(bet[e]),
                        "intervention_cost": float(costs[e]),
                        "Lacc_no_action": float(L0),
                        "Leq_no_action": float(E0),
                        "Lacc_exact": float(L),
                        "Leq_exact": float(E),
                        "gain_Lacc_exact": float(L0-L),
                        "gain_Leq_exact": float(E0-E),
                    })

        scen_cov = pd.DataFrame([r for r in coverage_rows if r["scenario_id"] == sid])
        log.log(
            f"Candidate-redesign audit scenario={sid}: T={T}; "
            + "; ".join(
                f"{rule}: path-share={g['share_candidate_edges_used_by_paths'].mean():.1%}, "
                f"deficit-coverage={g['absolute_deficit_coverage'].mean():.3%}"
                for rule, g in scen_cov.groupby("rule")
            )
        )

    detail = pd.DataFrame(detail_rows)
    coverage = pd.DataFrame(coverage_rows)
    overlap = pd.DataFrame(overlap_rows)

    detail.to_csv(paths.eval/"candidate_redesign_action_effects.csv", index=False)
    coverage.to_csv(paths.eval/"candidate_redesign_coverage.csv", index=False)
    overlap.to_csv(paths.eval/"candidate_redesign_rule_overlap.csv", index=False)

    # Aggregate exact effects without automatically choosing a rule.
    summary_rows = []
    for rule, g in detail.groupby("rule", sort=False):
        x = g["gain_Lacc_exact"].to_numpy(float)
        eq = g["gain_Leq_exact"].to_numpy(float)
        c = coverage[coverage.rule == rule]
        summary_rows.append({
            "rule": rule,
            "n_exact_action_evaluations": int(len(g)),
            "mean_gain_Lacc_exact": float(np.mean(x)),
            "median_gain_Lacc_exact": float(np.median(x)),
            "p90_gain_Lacc_exact": float(np.quantile(x,.90)),
            "max_gain_Lacc_exact": float(np.max(x)),
            "min_gain_Lacc_exact": float(np.min(x)),
            "share_positive_gain_Lacc": float(np.mean(x > 1e-12)),
            "mean_gain_Leq_exact": float(np.mean(eq)),
            "mean_path_relevant_share": float(c["share_candidate_edges_used_by_paths"].mean()),
            "mean_absolute_deficit_coverage": float(c["absolute_deficit_coverage"].mean()),
            "mean_candidate_path_use_coverage": float(c["candidate_path_use_coverage"].mean()),
            "median_candidate_relative_deficit": float(c["median_relative_deficit"].median()),
            "median_candidate_path_use_count": float(c["median_path_use_count"].median()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.eval/"candidate_redesign_summary.csv", index=False)

    # Paired state-level comparisons use each rule's best audited action.
    state_best = (
        detail.groupby(["scenario_id","t","rule"], as_index=False)
        .agg(
            best_gain_Lacc_exact=("gain_Lacc_exact","max"),
            median_gain_Lacc_exact=("gain_Lacc_exact","median"),
            positive_share=("gain_Lacc_exact", lambda s: float(np.mean(np.asarray(s,float)>1e-12))),
        )
    )
    state_best.to_csv(paths.eval/"candidate_redesign_state_best.csv", index=False)

    paired_rows = []
    wide = state_best.pivot(index=["scenario_id","t"], columns="rule",
                            values="best_gain_Lacc_exact")
    if "R0_current" in wide.columns:
        for alt in ["R1_path_relevant","R2_path_deficit","R3_path_deficit_use"]:
            if alt not in wide.columns:
                continue
            tmp = wide[["R0_current",alt]].dropna()
            for (sid,t), row in tmp.iterrows():
                b = float(row["R0_current"]); a = float(row[alt])
                paired_rows.append({
                    "scenario_id": int(sid), "t": int(t),
                    "alternative": alt,
                    "current_best_gain": b,
                    "alternative_best_gain": a,
                    "gain_difference": a-b,
                    "alternative_better": bool(a > b + 1e-12),
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(paths.eval/"candidate_redesign_paired_vs_current.csv", index=False)

    # Machine-readable audit summary.  No automatic rule selection.
    out = {
        "script_version": SCRIPT_VERSION,
        "model_or_data_modified": False,
        "policy_retrained": False,
        "candidate_cap_held_fixed": cap,
        "audit_validation_scenarios": [int(x) for x in audit_ids],
        "rules": summary_df.to_dict(orient="records"),
        "decision": "diagnostic_only_no_candidate_rule_selected",
        "interpretation_guard": (
            "A rule may be considered for the final model only if it has a clear "
            "functional interpretation and improves decision-relevant coverage/effects "
            "without using outcome information unavailable at decision time."
        ),
    }
    write_json(paths.eval/"candidate_redesign_audit.json", out)

    log.log(
        "Candidate-action redesign audit DONE: "
        + "; ".join(
            f"{r['rule']} max={r['max_gain_Lacc_exact']:.12g}, "
            f"median={r['median_gain_Lacc_exact']:.12g}, "
            f"path-share={r['mean_path_relevant_share']:.1%}"
            for r in summary_rows
        )
        + ". No candidate rule selected automatically."
    )
    return out


# -----------------------------------------------------------------------------
# Candidate-cap sensitivity and decision-signal audit
# -----------------------------------------------------------------------------

def candidate_cap_signal_audit(cfg, paths, log):
    """Read-only R3 cap-sensitivity and high-effect decision-signal audit.

    This stage does not retrain a policy and does not modify the physical model.
    It studies the pre-action R3 screening score
        absolute capacity deficit * candidate-path use count
    under K in {20, 40, 80, 160}. Coverage is evaluated over all validation
    scenario-times. Exact accessibility effects are evaluated on a deterministic
    union of selected candidates so the same edge is solved only once per state.

    No cap is selected automatically. The outputs are diagnostics intended to
    support a later, explicitly documented model choice.
    """
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
    static.betweenness = load_or_compute_betweenness(cfg, paths, edges, log)

    meta = pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids = meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    audit_ids = val_ids[:min(8, len(val_ids))]
    if len(audit_ids) == 0:
        raise RuntimeError("No validation scenarios available for cap-sensitivity audit.")

    caps = [20, 40, 80, 160]
    costs = np.asarray(intervention_costs(cfg, edges), float)
    B = baseline_budget(cfg, costs)
    frac = float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])

    pe = engine.PE.coalesce()
    pe_cols = pe.indices()[1].detach().cpu().numpy().astype(np.int64)
    path_use = np.bincount(pe_cols, minlength=engine.n_edges).astype(np.int64)

    K0 = np.asarray(static.K0, float)

    def stable_rank(ids, score):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return ids
        s = np.asarray(score, float)[ids]
        order = np.lexsort((ids, -np.nan_to_num(s, nan=-np.inf)))
        return ids[order]

    coverage_rows, selected_rows, effect_rows, stability_rows = [], [], [], []
    high_effect_threshold = 1e-8

    for sid in audit_ids:
        sid = int(sid)
        deg = np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        T = int(deg.shape[0])

        for t in range(T):
            Kexo = (1.0 - np.asarray(deg[t], float)) * K0
            abs_def = np.maximum(K0-Kexo, 0.0)
            rel_def = abs_def / np.maximum(K0, 1e-30)

            degraded = abs_def > 1e-12
            budget_ok = costs <= B + 1e-12
            path_relevant = path_use > 0
            eligible = np.where(degraded & budget_ok & path_relevant)[0]

            # R3 score uses only pre-action observable/model-state information.
            r3_score = abs_def * path_use.astype(float)
            ranked = stable_rank(eligible, r3_score)

            total_def = float(abs_def[eligible].sum()) if len(eligible) else 0.0
            total_use = float(path_use[eligible].sum()) if len(eligible) else 0.0

            sets = {}
            for Kcap in caps:
                ids = ranked[:min(Kcap, len(ranked))]
                sets[Kcap] = ids
                coverage_rows.append({
                    "scenario_id": sid, "t": t, "cap": Kcap,
                    "n_eligible_r3": int(len(eligible)),
                    "n_selected": int(len(ids)),
                    "retention_share": float(len(ids)/max(len(eligible),1)),
                    "absolute_deficit_coverage": float(abs_def[ids].sum()/max(total_def,1e-30)) if len(eligible) else float("nan"),
                    "candidate_path_use_coverage": float(path_use[ids].sum()/max(total_use,1e-30)) if len(eligible) else float("nan"),
                    "median_relative_deficit": float(np.median(rel_def[ids])) if len(ids) else float("nan"),
                    "median_absolute_deficit": float(np.median(abs_def[ids])) if len(ids) else float("nan"),
                    "median_path_use_count": float(np.median(path_use[ids])) if len(ids) else float("nan"),
                    "min_selected_r3_score": float(np.min(r3_score[ids])) if len(ids) else float("nan"),
                })
                for rank_pos, e in enumerate(ids, start=1):
                    selected_rows.append({
                        "scenario_id": sid, "t": t, "cap": Kcap,
                        "rank": rank_pos, "edge_index": int(e),
                        "r3_score": float(r3_score[e]),
                        "relative_deficit": float(rel_def[e]),
                        "absolute_deficit": float(abs_def[e]),
                        "path_use_count": int(path_use[e]),
                        "intervention_cost": float(costs[e]),
                    })

            # Nested-set stability. For a deterministic ranking, the expected
            # Jaccard is Ksmall/Klarge when enough candidates exist; exporting
            # it still verifies implementation and finite-eligibility effects.
            for ka, kb in zip(caps[:-1], caps[1:]):
                a, b = set(map(int, sets[ka])), set(map(int, sets[kb]))
                stability_rows.append({
                    "scenario_id": sid, "t": t, "cap_a": ka, "cap_b": kb,
                    "intersection": len(a & b), "union": len(a | b),
                    "jaccard": float(len(a & b)/max(len(a | b),1)),
                    "cap_a_contained_in_cap_b": bool(a.issubset(b)),
                })

            # Exact decision-signal audit. Evaluate the top 8 candidates from
            # each cap, then deduplicate. Because R3 sets are nested this is
            # intentionally cheap; it tests whether increasing K reveals a
            # materially different top decision signal without 160 solves/state.
            eval_union = set()
            for Kcap in caps:
                ids = sets[Kcap]
                if len(ids):
                    nprobe = min(8, len(ids))
                    probe_pos = np.unique(np.linspace(0, len(ids)-1, nprobe, dtype=int))
                    eval_union.update(map(int, ids[probe_pos]))

            _, q0, _ = engine.solve(Kexo)
            A0 = engine.accessibility_exact(q0, Kexo)
            _, L0, E0, _, _ = loss_components(
                A0, static.A0, static.population, static.vulnerable
            )

            local_effect = {}
            for e in sorted(eval_union):
                Kctl = Kexo.copy()
                deficit = max(K0[e]-Kctl[e], 0.0)
                Kctl[e] = min(Kctl[e] + frac*deficit, K0[e])
                _, q, _ = engine.solve(Kctl)
                A = engine.accessibility_exact(q, Kctl)
                _, L, E, _, _ = loss_components(
                    A, static.A0, static.population, static.vulnerable
                )
                gain = float(L0-L)
                local_effect[e] = gain
                effect_rows.append({
                    "scenario_id": sid, "t": t, "edge_index": e,
                    "r3_score": float(r3_score[e]),
                    "relative_deficit": float(rel_def[e]),
                    "absolute_deficit": float(abs_def[e]),
                    "path_use_count": int(path_use[e]),
                    "intervention_cost": float(costs[e]),
                    "Lacc_no_action": float(L0),
                    "Leq_no_action": float(E0),
                    "Lacc_exact": float(L),
                    "Leq_exact": float(E),
                    "gain_Lacc_exact": gain,
                    "gain_Leq_exact": float(E0-E),
                    "high_effect_gt_1e8": bool(gain > high_effect_threshold),
                })

        log.log(f"Cap-sensitivity audit scenario={sid}: T={T} complete")

    coverage = pd.DataFrame(coverage_rows)
    selected = pd.DataFrame(selected_rows)
    effects = pd.DataFrame(effect_rows)
    stability = pd.DataFrame(stability_rows)

    coverage.to_csv(paths.eval/"candidate_cap_sensitivity_coverage.csv", index=False)
    selected.to_csv(paths.eval/"candidate_cap_sensitivity_selected_edges.csv", index=False)
    effects.to_csv(paths.eval/"candidate_cap_sensitivity_exact_effects.csv", index=False)
    stability.to_csv(paths.eval/"candidate_cap_sensitivity_stability.csv", index=False)

    # Attach exact effects to every cap in which the probed edge is selected.
    selected_effects = selected.merge(
        effects[["scenario_id","t","edge_index","gain_Lacc_exact","gain_Leq_exact"]],
        on=["scenario_id","t","edge_index"], how="inner"
    )
    selected_effects.to_csv(
        paths.eval/"candidate_cap_sensitivity_selected_exact_effects.csv", index=False
    )

    cap_summary_rows = []
    for Kcap in caps:
        c = coverage[coverage.cap == Kcap]
        e = selected_effects[selected_effects.cap == Kcap]
        x = e["gain_Lacc_exact"].to_numpy(float)
        cap_summary_rows.append({
            "cap": Kcap,
            "n_states": int(len(c)),
            "mean_n_selected": float(c.n_selected.mean()),
            "mean_absolute_deficit_coverage": float(c.absolute_deficit_coverage.mean()),
            "median_absolute_deficit_coverage": float(c.absolute_deficit_coverage.median()),
            "mean_candidate_path_use_coverage": float(c.candidate_path_use_coverage.mean()),
            "median_candidate_path_use_coverage": float(c.candidate_path_use_coverage.median()),
            "n_exact_effect_observations": int(len(x)),
            "median_gain_Lacc_exact": float(np.median(x)) if len(x) else float("nan"),
            "p90_gain_Lacc_exact": float(np.quantile(x,.90)) if len(x) else float("nan"),
            "max_gain_Lacc_exact": float(np.max(x)) if len(x) else float("nan"),
            "share_positive_gain_Lacc": float(np.mean(x > 1e-12)) if len(x) else float("nan"),
            "share_high_effect_gt_1e8": float(np.mean(x > high_effect_threshold)) if len(x) else float("nan"),
        })
    cap_summary = pd.DataFrame(cap_summary_rows)
    cap_summary.to_csv(paths.eval/"candidate_cap_sensitivity_summary.csv", index=False)

    high = effects[effects.gain_Lacc_exact > high_effect_threshold].copy()
    if len(high):
        high["physically_degraded"] = high.relative_deficit > 1e-12
        high["functionally_path_used"] = high.path_use_count > 0
        high["substantial_relative_deficit_gt_1pct"] = high.relative_deficit > .01
    high.to_csv(paths.eval/"candidate_cap_high_effect_events.csv", index=False)

    # Incremental coverage: transparent saturation diagnostic, not an optimizer.
    incr_rows = []
    cs = cap_summary.set_index("cap")
    for a, b in zip(caps[:-1], caps[1:]):
        incr_rows.append({
            "cap_from": a, "cap_to": b,
            "delta_mean_absolute_deficit_coverage":
                float(cs.loc[b,"mean_absolute_deficit_coverage"] -
                      cs.loc[a,"mean_absolute_deficit_coverage"]),
            "delta_mean_path_use_coverage":
                float(cs.loc[b,"mean_candidate_path_use_coverage"] -
                      cs.loc[a,"mean_candidate_path_use_coverage"]),
        })
    incremental = pd.DataFrame(incr_rows)
    incremental.to_csv(paths.eval/"candidate_cap_sensitivity_incremental.csv", index=False)

    out = {
        "script_version": SCRIPT_VERSION,
        "model_or_data_modified": False,
        "policy_retrained": False,
        "screening_rule": "R3_absolute_capacity_deficit_times_candidate_path_use",
        "caps_audited": caps,
        "audit_validation_scenarios": [int(x) for x in audit_ids],
        "high_effect_threshold_gain_Lacc": high_effect_threshold,
        "cap_summary": cap_summary.to_dict(orient="records"),
        "incremental_coverage": incremental.to_dict(orient="records"),
        "n_high_effect_events": int(len(high)),
        "all_high_effect_events_physically_degraded":
            bool(high.physically_degraded.all()) if len(high) else None,
        "all_high_effect_events_functionally_path_used":
            bool(high.functionally_path_used.all()) if len(high) else None,
        "decision": "diagnostic_only_no_cap_selected",
        "selection_guard": (
            "A final cap should be justified by transparent coverage/stability "
            "saturation and computational feasibility, not by maximizing validation "
            "accessibility gains ex post."
        ),
    }
    write_json(paths.eval/"candidate_cap_signal_audit.json", out)

    log.log(
        "Candidate-cap signal audit DONE: "
        + "; ".join(
            f"K={int(r['cap'])}: deficit={r['mean_absolute_deficit_coverage']:.3%}, "
            f"path-use={r['mean_candidate_path_use_coverage']:.3%}, "
            f"P90gain={r['p90_gain_Lacc_exact']:.12g}"
            for r in cap_summary_rows
        )
        + f"; high-effect events>{high_effect_threshold:g}: {len(high)}. "
          "No cap selected automatically."
    )
    return out


# -----------------------------------------------------------------------------
# Extended candidate-cap saturation audit
# -----------------------------------------------------------------------------

def candidate_cap_extended_audit(cfg, paths, log):
    """Read-only saturation audit for the R3 candidate correspondence.

    Purpose
    -------
    Evaluate whether a fixed top-K candidate cap can be justified by transparent
    saturation of physically and functionally relevant coverage. This stage:
      * does NOT retrain a policy;
      * does NOT run exact accessibility counterfactuals;
      * does NOT modify the disruption, routing, budget, or intervention model;
      * uses only pre-action information in the R3 score.

    R3 score
    --------
        score_e,t = (K0_e - K_e,t)_+ * path_use_e

    Caps audited
    ------------
        K in {40, 80, 160, 320, 640}

    The audit also tracks the ranks of previously detected high-effect edges
    when the v1.0.28 high-effect event file is available. No cap is selected
    automatically.
    """
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)

    meta = pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    val_ids = meta.loc[meta.split=="validation","scenario_id"].astype(int).to_numpy()
    audit_ids = val_ids[:min(8, len(val_ids))]
    if len(audit_ids) == 0:
        raise RuntimeError("No validation scenarios available for extended cap audit.")

    caps = [40, 80, 160, 320, 640]
    costs = np.asarray(intervention_costs(cfg, edges), float)
    B = baseline_budget(cfg, costs)

    pe = engine.PE.coalesce()
    pe_cols = pe.indices()[1].detach().cpu().numpy().astype(np.int64)
    path_use = np.bincount(pe_cols, minlength=engine.n_edges).astype(np.int64)

    K0 = np.asarray(static.K0, float)

    def stable_rank(ids, score):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return ids
        s = np.asarray(score, float)[ids]
        order = np.lexsort((ids, -np.nan_to_num(s, nan=-np.inf)))
        return ids[order]

    # Optional: reuse the exact high-effect events discovered by v1.0.28.
    high_path = paths.eval/"candidate_cap_high_effect_events.csv"
    high_lookup = {}
    if high_path.exists():
        high_prev = pd.read_csv(high_path)
        required = {"scenario_id","t","edge_index","gain_Lacc_exact"}
        if required.issubset(high_prev.columns):
            for _, r in high_prev.iterrows():
                key = (int(r.scenario_id), int(r.t), int(r.edge_index))
                high_lookup[key] = float(r.gain_Lacc_exact)
            log.log(
                f"Extended cap audit: loaded {len(high_lookup)} previously "
                "identified high-effect state-edge events."
            )
        else:
            log.log(
                "Extended cap audit: previous high-effect file exists but lacks "
                "required columns; rank tracking skipped."
            )
    else:
        log.log(
            "Extended cap audit: previous high-effect file not found; "
            "rank tracking skipped."
        )

    state_rows = []
    selected_rows = []
    stability_rows = []
    high_rank_rows = []

    for sid in audit_ids:
        sid = int(sid)
        deg = np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        T = int(deg.shape[0])

        for t in range(T):
            Kexo = (1.0 - np.asarray(deg[t], float)) * K0
            abs_def = np.maximum(K0-Kexo, 0.0)
            rel_def = abs_def / np.maximum(K0, 1e-30)

            degraded = abs_def > 1e-12
            budget_ok = costs <= B + 1e-12
            path_relevant = path_use > 0
            eligible = np.where(degraded & budget_ok & path_relevant)[0]

            score = abs_def * path_use.astype(float)
            ranked = stable_rank(eligible, score)
            rank_map = {int(e): int(r+1) for r, e in enumerate(ranked)}

            total_def = float(abs_def[eligible].sum()) if len(eligible) else 0.0
            total_use = float(path_use[eligible].sum()) if len(eligible) else 0.0
            total_score = float(score[eligible].sum()) if len(eligible) else 0.0
            max_score = float(score[ranked[0]]) if len(ranked) else float("nan")

            sets = {}
            for Kcap in caps:
                ids = ranked[:min(Kcap, len(ranked))]
                sets[Kcap] = ids
                min_score = float(score[ids[-1]]) if len(ids) else float("nan")
                state_rows.append({
                    "scenario_id": sid,
                    "t": t,
                    "cap": Kcap,
                    "n_eligible_r3": int(len(eligible)),
                    "n_selected": int(len(ids)),
                    "retention_share": float(len(ids)/max(len(eligible),1)),
                    "absolute_deficit_coverage": (
                        float(abs_def[ids].sum()/max(total_def,1e-30))
                        if len(eligible) else float("nan")
                    ),
                    "candidate_path_use_coverage": (
                        float(path_use[ids].sum()/max(total_use,1e-30))
                        if len(eligible) else float("nan")
                    ),
                    "r3_score_coverage": (
                        float(score[ids].sum()/max(total_score,1e-30))
                        if len(eligible) else float("nan")
                    ),
                    "median_relative_deficit": (
                        float(np.median(rel_def[ids])) if len(ids) else float("nan")
                    ),
                    "median_absolute_deficit": (
                        float(np.median(abs_def[ids])) if len(ids) else float("nan")
                    ),
                    "median_path_use_count": (
                        float(np.median(path_use[ids])) if len(ids) else float("nan")
                    ),
                    "max_r3_score": max_score,
                    "min_selected_r3_score": min_score,
                    "min_to_max_score_ratio": (
                        float(min_score/max_score)
                        if len(ids) and np.isfinite(max_score) and max_score > 0
                        else float("nan")
                    ),
                })

                for rank_pos, e in enumerate(ids, start=1):
                    selected_rows.append({
                        "scenario_id": sid,
                        "t": t,
                        "cap": Kcap,
                        "rank": rank_pos,
                        "edge_index": int(e),
                        "r3_score": float(score[e]),
                        "relative_deficit": float(rel_def[e]),
                        "absolute_deficit": float(abs_def[e]),
                        "path_use_count": int(path_use[e]),
                    })

            # Deterministic nesting/stability checks.
            for ka, kb in zip(caps[:-1], caps[1:]):
                a, b = set(map(int, sets[ka])), set(map(int, sets[kb]))
                stability_rows.append({
                    "scenario_id": sid,
                    "t": t,
                    "cap_a": ka,
                    "cap_b": kb,
                    "intersection": int(len(a & b)),
                    "union": int(len(a | b)),
                    "jaccard": float(len(a & b)/max(len(a | b),1)),
                    "cap_a_contained_in_cap_b": bool(a.issubset(b)),
                })

            # Track ranks of previously validated high-effect edges.
            if high_lookup:
                for (hsid, ht, e), gain in high_lookup.items():
                    if hsid != sid or ht != t:
                        continue
                    e = int(e)
                    r = rank_map.get(e, None)
                    high_rank_rows.append({
                        "scenario_id": sid,
                        "t": t,
                        "edge_index": e,
                        "previous_gain_Lacc_exact": gain,
                        "eligible_under_r3": bool(e in rank_map),
                        "r3_rank": int(r) if r is not None else np.nan,
                        "r3_score": float(score[e]) if 0 <= e < len(score) else np.nan,
                        "relative_deficit": float(rel_def[e]) if 0 <= e < len(rel_def) else np.nan,
                        "absolute_deficit": float(abs_def[e]) if 0 <= e < len(abs_def) else np.nan,
                        "path_use_count": int(path_use[e]) if 0 <= e < len(path_use) else np.nan,
                        "included_K40": bool(r is not None and r <= 40),
                        "included_K80": bool(r is not None and r <= 80),
                        "included_K160": bool(r is not None and r <= 160),
                        "included_K320": bool(r is not None and r <= 320),
                        "included_K640": bool(r is not None and r <= 640),
                    })

        log.log(f"Extended cap audit scenario={sid}: T={T} complete")

    state = pd.DataFrame(state_rows)
    selected = pd.DataFrame(selected_rows)
    stability = pd.DataFrame(stability_rows)
    high_rank = pd.DataFrame(high_rank_rows)

    state.to_csv(paths.eval/"candidate_cap_extended_state_coverage.csv", index=False)
    selected.to_csv(paths.eval/"candidate_cap_extended_selected_edges.csv", index=False)
    stability.to_csv(paths.eval/"candidate_cap_extended_stability.csv", index=False)
    high_rank.to_csv(paths.eval/"candidate_cap_extended_high_effect_ranks.csv", index=False)

    summary_rows = []
    for Kcap in caps:
        g = state[state.cap == Kcap]
        summary_rows.append({
            "cap": Kcap,
            "n_states": int(len(g)),
            "mean_n_eligible_r3": float(g.n_eligible_r3.mean()),
            "median_n_eligible_r3": float(g.n_eligible_r3.median()),
            "mean_n_selected": float(g.n_selected.mean()),
            "mean_retention_share": float(g.retention_share.mean()),
            "mean_absolute_deficit_coverage": float(g.absolute_deficit_coverage.mean()),
            "median_absolute_deficit_coverage": float(g.absolute_deficit_coverage.median()),
            "mean_candidate_path_use_coverage": float(g.candidate_path_use_coverage.mean()),
            "median_candidate_path_use_coverage": float(g.candidate_path_use_coverage.median()),
            "mean_r3_score_coverage": float(g.r3_score_coverage.mean()),
            "median_r3_score_coverage": float(g.r3_score_coverage.median()),
            "median_min_to_max_score_ratio": float(g.min_to_max_score_ratio.median()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths.eval/"candidate_cap_extended_summary.csv", index=False)

    incremental_rows = []
    sm = summary.set_index("cap")
    for a, b in zip(caps[:-1], caps[1:]):
        add = b-a
        ddef = float(
            sm.loc[b,"mean_absolute_deficit_coverage"] -
            sm.loc[a,"mean_absolute_deficit_coverage"]
        )
        duse = float(
            sm.loc[b,"mean_candidate_path_use_coverage"] -
            sm.loc[a,"mean_candidate_path_use_coverage"]
        )
        dscore = float(
            sm.loc[b,"mean_r3_score_coverage"] -
            sm.loc[a,"mean_r3_score_coverage"]
        )
        incremental_rows.append({
            "cap_from": a,
            "cap_to": b,
            "additional_candidate_slots": add,
            "delta_mean_absolute_deficit_coverage": ddef,
            "delta_mean_candidate_path_use_coverage": duse,
            "delta_mean_r3_score_coverage": dscore,
            "deficit_coverage_gain_per_100_added_slots": float(ddef/add*100.0),
            "path_use_coverage_gain_per_100_added_slots": float(duse/add*100.0),
            "r3_score_coverage_gain_per_100_added_slots": float(dscore/add*100.0),
        })
    incremental = pd.DataFrame(incremental_rows)
    incremental.to_csv(paths.eval/"candidate_cap_extended_incremental.csv", index=False)

    # High-effect inclusion summary if the v1.0.28 events are available.
    high_summary = {}
    if len(high_rank):
        high_summary = {
            "n_previous_high_effect_events": int(len(high_rank)),
            "n_distinct_high_effect_edges": int(high_rank.edge_index.nunique()),
        }
        for Kcap in caps:
            col = f"included_K{Kcap}"
            high_summary[f"share_high_effect_events_in_K{Kcap}"] = (
                float(high_rank[col].mean()) if col in high_rank else None
            )
            high_summary[f"n_high_effect_events_in_K{Kcap}"] = (
                int(high_rank[col].sum()) if col in high_rank else None
            )

    out = {
        "script_version": SCRIPT_VERSION,
        "model_or_data_modified": False,
        "policy_retrained": False,
        "exact_accessibility_counterfactuals_run": False,
        "screening_rule": "R3_absolute_capacity_deficit_times_candidate_path_use",
        "caps_audited": caps,
        "audit_validation_scenarios": [int(x) for x in audit_ids],
        "cap_summary": summary.to_dict(orient="records"),
        "incremental_coverage": incremental.to_dict(orient="records"),
        "high_effect_rank_tracking": high_summary,
        "decision": "diagnostic_only_no_cap_selected",
        "selection_guard": (
            "A final fixed top-K cap should be retained only if additional slots "
            "show transparent diminishing returns in pre-action R3 coverage and "
            "do not systematically exclude previously validated high-effect "
            "actions. Otherwise the candidate correspondence should be redesigned "
            "rather than justified by computational convenience alone."
        ),
    }
    write_json(paths.eval/"candidate_cap_extended_audit.json", out)

    log.log(
        "Extended candidate-cap audit DONE: "
        + "; ".join(
            f"K={int(r['cap'])}: deficit={r['mean_absolute_deficit_coverage']:.3%}, "
            f"path-use={r['mean_candidate_path_use_coverage']:.3%}, "
            f"R3-score={r['mean_r3_score_coverage']:.3%}"
            for r in summary_rows
        )
        + ". No cap selected automatically."
    )
    return out


# -----------------------------------------------------------------------------
# Evaluation, robust statistics, exports
# -----------------------------------------------------------------------------

def load_graph_model(cfg,paths,engine,edges,zones,seed,ablation=None,spec=None):
    if (spec or ablation or "B5") in {"B5","B5_PPO"}:
        enforce_final_action_checkpoint_protocol(cfg,paths,allow_initialize=False)
    torch=require_torch();car_node=pd.read_csv(paths.processed/"training_car_node_feature.csv");cfg_local=json.loads(json.dumps(cfg));cfg_local["_paths"]={"processed":str(paths.processed)}
    st=build_graph_tensors(cfg_local,edges,zones,car_node,engine.device);Cls=make_graph_policy_class();m=Cls(cfg["graph_policy"]["hidden_dim"],cfg["graph_policy"]["message_layers"],ablation=="A1",ablation=="A2").to(engine.device)
    model_spec = spec or ablation or "B5"
    p=paths.models/f"{model_spec}_seed_{seed}.pt";m.load_state_dict(torch.load(p,map_location=engine.device));m.eval();return m,st


def deterministic_graph_policy(cfg,model,st,static,device,frozen=False):
    torch=require_torch()
    cache={}

    def p(s):
        current_state={
            "t":s["t"],"budget":s["budget"],"B":s["B"],
            "Kpre":s["Kpre"],"K0":static.K0,
            "degradation":s["degradation"],
        }
        if frozen:
            if "state" not in cache:
                cache["state"]={
                    "t":int(current_state["t"]),
                    "budget":float(current_state["budget"]),
                    "B":float(current_state["B"]),
                    "Kpre":np.asarray(
                        current_state["Kpre"],dtype=np.float32
                    ).copy(),
                    "K0":static.K0,
                    "degradation":np.asarray(
                        current_state["degradation"],dtype=np.float32
                    ).copy(),
                }
            policy_state=cache["state"]
        else:
            policy_state=current_state

        feas=list(map(int,s["feasible"]))
        with torch.no_grad():
            edge_logits,no=graph_policy_logits(
                model,st,policy_state,cfg,device,edge_indices=feas
            )
            cl=torch.cat([edge_logits,no.view(1)])
            j=int(torch.argmax(cl).item())
        cand=feas+[-1]
        return int(cand[j])

    return p



def ensure_ablation_models(cfg,paths,engine,static,edges,zones,log):
    torch=require_torch()
    for abl in ["A1","A2","A3","A4","A5"]:
        for seed in active_numerical_seeds(cfg):
            p=paths.models/f"{abl}_seed_{seed}.pt"
            if p.exists():continue
            model,st,h,v=train_one_graph_seed(cfg,paths,engine,static,edges,zones,int(seed),abl,log)
            torch.save(model.state_dict(),p)


def _ablation_protocol_signature(cfg):
    """Protocol signature for A1--A5 checkpoints.

    All five ablations genuinely require separate training:
      A1 changes the graph architecture (no message passing);
      A2 removes the CAR node feature;
      A3 changes the training objective by setting lambda_E=0;
      A4 changes the training objective by setting lambda_R=0;
      A5 freezes sequential score updating during training and validation.
    """
    return {
        "script_family":"v1.0.36_ablation_training",
        "candidate_action_protocol":final_action_protocol_signature(cfg),
        "active_seeds":active_numerical_seeds(cfg),
        "epochs":int(cfg["graph_policy"]["epochs"]),
        "episodes_per_epoch":int(cfg["graph_policy"]["episodes_per_epoch"]),
        "validation_every":int(cfg["graph_policy"]["validation_every"]),
        "early_stopping_patience":int(
            cfg["graph_policy"]["early_stopping_patience"]
        ),
        "validation_scenarios_training":int(
            cfg["graph_policy"]["validation_scenarios_training"]
        ),
        "training_accessibility":str(
            cfg["graph_policy"]["training_accessibility"]
        ),
        "reported_accessibility":str(
            cfg["graph_policy"]["final_evaluation_accessibility"]
        ),
        "ablations":{
            "A1":"no_message_passing",
            "A2":"no_training_CAR_node_feature",
            "A3":"lambda_equity_zero_during_training",
            "A4":"lambda_tail_zero_during_training",
            "A5":"frozen_sequential_policy_scores_during_training_and_validation",
        },
    }


def train_ablation_models(cfg,paths,log):
    """Explicit, resumable A1--A5 training under the final R3/Top-160 protocol.

    This stage never touches B5/B5_PPO checkpoints. Each completed ablation/seed
    checkpoint is immediately durable, so interruption loses at most one model.
    """
    enforce_final_action_checkpoint_protocol(
        cfg,paths,allow_initialize=False
    )
    sig=_ablation_protocol_signature(cfg)
    marker=paths.models/"ablation_protocol_v1_0_36.json"

    existing=list(paths.models.glob("A[1-5]_seed_*.pt"))
    if marker.exists():
        old=read_json(marker)
        if old!=sig:
            raise RuntimeError(
                "Existing A1--A5 checkpoints use a different ablation protocol. "
                "Archive/remove the A1--A5 checkpoints and "
                "ablation_protocol_v1_0_36.json before retraining."
            )
    elif existing:
        raise RuntimeError(
            "A1--A5 checkpoints exist without the v1.0.36 ablation protocol "
            "marker. Refusing silent reuse."
        )
    else:
        write_json(marker,sig)

    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(
        cfg,paths,edges,log
    )
    torch=require_torch()

    seeds=active_numerical_seeds(cfg)
    abls=["A1","A2","A3","A4","A5"]
    total=len(abls)*len(seeds)
    completed=0
    all_history=[]
    started=time.perf_counter()

    # Reuse any already completed models only when the protocol marker matches.
    for abl in abls:
        for seed in seeds:
            p=paths.models/f"{abl}_seed_{int(seed)}.pt"
            if p.exists():
                completed+=1

    log.log(
        f"Explicit ablation training START: {completed}/{total} checkpoints "
        "already complete under the same protocol"
    )

    for abl in abls:
        for seed in seeds:
            p=paths.models/f"{abl}_seed_{int(seed)}.pt"
            if p.exists():
                log.log(f"Reuse trained {abl} seed={seed}: {p.name}")
                continue

            t0=time.perf_counter()
            log.log(
                f"{abl} seed={seed} START "
                f"({completed+1}/{total} target checkpoint)"
            )
            model,st,hist,v=train_one_graph_seed(
                cfg,paths,engine,static,edges,zones,
                int(seed),abl,log
            )

            # Atomic checkpoint write: temporary file then replace.
            tmp=p.with_suffix(".pt.tmp")
            torch.save(model.state_dict(),tmp)
            tmp.replace(p)

            hist=hist.copy()
            hist["seed"]=int(seed)
            hist["spec"]=abl
            all_history.append(hist)

            # Per-model validation metadata for auditability.
            write_json(
                paths.eval/f"ablation_training_{abl}_seed_{int(seed)}.json",
                {
                    "script_version":SCRIPT_VERSION,
                    "spec":abl,
                    "seed":int(seed),
                    "checkpoint":p.name,
                    "best_validation_objective":float(v),
                    "elapsed_seconds":float(time.perf_counter()-t0),
                    "protocol":sig,
                }
            )

            completed+=1
            log.log(_progress_message(
                "Ablation checkpoints",
                completed,total,started,
                time.perf_counter()-t0
            ))

            del model,st
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_history:
        new=pd.concat(all_history,ignore_index=True)
        hp=paths.eval/"training_history_ablations.csv"
        if hp.exists():
            old=pd.read_csv(hp)
            new=pd.concat([old,new],ignore_index=True)
            new=new.drop_duplicates(
                subset=["spec","seed","epoch"],keep="last"
            )
        new.to_csv(hp,index=False)

    inv=_evaluation_model_inventory(cfg,paths)
    inv.to_csv(paths.eval/"evaluation_model_inventory.csv",index=False)
    missing=inv.loc[~inv.exists]
    write_json(paths.eval/"ablation_training_completion.json",{
        "script_version":SCRIPT_VERSION,
        "protocol":sig,
        "n_ablation_checkpoints_expected":total,
        "n_ablation_checkpoints_present":int(
            inv[inv.spec.isin(abls)].exists.sum()
        ),
        "n_total_learned_checkpoints_present":int(inv.exists.sum()),
        "n_total_learned_checkpoints_required":int(len(inv)),
        "n_missing_total":int(len(missing)),
        "missing":[
            {"spec":str(r.spec),"seed":int(r.seed)}
            for r in missing.itertuples()
        ],
    })
    if len(missing):
        raise RuntimeError(
            "Ablation stage completed but evaluation inventory is still "
            f"missing {len(missing)} learned checkpoint(s)."
        )
    log.log(
        f"Explicit ablation training COMPLETE: {total}/{total} A1--A5 "
        "checkpoints; evaluation inventory 70/70."
    )



def paired_bootstrap_diff(a,b,nboot,alpha,seed):
    """Returns mean(a-b) and percentile CI using paired resampling."""
    a=np.asarray(a,float);b=np.asarray(b,float);mask=np.isfinite(a)&np.isfinite(b);a=a[mask];b=b[mask];n=len(a)
    if n==0:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed);d=a-b;boots=np.empty(nboot,float)
    chunk=500
    for i in range(0,nboot,chunk):
        m=min(chunk,nboot-i);idx=rng.integers(0,n,size=(m,n));boots[i:i+m]=d[idx].mean(axis=1)
    lo=np.quantile(boots,(1-alpha)/2);hi=np.quantile(boots,1-(1-alpha)/2);return float(d.mean()),float(lo),float(hi)


def bootstrap_cvar_diff(a,b,nboot,alpha_cvar,conf,seed):
    a=np.asarray(a,float);b=np.asarray(b,float);mask=np.isfinite(a)&np.isfinite(b);a=a[mask];b=b[mask];n=len(a);rng=np.random.default_rng(seed);boots=np.empty(nboot)
    for i in range(nboot):
        idx=rng.integers(0,n,size=n);boots[i]=cvar_empirical(a[idx],alpha_cvar)-cvar_empirical(b[idx],alpha_cvar)
    est=cvar_empirical(a,alpha_cvar)-cvar_empirical(b,alpha_cvar);lo=np.quantile(boots,(1-conf)/2);hi=np.quantile(boots,1-(1-conf)/2);return float(est),float(lo),float(hi)


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p=np.asarray(pvals,float);n=len(p);order=np.argsort(p);adj=np.empty(n);running=0.0
    for rank,idx in enumerate(order):
        val=(n-rank)*p[idx];running=max(running,val);adj[idx]=min(running,1.0)
    return adj



def _scenario_mean_outcomes(res: pd.DataFrame, spec: str, split: str) -> pd.DataFrame:
    """Scenario-level outcomes, averaging numerical-policy seeds before inference."""
    g=res[(res.spec==spec)&(res.split==split)].groupby("scenario_id",as_index=True).agg(Z=("Z","mean"),Q=("Q","mean"))
    return g.sort_index()


def _criterion_from_sample(z: np.ndarray, q: np.ndarray, cfg: Mapping[str,Any]) -> float:
    z=np.asarray(z,float); q=np.asarray(q,float)
    return float(np.mean(z)+float(cfg["risk"]["lambda_equity"])*np.mean(q)+float(cfg["risk"]["lambda_tail"])*cvar_empirical(z,float(cfg["risk"]["cvar_alpha"])))


def paired_full_criterion_bootstrap(a: pd.DataFrame,b: pd.DataFrame,cfg: Mapping[str,Any],nboot: int,conf: float,seed: int):
    """Paired scenario bootstrap; recomputes CVaR and full J in every draw."""
    ids=a.index.intersection(b.index)
    aa=a.loc[ids,["Z","Q"]].to_numpy(float); bb=b.loc[ids,["Z","Q"]].to_numpy(float)
    mask=np.isfinite(aa).all(axis=1)&np.isfinite(bb).all(axis=1); aa=aa[mask];bb=bb[mask];n=len(aa)
    if n==0:return (np.nan,np.nan,np.nan,np.array([],float))
    est=_criterion_from_sample(aa[:,0],aa[:,1],cfg)-_criterion_from_sample(bb[:,0],bb[:,1],cfg)
    rng=np.random.default_rng(seed);boots=np.empty(nboot,float)
    for i in range(nboot):
        ii=rng.integers(0,n,size=n)
        boots[i]=_criterion_from_sample(aa[ii,0],aa[ii,1],cfg)-_criterion_from_sample(bb[ii,0],bb[ii,1],cfg)
    lo,hi=np.quantile(boots,[(1-conf)/2,1-(1-conf)/2])
    return float(est),float(lo),float(hi),boots


def paired_signflip_pvalue(d: np.ndarray,n_resamples: int,seed: int) -> float:
    """Two-sided paired randomization test for a mean difference."""
    d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d)
    if n==0:return np.nan
    obs=abs(float(np.mean(d)))
    # Exact enumeration when feasible; otherwise deterministic Monte Carlo.
    if n<=20:
        total=1<<n;extreme=0
        for mask in range(total):
            signs=np.fromiter((1.0 if (mask>>j)&1 else -1.0 for j in range(n)),dtype=float,count=n)
            extreme += abs(float(np.mean(signs*d))) >= obs-1e-15
        return float(extreme/total)
    rng=np.random.default_rng(seed);extreme=0;done=0;chunk=2000
    while done<n_resamples:
        m=min(chunk,n_resamples-done)
        signs=rng.integers(0,2,size=(m,n),dtype=np.int8)*2-1
        vals=np.abs((signs*d).mean(axis=1));extreme+=int(np.sum(vals>=obs-1e-15));done+=m
    return float((extreme+1)/(n_resamples+1))


def paired_wilcoxon_pvalue(d: np.ndarray) -> float:
    d=np.asarray(d,float);d=d[np.isfinite(d)]
    if len(d)==0 or np.all(np.abs(d)<=1e-15):return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d,zero_method="pratt",alternative="two-sided",method="auto").pvalue)
    except Exception:
        return np.nan


def rank_biserial_paired(d: np.ndarray) -> float:
    """Signed-rank matched-pairs rank-biserial effect; positive means first policy has larger loss."""
    d=np.asarray(d,float);d=d[np.isfinite(d)&(np.abs(d)>1e-15)]
    if len(d)==0:return 0.0
    from scipy.stats import rankdata
    r=rankdata(np.abs(d),method="average");den=float(r.sum())
    return float((r[d>0].sum()-r[d<0].sum())/den) if den>0 else 0.0


def _paired_policy_inference(cfg,paths,res,log):
    """Publication-facing paired inference on exact full-network test trajectories."""
    nboot=int(cfg["statistics"]["bootstrap_resamples"]);conf=float(cfg["statistics"]["confidence_level"])
    nperm=int(cfg["statistics"].get("signflip_resamples",100000));base_seed=int(active_numerical_seeds(cfg)[0])
    alpha=float(cfg["risk"]["cvar_alpha"]); rows=[]
    b5=_scenario_mean_outcomes(res,"B5","test")
    for pos,k in enumerate(["B0","B1","B2","B3","B4"]):
        bk=_scenario_mean_outcomes(res,k,"test");ids=b5.index.intersection(bk.index)
        a=bk.loc[ids];b=b5.loc[ids];dz=(a.Z-b.Z).to_numpy();dq=(a.Q-b.Q).to_numpy()
        bz=paired_bootstrap_diff(a.Z,b.Z,nboot,conf,base_seed+100+10*pos)
        bq=paired_bootstrap_diff(a.Q,b.Q,nboot,conf,base_seed+101+10*pos)
        bc=bootstrap_cvar_diff(a.Z,b.Z,nboot,alpha,conf,base_seed+102+10*pos)
        bj=paired_full_criterion_bootstrap(a,b,cfg,nboot,conf,base_seed+103+10*pos)
        rows.append({"comparison":f"B5_vs_{k}","reference":k,"candidate":"B5","primary_comparison":k=="B4","n_scenarios":len(ids),
            "delta_Z_reference_minus_B5":bz[0],"Z_ci_lo":bz[1],"Z_ci_hi":bz[2],"Z_signflip_p":paired_signflip_pvalue(dz,nperm,base_seed+104+10*pos),"Z_wilcoxon_p":paired_wilcoxon_pvalue(dz),"Z_rank_biserial":rank_biserial_paired(dz),
            "delta_Q_reference_minus_B5":bq[0],"Q_ci_lo":bq[1],"Q_ci_hi":bq[2],"Q_signflip_p":paired_signflip_pvalue(dq,nperm,base_seed+105+10*pos),"Q_wilcoxon_p":paired_wilcoxon_pvalue(dq),"Q_rank_biserial":rank_biserial_paired(dq),
            "delta_CVaR_reference_minus_B5":bc[0],"CVaR_ci_lo":bc[1],"CVaR_ci_hi":bc[2],
            "delta_J_reference_minus_B5":bj[0],"J_ci_lo":bj[1],"J_ci_hi":bj[2],"fraction_B5_lower_Z":float(np.mean(dz>0))})
    out=pd.DataFrame(rows)
    # Holm correction is family-wise across the five B5-vs-baseline comparisons, separately by outcome/test.
    if len(out):
        for col in ["Z_signflip_p","Z_wilcoxon_p","Q_signflip_p","Q_wilcoxon_p"]:
            out[col+"_holm"]=holm_adjust(out[col].to_numpy(float))
    out.to_csv(paths.eval/"final_paired_policy_inference.csv",index=False)
    # Structural holdout inference is reported separately and is not mixed into the primary multiplicity family.
    structural=[]
    for split in ["test","test_structural"]:
        a=_scenario_mean_outcomes(res,"B4",split);b=_scenario_mean_outcomes(res,"B5",split);ids=a.index.intersection(b.index)
        aa=a.loc[ids];bb=b.loc[ids];dz=(aa.Z-bb.Z).to_numpy();dq=(aa.Q-bb.Q).to_numpy()
        bz=paired_bootstrap_diff(aa.Z,bb.Z,nboot,conf,base_seed+300);bq=paired_bootstrap_diff(aa.Q,bb.Q,nboot,conf,base_seed+301);bc=bootstrap_cvar_diff(aa.Z,bb.Z,nboot,alpha,conf,base_seed+302);bj=paired_full_criterion_bootstrap(aa,bb,cfg,nboot,conf,base_seed+303)
        structural.append({"split":split,"comparison":"B4_minus_B5","n_scenarios":len(ids),"delta_Z":bz[0],"Z_ci_lo":bz[1],"Z_ci_hi":bz[2],"delta_Q":bq[0],"Q_ci_lo":bq[1],"Q_ci_hi":bq[2],"delta_CVaR":bc[0],"CVaR_ci_lo":bc[1],"CVaR_ci_hi":bc[2],"delta_J":bj[0],"J_ci_lo":bj[1],"J_ci_hi":bj[2],"Z_signflip_p":paired_signflip_pvalue(dz,nperm,base_seed+304),"Z_wilcoxon_p":paired_wilcoxon_pvalue(dz),"fraction_B5_lower_Z":float(np.mean(dz>0)) if len(dz) else np.nan})
    pd.DataFrame(structural).to_csv(paths.eval/"final_B4_B5_holdout_inference.csv",index=False)
    log.log("Final paired inference exported: B5 vs B0-B4 with full-J bootstrap, sign-flip/Wilcoxon/Holm, plus B4-B5 structural holdout")
    return out

def precompute_evaluation_benchmarks(cfg,paths,log):
    """Construct/cache B3 and B4 without evaluating test outcomes."""
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(cfg,paths,edges,log)
    t0=time.perf_counter()
    b3=compute_b3_scores(cfg,paths,engine,static,edges,log)
    b4=b4_open_loop_search(cfg,paths,engine,static,edges,log)
    finite=np.isfinite(b3)
    out={
        "script_version":SCRIPT_VERSION,
        "candidate_action_protocol":final_action_protocol_signature(cfg),
        "B3":{
            "n_scored_edges":int(finite.sum()),
            "construction_accessibility":"candidate_path_surrogate",
            "reported_evaluation_accessibility":"exact_full_network",
        },
        "B4":{
            "sequence":[int(x) for x in b4],
            "construction_accessibility":"candidate_path_surrogate",
            "reported_evaluation_accessibility":"exact_full_network",
            "batched_search":True,
        },
        "elapsed_seconds":float(time.perf_counter()-t0),
    }
    write_json(paths.eval/"evaluation_benchmark_precompute_summary.json",out)
    log.log(
        f"Evaluation benchmark precompute DONE: B3 edges={int(finite.sum())}, "
        f"B4 sequence length={len(b4)}, elapsed={out['elapsed_seconds']/60:.1f}min"
    )
    return out


class ExactEvaluationStateCache:
    """Persistent cache of exact decision-relevant loss components.

    A cache key is scenario-state specific and action-prefix specific. Under the
    fixed intervention dynamics, the same scenario and action prefix imply the
    same controlled capacity vector and therefore the same exact full-network
    accessibility outcome. Policy decisions themselves are never cached.
    """
    def __init__(self, db_path, protocol_tag):
        self.db_path=Path(db_path)
        self.protocol_tag=str(protocol_tag)
        self.conn=sqlite3.connect(str(self.db_path),timeout=120.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exact_state_loss (
                protocol TEXT NOT NULL,
                scenario_token TEXT NOT NULL,
                t INTEGER NOT NULL,
                action_prefix TEXT NOT NULL,
                Lacc REAL NOT NULL,
                Leq REAL NOT NULL,
                Lv REAL NOT NULL,
                Lcomp REAL NOT NULL,
                PRIMARY KEY(protocol,scenario_token,t,action_prefix)
            )
            """
        )
        self.conn.commit()
        self.hits=0
        self.misses=0
        self.compute_seconds=0.0

    @staticmethod
    def prefix_key(actions):
        return ",".join(map(str,map(int,actions)))

    def get(self,scenario_token,t,actions):
        key=self.prefix_key(actions)
        row=self.conn.execute(
            """
            SELECT Lacc,Leq,Lv,Lcomp
            FROM exact_state_loss
            WHERE protocol=? AND scenario_token=? AND t=? AND action_prefix=?
            """,
            (self.protocol_tag,str(scenario_token),int(t),key)
        ).fetchone()
        if row is None:
            self.misses+=1
            return None
        self.hits+=1
        return tuple(map(float,row))

    def put(self,scenario_token,t,actions,Lacc,Leq,Lv,Lcomp):
        key=self.prefix_key(actions)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO exact_state_loss
            (protocol,scenario_token,t,action_prefix,Lacc,Leq,Lv,Lcomp)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                self.protocol_tag,str(scenario_token),int(t),key,
                float(Lacc),float(Leq),float(Lv),float(Lcomp)
            )
        )
        self.conn.commit()

    def close(self):
        with contextlib.suppress(Exception):
            self.conn.commit()
        with contextlib.suppress(Exception):
            self.conn.close()

    @property
    def requests(self):
        return int(self.hits+self.misses)

    @property
    def hit_rate(self):
        return float(self.hits/max(self.requests,1))


def _exact_cache_protocol(cfg,engine,static):
    """Cryptographic signature of every model-independent object entering
    exact reported accessibility/losses.

    v1.0.37 deliberately hashes topology reductions, zone/service locations
    and weights, population/vulnerability arrays, reference quantities, and
    the final action protocol. This prevents accidental reuse of an exact-state
    cache after a data, topology, service, weighting, or protocol change even
    when dimensions happen to remain unchanged.
    """
    h=hashlib.sha256()

    def upd_array(name,x,dtype=None):
        a=np.asarray(x if dtype is None else np.asarray(x,dtype=dtype))
        h.update(str(name).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(np.asarray(a.shape,dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(a).tobytes())

    h.update(
        json.dumps(
            {
                "cache_schema":"v1.0.41_exact_state_loss",
                "candidate_action_protocol":
                    final_action_protocol_signature(cfg),
                "accessibility":{
                    "kappa":float(engine.kappa),
                    "gamma":float(static.gamma),
                    "impedance":str(cfg["accessibility"]["impedance"]),
                },
                "routing":{
                    "iterations":int(cfg["network"]["routing_iterations"]),
                    "damping":float(cfg["network"]["routing_damping"]),
                    "temperature":float(
                        cfg["network"]["route_choice_temperature"]
                    ),
                },
                "intervention":{
                    "retention":float(
                        cfg["intervention"]["restoration_retention"]
                    ),
                    "fraction":float(
                        cfg["intervention"][
                            "restoration_fraction_of_remaining_deficit"
                        ]
                    ),
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    )

    upd_array("K0",static.K0,np.float64)
    upd_array("q0",static.q0,np.float64)
    upd_array("A0",static.A0,np.float64)
    upd_array("population",static.population,np.float64)
    upd_array("vulnerable",static.vulnerable,np.uint8)

    # Exact full-network accessibility graph after parallel-edge reduction.
    upd_array("pair_u",engine._pair_u,np.int64)
    upd_array("pair_v",engine._pair_v,np.int64)
    upd_array("edge_pair",engine._edge_pair,np.int64)
    upd_array("zone_node_i",engine._zone_node_i,np.int64)
    upd_array("service_node_i",engine._service_node_i,np.int64)
    upd_array("service_O",engine._service_O,np.float64)

    # Routing reference demand/candidate-path objects also determine q(K).
    upd_array(
        "routing_q0",
        engine.q0.detach().cpu().numpy(),
        np.float64
    )
    upd_array(
        "routing_f0",
        engine.f0.detach().cpu().numpy(),
        np.float64
    )
    upd_array(
        "path_od",
        engine.path_od.detach().cpu().numpy(),
        np.int64
    )
    upd_array(
        "od_demand",
        engine.od_demand.detach().cpu().numpy(),
        np.float64
    )

    return h.hexdigest()


def _scenario_token(scenario_id,degradation):
    h=hashlib.sha256(
        np.asarray(degradation,dtype=np.float32).tobytes()
    ).hexdigest()[:20]
    return f"{int(scenario_id)}:{h}"


def trajectory_rollout_cache_only(
    cfg,engine,static,edges,degradation,policy_fn,seed,
    scenario_id,cache,return_steps=False
):
    """Final evaluation assembly from the completed exact-state cache only.

    Policy decisions and intervention dynamics are reconstructed exactly.
    Reported losses are read from the exact full-network cache. A cache miss is
    a hard protocol failure: this function never calls engine.solve,
    accessibility_exact, or any fallback exact evaluator.
    """
    rng=np.random.default_rng(seed)
    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs); budget=float(B)
    restore=np.zeros_like(static.K0,float)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    cap=int(cfg["intervention"]["max_candidate_edges_per_step"])
    Z=Q=0.0; step_rows=[]; actions=[]
    stok=_scenario_token(scenario_id,degradation)

    for t in range(degradation.shape[0]):
        if t>0: restore*=retention
        Kexo=(1-degradation[t])*static.K0
        deficit=np.maximum(static.K0-(Kexo+restore),0)
        Kpre=np.minimum(Kexo+restore,static.K0)
        feas=policy_candidate_edges(cfg,engine,Kpre,static.K0,costs,budget,cap)
        # Same policy-visible state as the production precompute, without
        # multi-megabyte copies that are irrelevant to all final policies.
        state={"t":t,"budget":budget,"B":B,"Kpre":Kpre,
               "feasible":feas,"degradation":degradation[t],"rng":rng}
        action=int(policy_fn(state)) if len(feas) else -1
        if action>=0:
            if action not in set(map(int,feas)):
                raise RuntimeError(
                    f"Cache-only evaluation selected infeasible edge {action} "
                    f"at scenario={scenario_id}, t={t}"
                )
            restore[action]+=frac*deficit[action]
            budget-=costs[action]
        actions.append(action)

        cached=cache.get(stok,t,actions)
        if cached is None:
            raise RuntimeError(
                "CACHE MISS in strict final evaluation: "
                f"scenario={scenario_id}, t={t}, "
                f"action_prefix={ExactEvaluationStateCache.prefix_key(actions)}, "
                f"protocol={cache.protocol_tag}. "
                "No exact recomputation was performed. "
                "Run the v1.1.0-B reproduce/resume workflow so the frozen-B2 "
                "cache-coverage repair can complete before strict evaluation."
            )
        Lacc,Leq,Lv,Lc=cached
        Z+=(static.gamma**t)*Lacc
        Q+=(static.gamma**t)*Leq
        step_rows.append({"t":t,"action":action,"budget":budget,
                          "Lacc":Lacc,"Leq":Leq,"Lv":Lv,"Lcomp":Lc,
                          "n_feasible":len(feas)})
    return (Z,Q,step_rows) if return_steps else (Z,Q)


def trajectory_rollout_exact_cached(
    cfg,engine,static,edges,degradation,policy_fn,seed,
    scenario_id,cache,return_steps=False
):
    """Exact trajectory rollout with loss-state memoization only.

    This function is scientifically identical to ``trajectory_rollout`` for
    reported outcomes. On a cache miss it calls the same ``engine.solve`` and
    ``engine.accessibility_exact`` functions. On a cache hit it reuses only the
    resulting scalar loss components for an identical scenario/action prefix.
    """
    rng=np.random.default_rng(seed)
    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    budget=B
    restore=np.zeros_like(static.K0,float)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(
        cfg["intervention"]["restoration_fraction_of_remaining_deficit"]
    )
    T=degradation.shape[0]-1
    Z=Q=0.0
    step_rows=[]
    actions=[]
    stok=_scenario_token(scenario_id,degradation)

    for t in range(T+1):
        if t>0:
            restore*=retention
        Kexo=(1-degradation[t])*static.K0
        deficit=np.maximum(static.K0-(Kexo+restore),0)
        Kpre=np.minimum(Kexo+restore,static.K0)
        feas=policy_candidate_edges(
            cfg,engine,Kpre,static.K0,costs,budget,
            int(cfg["intervention"]["max_candidate_edges_per_step"])
        )
        state={
            "t":t,"budget":budget,"B":B,
            "Kpre":Kpre.copy(),"Kexo":Kexo.copy(),
            "restore":restore.copy(),"feasible":feas.copy(),
            "costs":costs,
            "degradation":degradation[t].copy(),
            "rng":rng
        }
        action=int(policy_fn(state)) if len(feas) else -1
        if action>=0:
            if action not in set(map(int,feas)):
                raise RuntimeError(
                    f"Policy selected infeasible edge {action} at t={t}"
                )
            inc=frac*deficit[action]
            restore[action]+=inc
            budget-=costs[action]

        actions.append(action)
        cached=cache.get(stok,t,actions)
        if cached is None:
            Kctl=np.minimum(Kexo+restore,static.K0)
            c0=time.perf_counter()
            _,q,_=engine.solve(Kctl)
            A=engine.accessibility_exact(q,Kctl)
            _,Lacc,Leq,Lv,Lc=loss_components(
                A,static.A0,static.population,static.vulnerable
            )
            cache.compute_seconds+=time.perf_counter()-c0
            cache.put(stok,t,actions,Lacc,Leq,Lv,Lc)
        else:
            Lacc,Leq,Lv,Lc=cached

        Z+=(static.gamma**t)*Lacc
        Q+=(static.gamma**t)*Leq
        step_rows.append({
            "t":t,"action":action,"budget":budget,
            "Lacc":Lacc,"Leq":Leq,"Lv":Lv,"Lcomp":Lc,
            "n_feasible":len(feas)
        })

    return (Z,Q,step_rows) if return_steps else (Z,Q)


def trajectory_action_prefix_only(
    cfg,engine,static,edges,degradation,policy_fn,seed
):
    """Generate the exact intervention action prefix without reported outcomes.

    This reproduces the intervention state transition and final R3/Top-K
    feasibility correspondence used by ``trajectory_rollout`` but deliberately
    skips ``engine.solve`` and ``accessibility_exact``. It is therefore useful
    for counting how many distinct exact network states the final evaluation
    will actually require.
    """
    rng=np.random.default_rng(seed)
    costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs)
    budget=B
    restore=np.zeros_like(static.K0,float)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(
        cfg["intervention"]["restoration_fraction_of_remaining_deficit"]
    )
    actions=[]
    T=degradation.shape[0]-1

    for t in range(T+1):
        if t>0:
            restore*=retention
        Kexo=(1-degradation[t])*static.K0
        deficit=np.maximum(static.K0-(Kexo+restore),0)
        Kpre=np.minimum(Kexo+restore,static.K0)
        feas=policy_candidate_edges(
            cfg,engine,Kpre,static.K0,costs,budget,
            int(cfg["intervention"]["max_candidate_edges_per_step"])
        )
        state={
            "t":t,"budget":budget,"B":B,
            "Kpre":Kpre.copy(),"Kexo":Kexo.copy(),
            "restore":restore.copy(),"feasible":feas.copy(),
            "costs":costs,
            "degradation":degradation[t].copy(),
            "rng":rng,
        }
        action=int(policy_fn(state)) if len(feas) else -1
        if action>=0:
            if action not in set(map(int,feas)):
                raise RuntimeError(
                    f"Prefix audit: infeasible action {action} at t={t}"
                )
            restore[action]+=frac*deficit[action]
            budget-=costs[action]
        actions.append(action)

    return actions


def evaluation_prefix_audit(cfg,paths,log):
    """Count distinct exact states before running expensive exact evaluation.

    The audit covers B0, B1, B3, B4, B5, A1--A5, and B5_PPO over all 180
    held-out scenarios and all prescribed seeds. B2 is intentionally excluded
    because its myopic candidate-path counterfactual construction itself
    requires many routing solves; its maximum 6*180 states are added separately
    to the conservative workload bound.

    No exact Dijkstra accessibility is computed and no model is modified.
    """
    torch=require_torch()
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(
        cfg,paths,edges,log
    )

    inv=_evaluation_model_inventory(cfg,paths)
    missing=inv.loc[~inv.exists]
    if len(missing):
        raise RuntimeError(
            f"Prefix audit requires complete learned inventory; "
            f"{len(missing)} checkpoint(s) are missing."
        )

    b3scores=compute_b3_scores(cfg,paths,engine,static,edges,log)
    b4seq=b4_open_loop_search(cfg,paths,engine,static,edges,log)

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].copy()
    test=test.sort_values(["split","scenario_id"]).reset_index(drop=True)
    seeds=active_numerical_seeds(cfg)

    rows=[]
    prefix_keys=set()
    by_scenario={}
    t0=time.perf_counter()

    def record(spec,seed,sid,split,actions):
        stok=f"{int(sid)}"
        for t in range(len(actions)):
            key=(stok,int(t),tuple(map(int,actions[:t+1])))
            prefix_keys.add(key)
            by_scenario.setdefault(int(sid),set()).add(
                (int(t),tuple(map(int,actions[:t+1])))
            )
        rows.append({
            "spec":spec,
            "seed":int(seed),
            "scenario_id":int(sid),
            "split":str(split),
            "action_sequence":"|".join(map(str,map(int,actions))),
            "n_actions":int(sum(int(a)>=0 for a in actions)),
        })

    # B0: stochastic benchmark, all prescribed seeds.
    for seed in seeds:
        pol=b0_policy_factory(int(seed))
        for r in test.itertuples():
            sid=int(r.scenario_id)
            d=np.load(
                paths.scenarios/f"scenario_{sid:05d}.npz"
            )["degradation"]
            # Keep one policy object across scenarios exactly as evaluate_all.
            a=trajectory_action_prefix_only(
                cfg,engine,static,edges,d,pol,int(seed)+sid
            )
            record("B0",seed,sid,r.split,a)

    # B1/B3/B4 deterministic fixed policies. B2 excluded deliberately.
    fixed={
        "B1":b1_policy_factory(static.betweenness),
        "B3":b3_policy_factory(b3scores),
        "B4":b4_policy_factory(b4seq),
    }
    fixed_seed=int(seeds[0])
    for spec,pol in fixed.items():
        for r in test.itertuples():
            sid=int(r.scenario_id)
            d=np.load(
                paths.scenarios/f"scenario_{sid:05d}.npz"
            )["degradation"]
            a=trajectory_action_prefix_only(
                cfg,engine,static,edges,d,pol,fixed_seed+sid
            )
            record(spec,fixed_seed,sid,r.split,a)

    # Learned deterministic policies.
    learned_specs=["B5","A1","A2","A3","A4","A5"]
    if bool(cfg.get("graph_ppo",{}).get("enabled",False)):
        learned_specs.append("B5_PPO")

    for spec in learned_specs:
        for seed in seeds:
            if spec=="B5_PPO":
                model,st=load_graph_model(
                    cfg,paths,engine,edges,zones,int(seed),
                    spec="B5_PPO"
                )
            else:
                model,st=load_graph_model(
                    cfg,paths,engine,edges,zones,int(seed),
                    None if spec=="B5" else spec
                )
            for r in test.itertuples():
                sid=int(r.scenario_id)
                d=np.load(
                    paths.scenarios/f"scenario_{sid:05d}.npz"
                )["degradation"]
                pol=deterministic_graph_policy(
                    cfg,model,st,static,engine.device,
                    frozen=(spec=="A5")
                )
                a=trajectory_action_prefix_only(
                    cfg,engine,static,edges,d,pol,int(seed)+sid
                )
                record(spec,seed,sid,r.split,a)
            del model,st
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.log(
                f"Prefix audit {spec} seed={seed}: "
                f"rows={len(rows)}, unique exact prefixes={len(prefix_keys)}"
            )

    df=pd.DataFrame(rows)
    df.to_csv(
        paths.eval/"evaluation_action_prefix_audit.csv",index=False
    )

    n_total_states=int(sum(len(x.split("|")) for x in df.action_sequence))
    n_unique=int(len(prefix_keys))
    implied_hits=n_total_states-n_unique
    hit_rate=implied_hits/max(n_total_states,1)

    # B2 has one deterministic seed and at most 6*180 distinct states.
    b2_upper=int(
        (int(cfg["disruptions"]["horizon_T"])+1)*len(test)
    )
    total_with_b2_upper=n_total_states+b2_upper
    unique_with_b2_upper=n_unique+b2_upper
    conservative_hit_rate=1.0-(
        unique_with_b2_upper/max(total_with_b2_upper,1)
    )

    # Runtime scale from the measured v1.0.35 B4 preflight when available.
    pre=paths.eval/"evaluation_preflight_summary.json"
    seconds_per_exact_state=None
    if pre.exists():
        pobj=read_json(pre)
        probe=pobj.get("single_exact_B4_probe",{})
        sec=float(probe.get("seconds",0.0))
        nst=int(probe.get("n_steps",0))
        if sec>0 and nst>0:
            seconds_per_exact_state=sec/nst

    conservative_hours=None
    if seconds_per_exact_state is not None:
        conservative_hours=(
            unique_with_b2_upper*seconds_per_exact_state/3600.0
        )

    per_spec=(
        df.groupby("spec")
        .agg(
            n_policy_trajectories=("scenario_id","size"),
            unique_action_sequences=("action_sequence","nunique"),
            mean_interventions=("n_actions","mean"),
        )
        .reset_index()
    )
    per_spec.to_csv(
        paths.eval/"evaluation_prefix_summary_by_spec.csv",index=False
    )

    out={
        "script_version":SCRIPT_VERSION,
        "evaluation_accessibility":"exact_full_network",
        "n_test_scenarios":int(len(test)),
        "covered_specs":sorted(df.spec.unique().tolist()),
        "B2_prefix_audit_excluded":True,
        "B2_reason":(
            "B2 action construction requires candidate-path counterfactual "
            "routing; its complete six-state-per-scenario workload is added "
            "as a conservative no-reuse upper bound."
        ),
        "covered_policy_trajectories":int(len(df)),
        "covered_exact_state_requests":n_total_states,
        "covered_unique_exact_state_prefixes":n_unique,
        "covered_implied_cache_hits":int(implied_hits),
        "covered_implied_hit_rate":float(hit_rate),
        "B2_unique_state_upper_bound":b2_upper,
        "conservative_total_state_requests_with_B2":
            int(total_with_b2_upper),
        "conservative_unique_states_with_B2":
            int(unique_with_b2_upper),
        "conservative_implied_hit_rate_with_B2":
            float(conservative_hit_rate),
        "seconds_per_exact_state_from_preflight":
            seconds_per_exact_state,
        "conservative_runtime_hours_from_preflight":
            conservative_hours,
        "elapsed_seconds":float(time.perf_counter()-t0),
        "cache_protocol":_exact_cache_protocol(cfg,engine,static),
        "scientific_note":(
            "Only identical scenario/action prefixes are treated as reusable. "
            "No reported accessibility or loss is approximated."
        ),
    }
    write_json(
        paths.eval/"evaluation_prefix_audit_summary.json",out
    )
    log.log(
        "Evaluation prefix audit COMPLETE: "
        f"covered requests={n_total_states}, unique={n_unique}, "
        f"implied hit rate={hit_rate:.1%}; "
        f"conservative with B2={conservative_hit_rate:.1%}"
        + (
            f"; projected exact runtime~{conservative_hours:.1f}h"
            if conservative_hours is not None else ""
        )
    )
    return out



def action_ablation_audit(cfg,paths,log):
    """Reproducible decision-level audit before expensive exact evaluation."""
    p=paths.eval/"evaluation_action_prefix_audit.csv"
    if not p.exists():
        raise RuntimeError(
            "Run --stage evaluation-prefix-audit first."
        )
    df=pd.read_csv(p)
    base=df[df.spec=="B5"][
        ["seed","scenario_id","split","action_sequence","n_actions"]
    ].rename(columns={
        "action_sequence":"B5_action_sequence",
        "n_actions":"B5_n_actions"
    })

    rows=[]
    for spec in ["A1","A2","A3","A4","A5","B5_PPO"]:
        x=df[df.spec==spec][
            ["seed","scenario_id","split","action_sequence","n_actions"]
        ]
        z=base.merge(
            x,on=["seed","scenario_id","split"],how="inner",
            validate="one_to_one"
        )
        same=z.B5_action_sequence.eq(z.action_sequence)
        rows.append({
            "spec":spec,
            "n_pairs":int(len(z)),
            "exact_action_sequence_agreement":float(same.mean()),
            "n_identical":int(same.sum()),
            "mean_B5_interventions":float(z.B5_n_actions.mean()),
            "mean_spec_interventions":float(z.n_actions.mean()),
        })
    out=pd.DataFrame(rows)
    out.to_csv(paths.eval/"action_ablation_audit.csv",index=False)
    write_json(paths.eval/"action_ablation_audit_summary.json",{
        "script_version":SCRIPT_VERSION,
        "basis":"held-out action sequences; no exact outcome approximation",
        "comparisons":out.to_dict(orient="records"),
        "interpretation_guard":(
            "Decision agreement is descriptive. It does not establish equality "
            "of learned parameters or causal irrelevance of an ablated input."
        ),
    })
    log.log(
        "Action ablation audit COMPLETE: "
        + "; ".join(
            f"{r.spec}={r.exact_action_sequence_agreement:.1%}"
            for r in out.itertuples()
        )
    )
    return out



def evaluation_cache_audit(cfg,paths,log):
    """Audit exact-cache equivalence and measure cross-policy state reuse.

    Uses only already-trained B5 models on a very small held-out subset.
    It does not train models and does not write publication-facing results.
    """
    torch=require_torch()
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(
        cfg,paths,edges,log
    )
    seeds=active_numerical_seeds(cfg)

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta.loc[meta.split=="test"].sort_values(
        "scenario_id"
    ).head(3)
    if len(test)==0:
        raise RuntimeError("No test scenarios available for cache audit.")

    audit_db=paths.eval/"exact_state_cache_audit_v1_0_41.sqlite"
    for suffix in ("","-wal","-shm"):
        fp=Path(str(audit_db)+suffix)
        if fp.exists():
            fp.unlink()

    protocol=_exact_cache_protocol(cfg,engine,static)
    cache=ExactEvaluationStateCache(audit_db,protocol)
    rows=[]
    tstart=time.perf_counter()

    # Strict equivalence on the first seed/scenario.
    seed=int(seeds[0])
    r=next(test.itertuples())
    sid=int(r.scenario_id)
    d=np.load(
        paths.scenarios/f"scenario_{sid:05d}.npz"
    )["degradation"]
    model,st=load_graph_model(
        cfg,paths,engine,edges,zones,seed,None
    )
    pol1=deterministic_graph_policy(
        cfg,model,st,static,engine.device,frozen=False
    )
    z0,q0,s0=trajectory_rollout(
        cfg,engine,static,edges,d,pol1,seed+sid,True
    )
    pol2=deterministic_graph_policy(
        cfg,model,st,static,engine.device,frozen=False
    )
    z1,q1,s1=trajectory_rollout_exact_cached(
        cfg,engine,static,edges,d,pol2,seed+sid,
        sid,cache,True
    )
    max_step=max(
        max(abs(float(a[k])-float(b[k])) for k in ("Lacc","Leq","Lv","Lcomp"))
        for a,b in zip(s0,s1)
    )
    eq_pass=bool(
        abs(z0-z1)<=1e-12
        and abs(q0-q1)<=1e-12
        and max_step<=1e-12
        and [x["action"] for x in s0]==[x["action"] for x in s1]
    )
    del model,st
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reuse audit over B5 10 seeds x 3 held-out scenarios.
    for seed in seeds:
        model,st=load_graph_model(
            cfg,paths,engine,edges,zones,int(seed),None
        )
        for r in test.itertuples():
            sid=int(r.scenario_id)
            d=np.load(
                paths.scenarios/f"scenario_{sid:05d}.npz"
            )["degradation"]
            pol=deterministic_graph_policy(
                cfg,model,st,static,engine.device,frozen=False
            )
            before_h=cache.hits
            before_m=cache.misses
            t0=time.perf_counter()
            z,q,steps=trajectory_rollout_exact_cached(
                cfg,engine,static,edges,d,pol,
                int(seed)+sid,sid,cache,True
            )
            rows.append({
                "seed":int(seed),
                "scenario_id":sid,
                "Z":float(z),"Q":float(q),
                "elapsed_seconds":float(time.perf_counter()-t0),
                "state_hits":int(cache.hits-before_h),
                "state_misses":int(cache.misses-before_m),
                "action_sequence":"|".join(
                    map(str,[int(x["action"]) for x in steps])
                ),
            })
        del model,st
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache.close()
    df=pd.DataFrame(rows)
    df.to_csv(paths.eval/"evaluation_cache_audit.csv",index=False)
    out={
        "script_version":SCRIPT_VERSION,
        "equivalence_pass":eq_pass,
        "legacy_Z":float(z0),
        "cached_Z":float(z1),
        "abs_Z_error":float(abs(z0-z1)),
        "legacy_Q":float(q0),
        "cached_Q":float(q1),
        "abs_Q_error":float(abs(q0-q1)),
        "max_step_loss_component_error":float(max_step),
        "n_models":int(len(seeds)),
        "n_scenarios":int(len(test)),
        "state_requests":int(cache.requests),
        "state_hits":int(cache.hits),
        "state_misses":int(cache.misses),
        "state_hit_rate":float(cache.hit_rate),
        "unique_action_sequences":int(df.action_sequence.nunique()),
        "elapsed_seconds":float(time.perf_counter()-tstart),
        "exact_compute_seconds_on_cache_misses":float(
            cache.compute_seconds
        ),
        "interpretation":(
            "Cache hits reuse exact loss components only for identical "
            "scenario/action prefixes. Policy selection is always recomputed."
        ),
    }
    write_json(paths.eval/"evaluation_cache_audit_summary.json",out)
    log.log(
        "Exact-state cache audit: "
        f"equivalence_pass={eq_pass}, "
        f"hit_rate={out['state_hit_rate']:.1%}, "
        f"hits={out['state_hits']}, misses={out['state_misses']}, "
        f"unique_action_sequences={out['unique_action_sequences']}, "
        f"elapsed={out['elapsed_seconds']/60:.1f}min"
    )
    if not eq_pass:
        raise RuntimeError(
            "Exact-state cache failed strict equivalence audit."
        )
    return out



# ---------------------------------------------------------------------------
# v1.0.40: process-parallel audit across independent exact network states.
# Module-level worker state is required for Windows "spawn" multiprocessing.
# ---------------------------------------------------------------------------
_V140_EXACT_WORKER = None

def _v140_exact_worker_init(payload):
    global _V140_EXACT_WORKER
    _V140_EXACT_WORKER = payload

def _v140_exact_worker(task):
    """Legacy-exact accessibility for one independent state.

    The numerical kernel is intentionally the same SciPy CSR + directed
    Dijkstra construction as RouteEngine.accessibility_exact. Parallelism is
    only across independent states. Hence no scientific object is changed.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    idx, q, K = task
    p = _V140_EXACT_WORKER
    q=np.asarray(q,float); K=np.asarray(K,float)

    pair_q=np.full(len(p["pair_u"]),np.inf,dtype=float)
    usable=(K>1e-9) & np.isfinite(q)
    np.minimum.at(pair_q,p["edge_pair"][usable],q[usable])
    keep=np.isfinite(pair_q)

    mat=csr_matrix(
        (pair_q[keep],(p["pair_u"][keep],p["pair_v"][keep])),
        shape=(p["n_graph_nodes"],p["n_graph_nodes"])
    )
    dist=dijkstra(
        mat.transpose().tocsr(),
        directed=True,
        indices=p["service_node_i"],
        return_predecessors=False
    )
    dz=dist[:,p["zone_node_i"]].T
    contrib=np.exp(
        -p["kappa"]*dz,
        where=np.isfinite(dz),
        out=np.zeros_like(dz)
    )*p["service_O"][None,:]
    return int(idx),contrib.sum(axis=1)


def exact_accessibility_process_batch(
    engine, static, state_items, workers=4, verify_first=True
):
    """Production-ready exact accessibility for independent states.

    Parameters
    ----------
    state_items : list of dict
        Each item must contain ``q`` and ``K`` plus any caller metadata.
    workers : int
        Number of spawned CPU processes. v1.0.40 audited workers=4 on the
        target machine.
    verify_first : bool
        If True, the first returned state is compared against the legacy exact
        solver at 1e-12 before results are released.

    Notes
    -----
    Parallelism is only across independent states. Every worker executes the
    same full-network directed SciPy Dijkstra definition as the legacy solver.
    Output order is identical to input order.
    """
    import multiprocessing as mp
    if not state_items:
        return []

    payload={
        "pair_u":np.asarray(engine._pair_u,dtype=np.int64),
        "pair_v":np.asarray(engine._pair_v,dtype=np.int64),
        "edge_pair":np.asarray(engine._edge_pair,dtype=np.int64),
        "n_graph_nodes":int(engine._n_graph_nodes),
        "zone_node_i":np.asarray(engine._zone_node_i,dtype=np.int64),
        "service_node_i":np.asarray(engine._service_node_i,dtype=np.int64),
        "service_O":np.asarray(engine._service_O,dtype=float),
        "kappa":float(engine.kappa),
    }
    tasks=[
        (i,np.asarray(x["q"],float),np.asarray(x["K"],float))
        for i,x in enumerate(state_items)
    ]
    nproc=max(1,min(int(workers),len(tasks)))
    if nproc==1:
        _v140_exact_worker_init(payload)
        raw=[_v140_exact_worker(x) for x in tasks]
    else:
        ctx=mp.get_context("spawn")
        with ctx.Pool(
            processes=nproc,
            initializer=_v140_exact_worker_init,
            initargs=(payload,)
        ) as pool:
            raw=pool.map(_v140_exact_worker,tasks,chunksize=1)

    raw=sorted(raw,key=lambda z:z[0])
    out=[]
    for i,A in raw:
        item=dict(state_items[int(i)])
        item["A_exact"]=np.asarray(A,float)
        out.append(item)

    if verify_first:
        x=state_items[0]
        A0=engine.accessibility_exact(
            np.asarray(x["q"],float),np.asarray(x["K"],float)
        )
        A1=out[0]["A_exact"]
        aerr=float(np.max(np.abs(A1-A0)))
        rel=float(np.max(
            np.abs(A1-A0)/np.maximum(np.abs(A0),1e-15)
        ))
        _,L0,E0,_,_=loss_components(
            A0,static.A0,static.population,static.vulnerable
        )
        _,L1,E1,_,_=loss_components(
            A1,static.A0,static.population,static.vulnerable
        )
        if not (
            np.array_equal(np.isfinite(A1),np.isfinite(A0))
            and aerr<=1e-12 and rel<=1e-12
            and abs(L1-L0)<=1e-12 and abs(E1-E0)<=1e-12
        ):
            raise RuntimeError(
                "Production exact-process equivalence gate FAILED: "
                f"A_abs={aerr:.3e}, A_rel={rel:.3e}, "
                f"Lacc={abs(L1-L0):.3e}, Leq={abs(E1-E0):.3e}"
            )
    return out


def b4_rank_policy_audit(cfg,paths,log):
    """Held-out admissibility/non-degeneracy audit for the v1.0.38+ B4 rank plan.

    This audit is decision-only: it does not compute exact accessibility.
    It verifies that the ex-ante rank plan maps into admissible R3/Top-K actions
    on held-out scenarios and quantifies no-action frequency. It must pass
    before B4 is used as the principal static/open-loop comparator.
    """
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(cfg,paths,edges,log)
    seq=b4_open_loop_search(cfg,paths,engine,static,edges,log)
    pol=b4_policy_factory(seq)

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].sort_values(
        ["split","scenario_id"]
    )
    seed=int(active_numerical_seeds(cfg)[0])
    rows=[]
    for r in test.itertuples():
        sid=int(r.scenario_id)
        d=np.load(
            paths.scenarios/f"scenario_{sid:05d}.npz"
        )["degradation"]
        actions=trajectory_action_prefix_only(
            cfg,engine,static,edges,d,pol,seed+sid
        )
        rows.append({
            "scenario_id":sid,
            "split":str(r.split),
            "action_sequence":"|".join(map(str,map(int,actions))),
            "n_interventions":int(sum(int(a)>=0 for a in actions)),
            "all_no_action":bool(all(int(a)<0 for a in actions)),
        })

    df=pd.DataFrame(rows)
    df.to_csv(paths.eval/"B4_rank_policy_audit.csv",index=False)
    n=len(df)
    all_no=int(df.all_no_action.sum())
    mean_n=float(df.n_interventions.mean())
    nondeg=bool(all_no<n and mean_n>0.0)

    out={
        "script_version":SCRIPT_VERSION,
        "benchmark":"B4",
        "rank_sequence":[int(x) for x in seq],
        "n_heldout_scenarios":int(n),
        "all_no_action_scenarios":all_no,
        "all_no_action_fraction":float(all_no/max(n,1)),
        "mean_interventions":mean_n,
        "unique_action_sequences":int(df.action_sequence.nunique()),
        "nondegeneracy_pass":nondeg,
        "scientific_guard":(
            "The rank sequence is fixed ex ante; only its deterministic mapping "
            "to the current admissible R3/Top-K correspondence is state-dependent."
        ),
    }
    write_json(paths.eval/"B4_rank_policy_audit_summary.json",out)
    if not nondeg:
        raise RuntimeError(
            "B4 rank-plan audit FAILED: benchmark remains degenerate on held-out "
            "scenarios. Do not launch exact evaluation."
        )
    log.log(
        "B4 rank-plan audit PASS: "
        f"rank_plan={seq}, all-no-action={all_no}/{n}, "
        f"mean interventions={mean_n:.3f}, "
        f"unique sequences={out['unique_action_sequences']}"
    )
    return out



def exact_state_process_parallel_audit(cfg,paths,log):
    """Final exact-runtime audit before accepting a long production run.

    v1.0.39 established that splitting service-source Dijkstra calls across
    threads gives no material speedup. v1.0.40 instead benchmarks independent
    exact states concurrently in separate processes. This matches the natural
    evaluation workload: states from different held-out trajectories are
    independent once their (q,K) objects have been generated.

    The audit:
      1. builds eight representative held-out routed states;
      2. computes the legacy exact accessibility serially;
      3. recomputes the same states with 2 and 4 spawned processes;
      4. requires strict statewise equality of A, Lacc, and Leq;
      5. reports WALL-CLOCK throughput speedup.

    No production evaluation path is modified by this stage.
    """
    import multiprocessing as mp

    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].sort_values(
        ["split","scenario_id"]
    ).head(8)

    states=[]
    for r in test.itertuples():
        sid=int(r.scenario_id)
        d=np.load(
            paths.scenarios/f"scenario_{sid:05d}.npz"
        )["degradation"]
        # Two dates are used when possible to vary topology/cost patterns.
        tt=0 if (len(states)%2==0) else min(1,d.shape[0]-1)
        K=(1-d[tt])*static.K0
        _,q,_=engine.solve(K)
        states.append((sid,str(r.split),tt,
                       np.asarray(q,float),np.asarray(K,float)))

    # Serial legacy reference.
    refs={}
    serial_rows=[]
    t_serial=time.perf_counter()
    for i,(sid,split,tt,q,K) in enumerate(states):
        ta=time.perf_counter()
        A=engine.accessibility_exact(q,K)
        sec=time.perf_counter()-ta
        _,La,Le,_,_=loss_components(
            A,static.A0,static.population,static.vulnerable
        )
        refs[i]=(A,float(La),float(Le))
        serial_rows.append({
            "mode":"serial_legacy","workers":1,"state_index":i,
            "scenario_id":sid,"t":tt,"seconds_individual":sec,
            "max_abs_A_error":0.0,"abs_Lacc_error":0.0,
            "abs_Leq_error":0.0,"equivalence_pass":True,
        })
    serial_wall=time.perf_counter()-t_serial

    payload={
        "pair_u":np.asarray(engine._pair_u,dtype=np.int64),
        "pair_v":np.asarray(engine._pair_v,dtype=np.int64),
        "edge_pair":np.asarray(engine._edge_pair,dtype=np.int64),
        "n_graph_nodes":int(engine._n_graph_nodes),
        "zone_node_i":np.asarray(engine._zone_node_i,dtype=np.int64),
        "service_node_i":np.asarray(engine._service_node_i,dtype=np.int64),
        "service_O":np.asarray(engine._service_O,dtype=float),
        "kappa":float(engine.kappa),
    }

    rows=list(serial_rows)
    summaries=[]
    atol_A=1e-12
    rtol_A=1e-12
    atol_loss=1e-12

    # Windows-safe spawn. More than four workers is deliberately not tested:
    # each process traverses a 328k-node graph and memory bandwidth can dominate.
    for workers in [2,4]:
        ctx=mp.get_context("spawn")
        tasks=[
            (i,np.asarray(q,float),np.asarray(K,float))
            for i,(_,_,_,q,K) in enumerate(states)
        ]
        tpar=time.perf_counter()
        with ctx.Pool(
            processes=workers,
            initializer=_v140_exact_worker_init,
            initargs=(payload,)
        ) as pool:
            results=pool.map(_v140_exact_worker,tasks,chunksize=1)
        wall=time.perf_counter()-tpar

        all_pass=True
        maxA=maxR=maxL=maxE=0.0
        for i,A1 in results:
            A0,L0,E0=refs[int(i)]
            _,L1,E1,_,_=loss_components(
                A1,static.A0,static.population,static.vulnerable
            )
            aerr=float(np.max(np.abs(A1-A0)))
            rel=float(np.max(
                np.abs(A1-A0)/np.maximum(np.abs(A0),1e-15)
            ))
            dl=float(abs(L1-L0)); de=float(abs(E1-E0))
            ok=bool(
                np.array_equal(np.isfinite(A1),np.isfinite(A0))
                and aerr<=atol_A and rel<=rtol_A
                and dl<=atol_loss and de<=atol_loss
            )
            all_pass &= ok
            maxA=max(maxA,aerr); maxR=max(maxR,rel)
            maxL=max(maxL,dl); maxE=max(maxE,de)
            sid,split,tt,_,_=states[int(i)]
            rows.append({
                "mode":"process_parallel","workers":workers,
                "state_index":int(i),"scenario_id":sid,"t":tt,
                "seconds_individual":np.nan,
                "max_abs_A_error":aerr,"abs_Lacc_error":dl,
                "abs_Leq_error":de,"equivalence_pass":ok,
            })

        speedup=serial_wall/max(wall,1e-12)
        summaries.append({
            "workers":workers,
            "serial_wall_seconds":float(serial_wall),
            "parallel_wall_seconds":float(wall),
            "wall_speedup":float(speedup),
            "all_equivalent":bool(all_pass),
            "max_abs_A_error":maxA,
            "max_rel_A_error":maxR,
            "max_abs_Lacc_error":maxL,
            "max_abs_Leq_error":maxE,
        })
        log.log(
            f"Exact state-process audit workers={workers}: "
            f"serial={serial_wall:.2f}s, parallel={wall:.2f}s, "
            f"wall speedup={speedup:.2f}x, equivalent={all_pass}, "
            f"Aerr={maxA:.3e}, Lacc={maxL:.3e}, Leq={maxE:.3e}"
        )

    pd.DataFrame(rows).to_csv(
        paths.eval/"exact_state_process_parallel_audit.csv",index=False
    )
    sdf=pd.DataFrame(summaries)
    sdf.to_csv(
        paths.eval/"exact_state_process_parallel_summary.csv",index=False
    )

    eligible=sdf[sdf.all_equivalent & (sdf.wall_speedup>=1.10)]
    if len(eligible):
        best=eligible.sort_values(
            ["parallel_wall_seconds","workers"],kind="stable"
        ).iloc[0]
        best_workers=int(best.workers)
        best_speedup=float(best.wall_speedup)
        best_wall=float(best.parallel_wall_seconds)
    else:
        best_workers=None; best_speedup=None; best_wall=None

    out={
        "script_version":SCRIPT_VERSION,
        "candidate":"process_parallel_independent_exact_states",
        "n_states":int(len(states)),
        "worker_grid":[2,4],
        "serial_wall_seconds":float(serial_wall),
        "tolerances":{
            "accessibility_abs":atol_A,
            "accessibility_rel":rtol_A,
            "loss_abs":atol_loss,
        },
        "results":summaries,
        "best_equivalent_workers":best_workers,
        "best_wall_speedup":best_speedup,
        "best_parallel_wall_seconds":best_wall,
        "production_solver_changed":False,
        "promotion_threshold_wall_speedup":1.10,
        "scientific_guard":(
            "Parallelism is only across independent exact states; every state "
            "uses the legacy full-network directed Dijkstra definition."
        ),
    }
    write_json(
        paths.eval/"exact_state_process_parallel_audit_summary.json",out
    )

    if best_workers is None:
        log.log(
            "Exact state-process audit COMPLETE: no material equivalent "
            "parallel speedup. Stop runtime optimization and retain legacy."
        )
    else:
        log.log(
            "Exact state-process audit COMPLETE: best equivalent "
            f"workers={best_workers}, wall speedup={best_speedup:.2f}x. "
            "Candidate is eligible for production integration."
        )
    return out



def exact_process_production_audit(cfg,paths,log):
    """Final gate for the reusable 4-process exact-state batch evaluator."""
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].sort_values(
        ["split","scenario_id"]
    ).head(12)

    items=[]
    refs=[]
    for k,r in enumerate(test.itertuples()):
        sid=int(r.scenario_id)
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        tt=k % min(3,d.shape[0])
        K=(1-d[tt])*static.K0
        _,q,_=engine.solve(K)
        items.append({"scenario_id":sid,"t":tt,"q":q,"K":K})

    ts=time.perf_counter()
    for x in items:
        A=engine.accessibility_exact(x["q"],x["K"])
        _,La,Le,_,_=loss_components(
            A,static.A0,static.population,static.vulnerable
        )
        refs.append((A,float(La),float(Le)))
    serial=time.perf_counter()-ts

    tp=time.perf_counter()
    got=exact_accessibility_process_batch(
        engine,static,items,workers=4,verify_first=True
    )
    parallel=time.perf_counter()-tp

    rows=[]
    all_pass=True
    for i,(x,(A0,L0,E0)) in enumerate(zip(got,refs)):
        A1=x["A_exact"]
        _,L1,E1,_,_=loss_components(
            A1,static.A0,static.population,static.vulnerable
        )
        ae=float(np.max(np.abs(A1-A0)))
        re=float(np.max(np.abs(A1-A0)/np.maximum(np.abs(A0),1e-15)))
        dl=float(abs(L1-L0)); de=float(abs(E1-E0))
        ok=bool(
            np.array_equal(np.isfinite(A1),np.isfinite(A0))
            and ae<=1e-12 and re<=1e-12
            and dl<=1e-12 and de<=1e-12
        )
        all_pass &= ok
        rows.append({
            "scenario_id":int(x["scenario_id"]),"t":int(x["t"]),
            "max_abs_A_error":ae,"max_rel_A_error":re,
            "abs_Lacc_error":dl,"abs_Leq_error":de,
            "equivalence_pass":ok,
        })

    speedup=serial/max(parallel,1e-12)
    pd.DataFrame(rows).to_csv(
        paths.eval/"exact_process_production_audit.csv",index=False
    )
    out={
        "script_version":SCRIPT_VERSION,
        "workers":4,
        "n_states":len(items),
        "serial_wall_seconds":float(serial),
        "parallel_wall_seconds":float(parallel),
        "wall_speedup":float(speedup),
        "all_equivalent":bool(all_pass),
        "production_batch_function_ready":bool(all_pass and speedup>=1.10),
        "production_evaluate_all_switched":False,
        "reason_evaluate_all_not_switched":(
            "Current trajectory evaluation is sequential within a trajectory. "
            "Blindly parallelizing its internal states would change action-state "
            "dependencies. The audited batch function is safe only where callers "
            "expose independent exact states."
        ),
    }
    write_json(
        paths.eval/"exact_process_production_audit_summary.json",out
    )
    log.log(
        "Exact process production audit: "
        f"serial={serial:.2f}s parallel={parallel:.2f}s "
        f"speedup={speedup:.2f}x equivalent={all_pass}"
    )
    if not out["production_batch_function_ready"]:
        raise RuntimeError(
            "4-process production exact batch did not pass the final gate."
        )
    return out



def exact_accessibility_fast_audit(cfg,paths,log):
    """Benchmark source-parallel exact accessibility against the legacy solver.

    No production solver is changed here. The candidate must reproduce the
    legacy zone accessibility vector and downstream loss functionals within
    strict tolerances on multiple held-out network states. Worker counts are
    benchmarked on the actual machine because thread scaling is platform and
    SciPy-build dependent.
    """
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].sort_values(
        ["split","scenario_id"]
    ).head(6)

    worker_grid=[1,2,4,8]
    rows=[]
    state_cache=[]
    # Build six representative routed states once; routing is not part of this
    # candidate benchmark.
    for r in test.itertuples():
        sid=int(r.scenario_id)
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        K=(1-d[0])*static.K0
        _,q,_=engine.solve(K)
        t0=time.perf_counter()
        A0=engine.accessibility_exact(q,K)
        legacy_s=time.perf_counter()-t0
        _,L0,E0,_,_=loss_components(
            A0,static.A0,static.population,static.vulnerable
        )
        state_cache.append((sid,str(r.split),K,q,A0,L0,E0,legacy_s))

    for workers in worker_grid:
        for sid,split,K,q,A0,L0,E0,legacy_s in state_cache:
            # One warm-up per worker setting/state is intentionally avoided:
            # reported timings include ordinary invocation overhead.
            t1=time.perf_counter()
            A1=engine.accessibility_exact_threaded(q,K,workers=workers)
            fast_s=time.perf_counter()-t1
            _,L1,E1,_,_=loss_components(
                A1,static.A0,static.population,static.vulnerable
            )
            absA=float(np.max(np.abs(A1-A0)))
            scale=np.maximum(np.abs(A0),1e-15)
            relA=float(np.max(np.abs(A1-A0)/scale))
            dL=float(abs(L1-L0)); dE=float(abs(E1-E0))
            finite_equal=bool(
                np.array_equal(np.isfinite(A1),np.isfinite(A0))
            )
            rows.append({
                "scenario_id":sid,"split":split,"workers":workers,
                "legacy_seconds":legacy_s,
                "candidate_seconds":fast_s,
                "speedup":legacy_s/max(fast_s,1e-12),
                "max_abs_A_error":absA,
                "max_rel_A_error":relA,
                "abs_Lacc_error":dL,
                "abs_Leq_error":dE,
                "finite_pattern_equal":finite_equal,
            })
            log.log(
                f"Fast exact audit scenario={sid} workers={workers}: "
                f"{fast_s:.3f}s vs legacy {legacy_s:.3f}s, "
                f"speedup={legacy_s/max(fast_s,1e-12):.2f}x, "
                f"Aerr={absA:.3e}, Lacc={dL:.3e}, Leq={dE:.3e}"
            )

    df=pd.DataFrame(rows)
    # Exact service-source partitioning should normally be bit-identical after
    # reassembly. We nevertheless use strict numerical guards for portability.
    atol_A=1e-12
    rtol_A=1e-12
    atol_loss=1e-12
    df["equivalence_pass"]=(
        df.finite_pattern_equal
        & (df.max_abs_A_error<=atol_A)
        & (df.max_rel_A_error<=rtol_A)
        & (df.abs_Lacc_error<=atol_loss)
        & (df.abs_Leq_error<=atol_loss)
    )
    df.to_csv(paths.eval/"exact_accessibility_fast_audit.csv",index=False)

    agg=(
        df.groupby("workers",as_index=False)
        .agg(
            mean_candidate_seconds=("candidate_seconds","mean"),
            median_candidate_seconds=("candidate_seconds","median"),
            mean_speedup=("speedup","mean"),
            min_speedup=("speedup","min"),
            all_equivalent=("equivalence_pass","all"),
            max_abs_A_error=("max_abs_A_error","max"),
            max_abs_Lacc_error=("abs_Lacc_error","max"),
            max_abs_Leq_error=("abs_Leq_error","max"),
        )
    )
    eligible=agg[agg.all_equivalent]
    if len(eligible):
        best=eligible.sort_values(
            ["mean_candidate_seconds","workers"],kind="stable"
        ).iloc[0]
        best_workers=int(best.workers)
        best_seconds=float(best.mean_candidate_seconds)
        best_speedup=float(best.mean_speedup)
    else:
        best_workers=None; best_seconds=None; best_speedup=None

    agg.to_csv(
        paths.eval/"exact_accessibility_fast_audit_by_workers.csv",index=False
    )
    out={
        "script_version":SCRIPT_VERSION,
        "candidate":"source_parallel_scipy_dijkstra_threads",
        "n_states":int(len(state_cache)),
        "worker_grid":worker_grid,
        "tolerances":{
            "accessibility_abs":atol_A,
            "accessibility_rel":rtol_A,
            "loss_abs":atol_loss,
        },
        "all_worker_results":agg.to_dict(orient="records"),
        "best_equivalent_workers":best_workers,
        "best_mean_candidate_seconds":best_seconds,
        "best_mean_speedup":best_speedup,
        "production_solver_changed":False,
        "promotion_rule":(
            "Promote only an equivalent worker setting with material measured "
            "speedup on the target machine; otherwise retain legacy exact solver."
        ),
    }
    write_json(
        paths.eval/"exact_accessibility_fast_audit_summary.json",out
    )
    if best_workers is None:
        log.log(
            "Fast exact audit COMPLETE: no candidate passed strict equivalence; "
            "production solver remains legacy."
        )
    else:
        log.log(
            "Fast exact audit COMPLETE: best equivalent setting "
            f"workers={best_workers}, mean={best_seconds:.3f}s, "
            f"speedup={best_speedup:.2f}x; production solver remains unchanged."
        )
    return out



def exact_accessibility_speed_audit(cfg,paths,log):
    """Profile the exact reported-accessibility bottleneck without changing it.

    v1.0.38 deliberately does not replace the production exact solver until a
    candidate implementation passes numerical equivalence on controlled states.
    This stage reports solve-time versus exact-accessibility-time separately on
    representative held-out states.
    """
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].sort_values(
        ["split","scenario_id"]
    ).head(5)
    rows=[]
    for r in test.itertuples():
        sid=int(r.scenario_id)
        d=np.load(
            paths.scenarios/f"scenario_{sid:05d}.npz"
        )["degradation"]
        # t=0 no-intervention state gives a reproducible representative state.
        K=(1-d[0])*static.K0
        t0=time.perf_counter()
        _,q,_=engine.solve(K)
        solve_s=time.perf_counter()-t0
        t1=time.perf_counter()
        A=engine.accessibility_exact(q,K)
        exact_s=time.perf_counter()-t1
        _,La,Le,_,_=loss_components(
            A,static.A0,static.population,static.vulnerable
        )
        rows.append({
            "scenario_id":sid,
            "split":str(r.split),
            "routing_solve_seconds":float(solve_s),
            "exact_accessibility_seconds":float(exact_s),
            "total_seconds":float(solve_s+exact_s),
            "Lacc":float(La),"Leq":float(Le),
        })
        log.log(
            f"Exact speed probe scenario={sid}: routing={solve_s:.3f}s, "
            f"accessibility_exact={exact_s:.3f}s, total={solve_s+exact_s:.3f}s"
        )

    df=pd.DataFrame(rows)
    df.to_csv(paths.eval/"exact_accessibility_speed_audit.csv",index=False)
    out={
        "script_version":SCRIPT_VERSION,
        "n_states":int(len(df)),
        "mean_routing_seconds":float(df.routing_solve_seconds.mean()),
        "mean_exact_accessibility_seconds":
            float(df.exact_accessibility_seconds.mean()),
        "mean_total_seconds":float(df.total_seconds.mean()),
        "accessibility_share_of_total":float(
            df.exact_accessibility_seconds.sum()/df.total_seconds.sum()
        ),
        "production_solver_changed":False,
        "decision_rule":(
            "Do not replace exact_full_network until a faster implementation "
            "passes statewise accessibility and loss equivalence."
        ),
    }
    write_json(paths.eval/"exact_accessibility_speed_audit_summary.json",out)
    log.log(
        "Exact accessibility speed audit COMPLETE: "
        f"mean total={out['mean_total_seconds']:.3f}s/state; "
        f"exact accessibility share={out['accessibility_share_of_total']:.1%}"
    )
    return out



def _evaluation_model_inventory(cfg,paths):
    """Read-only inventory. Evaluation must never train or overwrite models."""
    seeds=active_numerical_seeds(cfg)
    specs=["B5","A1","A2","A3","A4","A5"]
    if bool(cfg.get("graph_ppo",{}).get("enabled",False)):
        specs.append("B5_PPO")
    rows=[]
    for spec in specs:
        for seed in seeds:
            p=paths.models/f"{spec}_seed_{int(seed)}.pt"
            rows.append({
                "spec":spec,
                "seed":int(seed),
                "path":str(p),
                "exists":bool(p.exists()),
                "size_bytes":int(p.stat().st_size) if p.exists() else 0,
            })
    return pd.DataFrame(rows)


def evaluation_preflight(cfg,paths,log):
    """Read-only preflight for the expensive exact evaluation.

    It checks all required learned checkpoints, test scenario counts, cached
    B3/B4 objects, and times ONE exact B4 trajectory. No model is trained and
    no production checkpoint is modified.
    """
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(cfg,paths,edges,log)
    b3scores=compute_b3_scores(cfg,paths,engine,static,edges,log)
    b4seq=b4_open_loop_search(cfg,paths,engine,static,edges,log)

    inv=_evaluation_model_inventory(cfg,paths)
    inv_path=paths.eval/"evaluation_model_inventory.csv"
    inv.to_csv(inv_path,index=False)
    missing=inv.loc[~inv.exists].copy()

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].copy()
    counts=test.groupby("split").size().to_dict()

    probe={}
    if len(test):
        r=next(test.itertuples())
        sid=int(r.scenario_id)
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        pol=b4_policy_factory(b4seq)
        t0=time.perf_counter()
        z,q,steps=trajectory_rollout(
            cfg,engine,static,edges,d,pol,
            int(active_numerical_seeds(cfg)[0])+sid,True
        )
        sec=float(time.perf_counter()-t0)
        probe={
            "scenario_id":sid,
            "split":str(r.split),
            "seconds":sec,
            "Z":float(z),
            "Q":float(q),
            "n_steps":int(len(steps)),
        }

    n_fixed_model_runs=10+4  # B0 ten seeds + B1-B4 one seed each
    n_learned_model_runs=len(inv)
    n_scen=int(len(test))
    # This is deliberately a transparent first-order estimate only.
    # Learned policy scoring adds GNN overhead, while B2 action selection adds
    # surrogate ranking overhead; therefore do not call it a runtime guarantee.
    probe_sec=float(probe.get("seconds",0.0))
    rough_hours=(
        (n_fixed_model_runs+n_learned_model_runs)*n_scen*probe_sec/3600.0
        if probe_sec>0 else None
    )

    out={
        "script_version":SCRIPT_VERSION,
        "candidate_action_protocol":final_action_protocol_signature(cfg),
        "evaluation_accessibility":"exact_full_network",
        "n_test_scenarios_total":n_scen,
        "scenario_counts":{str(k):int(v) for k,v in counts.items()},
        "n_required_learned_checkpoints":int(len(inv)),
        "n_missing_learned_checkpoints":int(len(missing)),
        "missing_models":[
            {"spec":str(r.spec),"seed":int(r.seed)}
            for r in missing.itertuples()
        ],
        "b3_scored_edges":int(np.isfinite(b3scores).sum()),
        "b4_sequence":[int(x) for x in b4seq],
        "single_exact_B4_probe":probe,
        "rough_runtime_hours_from_B4_probe":rough_hours,
        "rough_runtime_warning":(
            "First-order probe only; B2 and learned policies have additional "
            "policy-selection cost. It is not a runtime guarantee."
        ),
        "evaluation_writes_model_checkpoints":False,
        "ablation_retraining_required_by_definition":{
            "A1":True,"A2":True,"A3":True,"A4":True,"A5":True
        },
        "ablation_reason":(
            "A1 changes architecture; A2 changes policy input; "
            "A3/A4 change the training objective; A5 changes sequential "
            "score updating during training and validation."
        ),
    }
    write_json(paths.eval/"evaluation_preflight_summary.json",out)

    log.log(
        f"Evaluation preflight: test scenarios={n_scen}, "
        f"learned checkpoints={len(inv)-len(missing)}/{len(inv)}, "
        f"missing={len(missing)}"
    )
    if probe:
        log.log(
            f"Exact B4 probe scenario={probe['scenario_id']}: "
            f"{probe['seconds']/60:.2f}min, Z={probe['Z']:.9g}, "
            f"Q={probe['Q']:.9g}"
        )
    if rough_hours is not None:
        log.log(
            f"First-order full-evaluation runtime from B4 probe: "
            f"~{rough_hours:.1f}h (diagnostic only)"
        )
    if len(missing):
        log.log(
            "PREFLIGHT BLOCKED: missing learned checkpoints. "
            "Evaluation v1.0.35 will not train them implicitly."
        )
    else:
        log.log(
            "PREFLIGHT PASS: all learned checkpoints exist; exact evaluation "
            "can run without modifying model files."
        )
    return out


def _evaluation_checkpoint_paths(paths):
    return (
        paths.eval/"policy_trajectory_results_work.csv",
        paths.eval/"policy_step_results_work.csv",
        paths.eval/"policy_evaluation_completed_models.json",
    )


def _load_evaluation_work(paths, protocol_tag):
    rp,sp,cp=_evaluation_checkpoint_paths(paths)
    rows=pd.read_csv(rp).to_dict("records") if rp.exists() else []
    steps=pd.read_csv(sp).to_dict("records") if sp.exists() else []
    completed=set()
    if cp.exists():
        obj=read_json(cp)
        if obj.get("protocol_tag")==protocol_tag:
            completed=set(map(str,obj.get("completed_models",[])))
        elif rows or steps:
            raise RuntimeError(
                "Evaluation work files exist under an incompatible protocol. "
                "Archive/remove only the *_work files before restarting."
            )
    elif rows or steps:
        raise RuntimeError(
            "Evaluation work CSVs exist without their protocol marker. "
            "Refusing ambiguous resume."
        )
    return rows,steps,completed


def _save_evaluation_work(paths, rows, steps, completed, protocol_tag):
    rp,sp,cp=_evaluation_checkpoint_paths(paths)
    _atomic_csv(pd.DataFrame(rows),rp)
    _atomic_csv(pd.DataFrame(steps),sp)
    write_json(cp,{
        "protocol_tag":protocol_tag,
        "completed_models":sorted(completed),
        "n_trajectory_rows":int(len(rows)),
        "n_step_rows":int(len(steps)),
        "updated_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
    })


def _evaluation_protocol_tag(cfg, test):
    ids=",".join(map(str,test.scenario_id.astype(int).tolist()))
    sig=json.dumps(final_action_protocol_signature(cfg),sort_keys=True)
    payload=(
        f"v1.0.35_exact_full_network|{sig}|"
        f"seeds={active_numerical_seeds(cfg)}|scenarios={ids}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def trajectory_exact_state_prefixes(cfg,engine,static,edges,degradation,policy_fn,seed,scenario_id):
    """Yield policy prefixes and controlled K with bounded transient memory."""
    rng=np.random.default_rng(seed); costs=intervention_costs(cfg,edges)
    B=baseline_budget(cfg,costs); budget=float(B)
    restore=np.zeros_like(static.K0,float)
    retention=float(cfg["intervention"]["restoration_retention"])
    frac=float(cfg["intervention"]["restoration_fraction_of_remaining_deficit"])
    cap=int(cfg["intervention"]["max_candidate_edges_per_step"])
    actions=[]; stok=_scenario_token(scenario_id,degradation)
    for t in range(degradation.shape[0]):
        if t>0: restore*=retention
        Kexo=(1-degradation[t])*static.K0
        deficit=np.maximum(static.K0-(Kexo+restore),0)
        Kpre=np.minimum(Kexo+restore,static.K0)
        feas=policy_candidate_edges(cfg,engine,Kpre,static.K0,costs,budget,cap)
        # Minimal state: omit multi-MB copies of Kexo/restore/costs. All final
        # policies use t,budget,B,Kpre,degradation,feasible (plus their own RNG).
        state={"t":t,"budget":budget,"B":B,"Kpre":Kpre,
               "feasible":feas,"degradation":degradation[t],"rng":rng}
        action=int(policy_fn(state)) if len(feas) else -1
        if action>=0:
            if action not in set(map(int,feas)):
                raise RuntimeError(f"Precompute infeasible action {action} at t={t}")
            restore[action]+=frac*deficit[action]; budget-=costs[action]
        actions.append(action)
        Kctl=np.minimum(Kexo+restore,static.K0)
        yield stok,int(t),tuple(map(int,actions)),Kctl


def evaluation_exact_cache_parallel_precompute(cfg,paths,log):
    """Bounded-memory, deduplicated exact-cache production with persistent workers."""
    import multiprocessing as mp
    torch=require_torch()
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(cfg,paths,edges,log)
    inv=_evaluation_model_inventory(cfg,paths); missing=inv.loc[~inv.exists]
    if len(missing):
        raise RuntimeError(f"Parallel exact-cache precompute: {len(missing)} checkpoint(s) missing.")
    b3scores=compute_b3_scores(cfg,paths,engine,static,edges,log)
    b4seq=b4_open_loop_search(cfg,paths,engine,static,edges,log)
    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].copy()
    test=test.sort_values(["split","scenario_id"]).reset_index(drop=True)
    rows=list(test.itertuples()); seeds=active_numerical_seeds(cfg)
    protocol=_exact_cache_protocol(cfg,engine,static)
    cache=ExactEvaluationStateCache(paths.eval/"exact_state_cache_v1_0_41.sqlite",protocol)

    payload={"pair_u":np.asarray(engine._pair_u,dtype=np.int64),
             "pair_v":np.asarray(engine._pair_v,dtype=np.int64),
             "edge_pair":np.asarray(engine._edge_pair,dtype=np.int64),
             "n_graph_nodes":int(engine._n_graph_nodes),
             "zone_node_i":np.asarray(engine._zone_node_i,dtype=np.int64),
             "service_node_i":np.asarray(engine._service_node_i,dtype=np.int64),
             "service_O":np.asarray(engine._service_O,dtype=float),
             "kappa":float(engine.kappa)}
    ctx=mp.get_context("spawn"); workers=4; chunk=16
    pool=ctx.Pool(processes=workers,initializer=_v140_exact_worker_init,initargs=(payload,))
    seen=set(); queue=[]; requests=0; unique_uncached=0; completed_exact=0
    started=time.perf_counter(); verified=False

    def flush():
        nonlocal queue,completed_exact,verified
        if not queue: return
        tasks=[]
        for i,x in enumerate(queue):
            tasks.append((i,x["q"],x["K"]))
        raw=pool.map(_v140_exact_worker,tasks,chunksize=1)
        raw=sorted(raw,key=lambda z:z[0])
        for i,A in raw:
            x=queue[int(i)]; A=np.asarray(A,float)
            if not verified:
                A0=engine.accessibility_exact(x["q"],x["K"])
                ae=float(np.max(np.abs(A-A0)))
                re=float(np.max(np.abs(A-A0)/np.maximum(np.abs(A0),1e-15)))
                if ae>1e-12 or re>1e-12:
                    raise RuntimeError(
                        f"Persistent exact worker equivalence FAILED: abs={ae:.3e}, rel={re:.3e}"
                    )
                log.log(f"Persistent exact worker equivalence PASS: abs={ae:.3e}, rel={re:.3e}")
                verified=True
            _,La,Le,Lv,Lc=loss_components(A,static.A0,static.population,static.vulnerable)
            cache.put(x["stok"],x["t"],x["actions"],La,Le,Lv,Lc)
            completed_exact+=1
        queue=[]

    def consume(states):
        nonlocal requests,unique_uncached,queue
        for stok,t,actions,K in states:
            requests+=1
            key=(stok,t,actions)
            if key in seen: continue
            seen.add(key)
            if cache.get(stok,t,actions) is not None: continue
            unique_uncached+=1
            # q and K live only until this small queue is flushed.
            _,q,_=engine.solve(K)
            queue.append({"stok":stok,"t":t,"actions":actions,
                          "q":np.asarray(q,float),"K":np.asarray(K,float).copy()})
            if len(queue)>=chunk:
                flush()
                if completed_exact%320==0:
                    elapsed=time.perf_counter()-started
                    rate=completed_exact/max(elapsed,1e-9)
                    log.log(f"Parallel exact-cache computed={completed_exact:,}, "
                            f"requests={requests:,}, unique seen={len(seen):,}, "
                            f"{rate:.3f} states/s")

    try:
        # B0 policy object is reused across scenarios exactly as evaluate_all.
        for seed in seeds:
            pol=b0_policy_factory(int(seed))
            for r in rows:
                sid=int(r.scenario_id); d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
                consume(trajectory_exact_state_prefixes(cfg,engine,static,edges,d,pol,int(seed)+sid,sid))
        log.log(f"Streaming prefixes B0 complete: requests={requests:,}, unique seen={len(seen):,}")

        fixed={"B1":lambda:b1_policy_factory(static.betweenness),
               "B2":lambda:b2_policy_factory(cfg,engine,static,edges),
               "B3":lambda:b3_policy_factory(b3scores),
               "B4":lambda:b4_policy_factory(b4seq)}
        seed0=int(seeds[0])
        for spec,fac in fixed.items():
            pol=fac()
            for r in rows:
                sid=int(r.scenario_id); d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
                consume(trajectory_exact_state_prefixes(cfg,engine,static,edges,d,pol,seed0+sid,sid))
            log.log(f"Streaming prefixes {spec} complete: requests={requests:,}, unique seen={len(seen):,}")

        learned=["B5","A1","A2","A3","A4","A5"]
        if bool(cfg.get("graph_ppo",{}).get("enabled",False)): learned.append("B5_PPO")
        for spec in learned:
            for seed in seeds:
                if spec=="B5_PPO":
                    model,st=load_graph_model(cfg,paths,engine,edges,zones,seed,spec="B5_PPO")
                else:
                    model,st=load_graph_model(cfg,paths,engine,edges,zones,seed,None if spec=="B5" else spec)
                for r in rows:
                    sid=int(r.scenario_id); d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
                    pol=deterministic_graph_policy(cfg,model,st,static,engine.device,frozen=(spec=="A5"))
                    consume(trajectory_exact_state_prefixes(cfg,engine,static,edges,d,pol,int(seed)+sid,sid))
                del model,st
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            log.log(f"Streaming prefixes {spec} complete: requests={requests:,}, unique seen={len(seen):,}")
        flush()
    finally:
        pool.close(); pool.join()

    elapsed=time.perf_counter()-started
    summary={"script_version":"1.0.47","protocol_hash":protocol,
             "requests":int(requests),"unique_seen_states":int(len(seen)),
             "unique_uncached_states_computed":int(unique_uncached),
             "workers":workers,"queue_chunk":chunk,
             "elapsed_seconds":float(elapsed),
             "reported_accessibility":"exact_full_network",
             "policy_or_checkpoint_changes":False,
             "bounded_memory_streaming":True,
             "persistent_worker_pool":True,
             "b4_rank_plan":list(map(int,b4seq))}
    write_json(paths.eval/"parallel_exact_cache_precompute_v1_0_47.json",summary)
    cache.close()
    log.log(f"Parallel exact-cache precompute COMPLETE: computed={unique_uncached:,}, "
            f"unique seen={len(seen):,}, requests={requests:,}, elapsed={elapsed/3600:.2f}h")



def evaluation_b2_freeze_precompute_v151(cfg,paths,log):
    """Freeze B2 sequences and guarantee exact-cache coverage for every prefix.

    v1.1.0-B fixes a resume bug in v1.1.0: when all 180 frozen B2 sequences
    already existed in the work JSON, the old code skipped every scenario and
    therefore never verified that those frozen prefixes were present in the
    current exact-state cache namespace. Final strict evaluation could then fail
    with a CACHE MISS even though the B2 sequence file itself was complete.

    Resume semantics:
      * existing frozen sequences are NEVER re-ranked;
      * every frozen prefix is reconstructed deterministically and looked up;
      * missing exact states are computed with the canonical full-network
        evaluator and inserted into the immutable cache namespace;
      * only scenarios without a frozen sequence invoke the B2 surrogate
        ranking rule, after which their selected sequence is frozen immediately.
    """
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    manifest=paths.eval/"parallel_exact_cache_precompute_v1_0_47.json"
    if not manifest.exists():
        raise RuntimeError(f"Missing completed cache manifest {manifest}")
    pm=json.loads(manifest.read_text(encoding="utf-8"))
    if int(pm.get("requests",-1)) != 90720:
        raise RuntimeError(
            "B2 cache repair requires the completed 90,720-request exact "
            f"precompute manifest; got requests={pm.get('requests')!r}."
        )
    protocol=str(pm["protocol_hash"])
    cache=ExactEvaluationStateCache(
        paths.eval/"exact_state_cache_v1_0_41.sqlite",protocol
    )

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].copy()
    test=test.sort_values(["split","scenario_id"]).reset_index(drop=True)
    seed=int(active_numerical_seeds(cfg)[0])

    seq_path=paths.eval/"b2_frozen_sequences_v1_0_51.json"
    work_path=paths.eval/"b2_frozen_sequences_v1_0_51_work.json"

    done={}
    sequence_source=None
    if seq_path.exists():
        raw=json.loads(seq_path.read_text(encoding="utf-8"))
        if str(raw.get("protocol_hash")) == protocol:
            done={
                str(k):list(map(int,v))
                for k,v in raw.get("sequences",{}).items()
            }
            sequence_source="final"
    if not done and work_path.exists():
        raw=json.loads(work_path.read_text(encoding="utf-8"))
        if str(raw.get("protocol_hash")) == protocol:
            done={
                str(k):list(map(int,v))
                for k,v in raw.get("sequences",{}).items()
            }
            sequence_source="work"

    expected={str(int(x)) for x in test.scenario_id}
    unknown=set(done)-expected
    if unknown:
        cache.close()
        raise RuntimeError(
            f"Frozen B2 file contains unexpected scenario IDs: "
            f"{sorted(unknown)[:10]}"
        )

    if done:
        log.log(
            f"B2 resume audit: loaded {len(done)}/{len(test)} frozen "
            f"scenario sequence(s) from {sequence_source} file; "
            "all prefixes will be cache-validated before evaluation"
        )

    # Dynamic B2 constructor is instantiated only if some sequences are absent.
    dynamic_pol = None
    if len(done) < len(test):
        dynamic_pol=b2_policy_factory(cfg,engine,static,edges)

    hits=0
    misses=0
    validated_states=0
    started=time.perf_counter()

    for j,r in enumerate(test.itertuples(),start=1):
        sid=int(r.scenario_id)
        sk=str(sid)
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]

        # Existing sequence: preserve it exactly and reconstruct its network states.
        if sk in done:
            frozen_seq=list(map(int,done[sk]))
            if len(frozen_seq) != int(d.shape[0]):
                cache.close()
                raise RuntimeError(
                    f"Frozen B2 sequence length mismatch for scenario {sid}: "
                    f"{len(frozen_seq)} vs {d.shape[0]}"
                )
            pol=frozen_edge_sequence_policy(frozen_seq)
            observed_actions=[]
            prefix_iter=trajectory_exact_state_prefixes(
                cfg,engine,static,edges,d,pol,seed+sid,sid
            )
        else:
            # No prior sequence: construct once with the canonical B2 surrogate.
            observed_actions=[]
            prefix_iter=trajectory_exact_state_prefixes(
                cfg,engine,static,edges,d,dynamic_pol,seed+sid,sid
            )

        for stok,t,prefix,K in prefix_iter:
            prefix=list(map(int,prefix))
            observed_actions=prefix
            validated_states += 1
            cached=cache.get(stok,t,prefix)
            if cached is not None:
                hits += 1
                continue

            # Scientific semantics are identical to final exact evaluation.
            misses += 1
            _,q,_=engine.solve(K)
            A=engine.accessibility_exact(q,K)
            _,La,Le,Lv,Lc=loss_components(
                A,static.A0,static.population,static.vulnerable
            )
            cache.put(stok,t,prefix,La,Le,Lv,Lc)

        if len(observed_actions) != int(d.shape[0]):
            cache.close()
            raise RuntimeError(
                f"B2 sequence reconstruction length mismatch for scenario {sid}: "
                f"{len(observed_actions)} vs {d.shape[0]}"
            )

        if sk in done:
            if list(map(int,observed_actions)) != list(map(int,done[sk])):
                cache.close()
                raise RuntimeError(
                    f"Frozen B2 sequence replay mismatch for scenario {sid}: "
                    f"stored={done[sk]}, replayed={observed_actions}"
                )
        else:
            done[sk]=list(map(int,observed_actions))
            write_json(work_path,{
                "script_version":SCRIPT_VERSION,
                "protocol_hash":protocol,
                "gain_quantization_decimals":8,
                "sequences":done
            })

        if j==1 or j%10==0 or j==len(test):
            log.log(
                f"B2 frozen-cache audit: {j}/{len(test)} scenarios | "
                f"frozen={len(done):,} | states checked={validated_states:,} | "
                f"cache hits={hits:,} | repaired exact states={misses:,} | "
                f"elapsed={(time.perf_counter()-started)/60:.1f}min"
            )

    if len(done)!=len(test) or set(done)!=expected:
        cache.close()
        raise RuntimeError(
            f"Frozen B2 construction incomplete/inconsistent: "
            f"{len(done)}/{len(test)}"
        )

    # Final second pass: zero cache misses are allowed before strict evaluation.
    missing_after=[]
    for r in test.itertuples():
        sid=int(r.scenario_id)
        d=np.load(paths.scenarios/f"scenario_{sid:05d}.npz")["degradation"]
        pol=frozen_edge_sequence_policy(done[str(sid)])
        for stok,t,prefix,K in trajectory_exact_state_prefixes(
            cfg,engine,static,edges,d,pol,seed+sid,sid
        ):
            if cache.get(stok,t,prefix) is None:
                missing_after.append(
                    (sid,int(t),ExactEvaluationStateCache.prefix_key(prefix))
                )
                if len(missing_after)>=10:
                    break
        if len(missing_after)>=10:
            break

    if missing_after:
        cache.close()
        raise RuntimeError(
            "B2 cache repair failed; missing frozen prefixes remain: "
            f"{missing_after}"
        )

    write_json(seq_path,{
        "script_version":SCRIPT_VERSION,
        "protocol_hash":protocol,
        "scenarios":int(len(test)),
        "seed":seed,
        "gain_quantization_decimals":8,
        "construction_accessibility":"candidate_path_surrogate",
        "reported_accessibility":"exact_full_network",
        "policy_or_checkpoint_changes":False,
        "resume_cache_validation":True,
        "cache_states_checked":int(validated_states),
        "cache_states_repaired":int(misses),
        "sequences":done
    })
    # Keep work file synchronized for interruption-safe future resumes.
    write_json(work_path,{
        "script_version":SCRIPT_VERSION,
        "protocol_hash":protocol,
        "gain_quantization_decimals":8,
        "sequences":done
    })

    cache.close()
    log.log(
        f"B2 FROZEN CACHE-COVERAGE PASS: scenarios={len(done)}, "
        f"states checked={validated_states:,}, cache hits={hits:,}, "
        f"repaired exact states={misses:,}; final evaluation will NOT re-rank B2."
    )

def frozen_edge_sequence_policy(actions):
    """Return the frozen edge id at each stage; feasibility is checked by rollout."""
    seq=tuple(map(int,actions))
    def pol(state):
        t=int(state["t"])
        return int(seq[t]) if t<len(seq) else -1
    return pol



def evaluate_all(cfg,paths,log):
    """Exact out-of-sample evaluation with model-level atomic resume.

    Scientific evaluation semantics are unchanged:
      * exact full-network accessibility is used for every reported trajectory;
      * B0 is evaluated over all numerical seeds;
      * B1-B4 use one deterministic numerical seed;
      * B5/A1-A5/B5_PPO use every trained numerical seed;
      * test and structural-holdout scenarios remain separate in inference.

    v1.0.35 changes only execution safety. A completed spec/seed block is
    checkpointed atomically. If interrupted inside a block, that whole block is
    rerun, which preserves the original B0 RNG sequence exactly.
    """
    torch=require_torch()
    engine,static,edges,zones,_=build_reference_engine(cfg,paths,log)
    static.betweenness=load_or_compute_betweenness(cfg,paths,edges,log)
    b3scores=compute_b3_scores(cfg,paths,engine,static,edges,log)
    b4seq=b4_open_loop_search(cfg,paths,engine,static,edges,log)

    # CRITICAL: evaluation is read-only with respect to learned checkpoints.
    # The old ensure_ablation_models() call is intentionally removed here.
    inv=_evaluation_model_inventory(cfg,paths)
    missing=inv.loc[~inv.exists]
    inv.to_csv(paths.eval/"evaluation_model_inventory.csv",index=False)
    if len(missing):
        miss=", ".join(
            f"{r.spec}:seed{int(r.seed)}"
            for r in missing.itertuples()
        )
        raise RuntimeError(
            "Exact evaluation requires all learned checkpoints to exist and "
            "will not train them implicitly. Missing: "+miss
        )

    meta=pd.read_csv(paths.scenarios/"scenario_manifest.csv")
    test=meta[meta.split.isin(["test","test_structural"])].copy()
    test=test.sort_values(["split","scenario_id"]).reset_index(drop=True)
    if len(test)==0:
        raise RuntimeError("No test/test_structural scenarios found.")

    protocol_tag=_evaluation_protocol_tag(cfg,test)
    rows,step_rows,completed=_load_evaluation_work(paths,protocol_tag)

    # v1.0.49: the cache protocol is the immutable protocol actually used
    # by the completed v1.0.47 precompute. Recomputing the SHA from freshly
    # rebuilt CUDA-derived floating arrays can change the hash despite an
    # unchanged scientific protocol. The precompute manifest is therefore the
    # authoritative cache namespace; strict key-level cache-only evaluation
    # below remains the operational compatibility check.
    precompute_manifest=paths.eval/"parallel_exact_cache_precompute_v1_0_47.json"
    if not precompute_manifest.exists():
        raise RuntimeError(
            "Missing completed exact-cache precompute manifest: "
            f"{precompute_manifest}. Refusing to guess a cache protocol."
        )
    pre_meta=json.loads(precompute_manifest.read_text(encoding="utf-8"))
    if int(pre_meta.get("requests",-1))!=90720:
        raise RuntimeError(
            "Exact-cache precompute manifest is incomplete/unexpected: "
            f"requests={pre_meta.get('requests')!r}, expected=90720."
        )
    unique_seen_states = int(pre_meta.get("unique_seen_states", -1))
    if unique_seen_states <= 0:
        raise RuntimeError(
            "Exact-cache precompute manifest has no valid unique-state count: "
            f"{pre_meta.get('unique_seen_states')!r}."
        )
    # The 43,814-state count is the reference-hardware realization. Fresh
    # training on another supported device may produce different action prefixes;
    # strict cache-key coverage below is the operational correctness gate.
    if unique_seen_states != 43814:
        log.log(
            "NOTE: exact-cache unique-state count differs from the reference "
            f"realization (43,814): current={unique_seen_states:,}. "
            "This is allowed only because every requested final state remains "
            "subject to strict cache-key coverage."
        )
    if str(pre_meta.get("reported_accessibility"))!="exact_full_network":
        raise RuntimeError("Precompute manifest is not exact_full_network.")
    if bool(pre_meta.get("policy_or_checkpoint_changes",True)):
        raise RuntimeError("Precompute manifest reports policy/checkpoint changes.")
    if list(map(int,pre_meta.get("b4_rank_plan",[])))!=[2,19,7,28,3,1]:
        raise RuntimeError("Precompute manifest does not contain frozen audited B4.")
    exact_cache_protocol=str(pre_meta["protocol_hash"])
    rebuilt_protocol=_exact_cache_protocol(cfg,engine,static)
    if rebuilt_protocol!=exact_cache_protocol:
        log.log(
            "NOTE: freshly rebuilt floating-state protocol hash differs from "
            "the completed precompute namespace; using the immutable manifest "
            f"protocol={exact_cache_protocol}. Strict cache-key misses remain fatal."
        )
    exact_cache=ExactEvaluationStateCache(
        paths.eval/"exact_state_cache_v1_0_41.sqlite",
        exact_cache_protocol
    )
    cache_rows=int(exact_cache.conn.execute(
        "SELECT COUNT(*) FROM exact_state_loss WHERE protocol=?",
        (exact_cache_protocol,)
    ).fetchone()[0])
    if cache_rows<=0:
        exact_cache.close()
        raise RuntimeError(
            "Strict final evaluation found zero exact-cache rows for protocol "
            f"{exact_cache_protocol}."
        )
    log.log(
        f"STRICT CACHE-ONLY evaluation: protocol={exact_cache_protocol}, "
        f"available exact states={cache_rows:,}; exact fallback DISABLED"
    )
    if completed:
        log.log(
            f"Resume exact evaluation: {len(completed)} completed model blocks, "
            f"{len(rows)} trajectories already checkpointed"
        )

    b2_frozen_path=paths.eval/"b2_frozen_sequences_v1_0_51.json"
    if not b2_frozen_path.exists():
        exact_cache.close()
        raise RuntimeError(
            "Missing frozen B2 sequences. Run first: "
            "--stage b2-freeze-precompute"
        )
    b2_frozen=json.loads(b2_frozen_path.read_text(encoding="utf-8"))
    if str(b2_frozen.get("protocol_hash"))!=exact_cache_protocol:
        exact_cache.close()
        raise RuntimeError("Frozen B2 protocol does not match exact-cache protocol.")
    if int(b2_frozen.get("scenarios",-1))!=len(test):
        exact_cache.close()
        raise RuntimeError("Frozen B2 scenario count does not match evaluation design.")
    b2_sequences={
        str(k):list(map(int,v))
        for k,v in b2_frozen.get("sequences",{}).items()
    }

    factories={
        "B0":lambda seed:b0_policy_factory(seed),
        "B1":lambda seed:b1_policy_factory(static.betweenness),
        "B2":lambda seed:None,  # frozen per-scenario sequence loaded below
        "B3":lambda seed:b3_policy_factory(b3scores),
        "B4":lambda seed:b4_policy_factory(b4seq),
    }
    eval_seeds=active_numerical_seeds(cfg)
    eval_every=max(
        1,int(cfg.get("progress",{}).get("evaluation_scenario_every",10))
    )

    # Fixed policies. Checkpoint only after a complete spec/seed block.
    fixed_total=10+4
    fixed_done=sum(
        key.startswith("fixed|") for key in completed
    )
    fixed_started=time.perf_counter()
    for spec,fac in factories.items():
        seeds=eval_seeds if spec=="B0" else [eval_seeds[0]]
        for seed in seeds:
            key=f"fixed|{spec}|{int(seed)}"
            if key in completed:
                log.log(f"Skip completed exact block {key}")
                continue
            pol=None if spec=="B2" else fac(seed)
            block_rows=[]
            block_steps=[]
            model_started=time.perf_counter()
            test_rows=list(test.itertuples())
            log.log(
                f"Evaluate exact fixed specification {spec} seed={seed}: "
                f"{len(test_rows)} scenario(s)"
            )
            for scen_pos,r in enumerate(test_rows,start=1):
                scen_t0=time.perf_counter()
                sid=int(r.scenario_id)
                d=np.load(
                    paths.scenarios/f"scenario_{sid:05d}.npz"
                )["degradation"]
                scen_pol=pol
                if spec=="B2":
                    if str(sid) not in b2_sequences:
                        raise RuntimeError(
                            f"Frozen B2 sequence missing scenario {sid}"
                        )
                    scen_pol=frozen_edge_sequence_policy(b2_sequences[str(sid)])
                z,q,steps=trajectory_rollout_cache_only(
                    cfg,engine,static,edges,d,scen_pol,seed+sid,
                    sid,exact_cache,True
                )
                block_rows.append({
                    "spec":spec,"seed":seed,"scenario_id":sid,
                    "split":r.split,"Z":z,"Q":q
                })
                for sr in steps:
                    block_steps.append({
                        "spec":spec,"seed":seed,
                        "scenario_id":sid,**sr
                    })
                if (
                    scen_pos==1 or scen_pos%eval_every==0
                    or scen_pos==len(test_rows)
                ):
                    log.log(_progress_message(
                        f"Exact {spec} seed={seed}",
                        scen_pos,len(test_rows),model_started,
                        time.perf_counter()-scen_t0
                    ))
            # Commit atomically at model boundary.
            rows.extend(block_rows)
            step_rows.extend(block_steps)
            completed.add(key)
            _save_evaluation_work(
                paths,rows,step_rows,completed,protocol_tag
            )
            fixed_done+=1
            log.log(_progress_message(
                "Exact fixed-policy model blocks",
                fixed_done,fixed_total,fixed_started,
                time.perf_counter()-model_started
            ))

    # Learned policies.
    learned_specs=["B5","A1","A2","A3","A4","A5"]
    if bool(cfg.get("graph_ppo",{}).get("enabled",False)):
        learned_specs.append("B5_PPO")
    learned_total=len(learned_specs)*len(eval_seeds)
    learned_done=sum(
        key.startswith("learned|") for key in completed
    )
    learned_started=time.perf_counter()

    for spec in learned_specs:
        log.log(
            f"Evaluate learned specification {spec}: "
            f"{len(eval_seeds)} seed model(s)"
        )
        for seed in eval_seeds:
            key=f"learned|{spec}|{int(seed)}"
            if key in completed:
                log.log(f"Skip completed exact block {key}")
                continue

            model_t0=time.perf_counter()
            if spec=="B5_PPO":
                model,st=load_graph_model(
                    cfg,paths,engine,edges,zones,seed,spec="B5_PPO"
                )
            else:
                model,st=load_graph_model(
                    cfg,paths,engine,edges,zones,seed,
                    None if spec=="B5" else spec
                )

            block_rows=[]
            block_steps=[]
            test_rows=list(test.itertuples())
            for scen_pos,r in enumerate(test_rows,start=1):
                scen_t0=time.perf_counter()
                sid=int(r.scenario_id)
                d=np.load(
                    paths.scenarios/f"scenario_{sid:05d}.npz"
                )["degradation"]
                # New policy object per scenario is intentional and matches
                # v1.0.34. It is especially required for A5's frozen cache.
                pol=deterministic_graph_policy(
                    cfg,model,st,static,engine.device,
                    frozen=(spec=="A5")
                )
                z,q,steps=trajectory_rollout_cache_only(
                    cfg,engine,static,edges,d,pol,seed+sid,
                    sid,exact_cache,True
                )
                block_rows.append({
                    "spec":spec,"seed":seed,"scenario_id":sid,
                    "split":r.split,"Z":z,"Q":q
                })
                for sr in steps:
                    block_steps.append({
                        "spec":spec,"seed":seed,
                        "scenario_id":sid,**sr
                    })
                if (
                    scen_pos==1 or scen_pos%eval_every==0
                    or scen_pos==len(test_rows)
                ):
                    log.log(_progress_message(
                        f"Exact {spec} seed={seed}",
                        scen_pos,len(test_rows),model_t0,
                        time.perf_counter()-scen_t0
                    ))

            rows.extend(block_rows)
            step_rows.extend(block_steps)
            completed.add(key)
            _save_evaluation_work(
                paths,rows,step_rows,completed,protocol_tag
            )
            learned_done+=1
            log.log(_progress_message(
                "Exact learned-policy model blocks",
                learned_done,learned_total,learned_started,
                time.perf_counter()-model_t0
            ))
            # Release per-seed GPU model before loading the next one.
            del model,st
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if exact_cache.misses!=0 or exact_cache.compute_seconds!=0.0:
        raise RuntimeError(
            f"Strict cache-only invariant failed: misses={exact_cache.misses}, "
            f"exact_compute_seconds={exact_cache.compute_seconds:.6f}"
        )
    log.log(
        f"STRICT CACHE-ONLY PASS: hits={exact_cache.hits:,}, misses=0, "
        "exact recomputation=0.0s"
    )

    res=pd.DataFrame(rows)
    steps=pd.DataFrame(step_rows)

    # Integrity checks before publication-facing files replace prior outputs.
    expected_fixed=(10+4)*len(test)
    expected_learned=learned_total*len(test)
    expected=expected_fixed+expected_learned
    if len(res)!=expected:
        raise RuntimeError(
            f"Exact evaluation integrity failure: rows={len(res)}, "
            f"expected={expected}."
        )
    dup=res.duplicated(["spec","seed","scenario_id"]).sum()
    if dup:
        raise RuntimeError(
            f"Exact evaluation contains {int(dup)} duplicate trajectory keys."
        )
    if not np.isfinite(res[["Z","Q"]].to_numpy(float)).all():
        raise RuntimeError("Non-finite Z/Q in exact evaluation results.")

    res.to_parquet(
        paths.eval/"policy_trajectory_results.parquet",index=False
    )
    steps.to_parquet(
        paths.eval/"policy_step_results.parquet",index=False
    )

    alpha=cfg["risk"]["cvar_alpha"]
    le=cfg["risk"]["lambda_equity"]
    lr=cfg["risk"]["lambda_tail"]
    summ=[]
    for (spec,seed,split),g in res.groupby(
        ["spec","seed","split"]
    ):
        cv=cvar_empirical(g.Z.to_numpy(),alpha)
        J=g.Z.mean()+le*g.Q.mean()+lr*cv
        summ.append({
            "spec":spec,"seed":seed,"split":split,
            "mean_Z":g.Z.mean(),"mean_Q":g.Q.mean(),
            "CVaR_Z":cv,"J":J,"n":len(g)
        })
    summary=pd.DataFrame(summ)
    summary.to_csv(
        paths.eval/"policy_summary_by_seed.csv",index=False
    )

    export_results(
        cfg,paths,res,steps,summary,edges,b3scores,log
    )
    _paired_policy_inference(cfg,paths,res,log)

    # Preserve work files as an auditable completion record rather than delete
    # them. The final parquet/CSV outputs above are publication-facing.
    write_json(paths.eval/"exact_evaluation_completion.json",{
        "script_version":SCRIPT_VERSION,
        "protocol_tag":protocol_tag,
        "evaluation_accessibility":"exact_full_network",
        "n_trajectory_rows":int(len(res)),
        "n_step_rows":int(len(steps)),
        "n_completed_model_blocks":int(len(completed)),
        "completed_model_blocks":sorted(completed),
        "integrity_duplicate_trajectory_keys":int(dup),
        "integrity_all_ZQ_finite":True,
        "exact_state_cache":{
            "requests":int(exact_cache.requests),
            "hits":int(exact_cache.hits),
            "misses":int(exact_cache.misses),
            "hit_rate":float(exact_cache.hit_rate),
            "exact_compute_seconds_on_misses":float(
                exact_cache.compute_seconds
            ),
            "protocol":exact_cache_protocol,
        },
    })
    log.log(
        f"Exact-state cache during evaluation: "
        f"hit_rate={exact_cache.hit_rate:.1%}, "
        f"hits={exact_cache.hits}, misses={exact_cache.misses}"
    )
    exact_cache.close()
    log.log(
        f"Exact evaluation COMPLETE: trajectories={len(res)}, "
        f"steps={len(steps)}, model blocks={len(completed)}"
    )
    return res,summary


def export_results(cfg,paths,res,steps,summary,edges,b3scores,log):
    # v1.0.52: canonical risk coefficients are local to export/inference.
    # This fixes post-processing scope only; trajectories and checkpoints are unchanged.
    le=float(cfg["risk"]["lambda_equity"])
    lr=float(cfg["risk"]["lambda_tail"])
    alpha=float(cfg["risk"]["cvar_alpha"])
    # Table 3: out-of-sample B0-B5; average first within seed then summarize algorithmic seeds.
    main=summary[(summary.split=="test") & (summary.spec.str.match(r"B[0-5]"))]
    tab3=main.groupby("spec").agg(mean_Z=("mean_Z","mean"),mean_Q=("mean_Q","mean"),CVaR_Z=("CVaR_Z","mean"),J=("J","mean"),J_seed_sd=("J","std"),n_seeds=("seed","nunique")).reset_index()
    usage=steps[steps.spec.str.match(r"B[0-5]")].groupby("spec").agg(interventions=("action",lambda s:(s>=0).mean()),mean_remaining_budget=("budget","mean")).reset_index();tab3=tab3.merge(usage,on="spec",how="left");tab3.to_csv(paths.table_data/"table3_policy_performance.csv",index=False)
    # Table 4 ablations relative B5.
    base=summary[(summary.spec=="B5")&(summary.split=="test")][["seed","mean_Z","mean_Q","CVaR_Z","J"]].set_index("seed")
    ar=[]
    for a in ["A1","A2","A3","A4","A5"]:
        aa=summary[(summary.spec==a)&(summary.split=="test")].set_index("seed")
        common=base.index.intersection(aa.index);d=aa.loc[common,["mean_Z","mean_Q","CVaR_Z","J"]]-base.loc[common,["mean_Z","mean_Q","CVaR_Z","J"]]
        ar.append({"spec":a,"delta_Z":d.mean_Z.mean(),"delta_Q":d.mean_Q.mean(),"delta_CVaR":d.CVaR_Z.mean(),"delta_J":d.J.mean(),"seed_IQR_delta_J":d.J.quantile(.75)-d.J.quantile(.25)})
    pd.DataFrame(ar).to_csv(paths.table_data/"table4_ablation_results.csv",index=False)
    # Table 5 structural holdout B5 vs B4.
    structural=[]
    for split in ["test","test_structural"]:
        b5=res[(res.spec=="B5")&(res.split==split)].groupby("scenario_id").agg(Z=("Z","mean"),Q=("Q","mean"))
        b4=res[(res.spec=="B4")&(res.split==split)].groupby("scenario_id").agg(Z=("Z","mean"),Q=("Q","mean"))
        ids=b5.index.intersection(b4.index);dZ=b4.loc[ids,"Z"]-b5.loc[ids,"Z"];dQ=b4.loc[ids,"Q"]-b5.loc[ids,"Q"]
        cvdiff=cvar_empirical(b4.loc[ids,"Z"].to_numpy(),cfg["risk"]["cvar_alpha"])-cvar_empirical(b5.loc[ids,"Z"].to_numpy(),cfg["risk"]["cvar_alpha"])
        structural.append({"holdout":split,"n":len(ids),"delta_Z_B4_minus_B5":dZ.mean(),"delta_Q_B4_minus_B5":dQ.mean(),"delta_CVaR_B4_minus_B5":cvdiff,"fraction_B5_lower_Z":float((dZ>0).mean())})
    pd.DataFrame(structural).to_csv(paths.table_data/"table5_structural_holdout.csv",index=False)
    # Figure 5 data.
    tab3.to_csv(paths.figure_data/"fig5_policy_frontier.csv",index=False)
    # Figure 6 seed/holdout data.
    summary[summary.spec.str.match(r"B[0-5]")].to_csv(paths.figure_data/"fig6_holdout_seed_robustness.csv",index=False)
    # Decision-policy descriptive data for 7.6 supplement.
    steps[steps.spec=="B5"].to_parquet(paths.figure_data/"policy_decision_steps_B5.parquet",index=False)
    # Figure 4 data: betweenness vs B3 CAR score.
    b=pd.read_csv(paths.processed/"edge_betweenness.csv");df=b.copy();df["S_CAR"]=b3scores[df.edge_id.astype(int)];df.to_csv(paths.figure_data/"fig4_betweenness_car.csv",index=False)
    # Statistical comparisons B5 vs B0-B4 using scenario-level seed-averaged outcomes.
    nboot=int(cfg["statistics"]["bootstrap_resamples"]);conf=float(cfg["statistics"]["confidence_level"]);seed=int(active_numerical_seeds(cfg)[0])
    tests=[]
    b5=res[(res.spec=="B5")&(res.split=="test")].groupby("scenario_id").agg(Z=("Z","mean"),Q=("Q","mean"))
    for k in ["B0","B1","B2","B3","B4"]:
        bk=res[(res.spec==k)&(res.split=="test")].groupby("scenario_id").agg(Z=("Z","mean"),Q=("Q","mean"));ids=b5.index.intersection(bk.index)
        dz=paired_bootstrap_diff(bk.loc[ids,"Z"],b5.loc[ids,"Z"],nboot,conf,seed);dq=paired_bootstrap_diff(bk.loc[ids,"Q"],b5.loc[ids,"Q"],nboot,conf,seed+1);dc=bootstrap_cvar_diff(bk.loc[ids,"Z"],b5.loc[ids,"Z"],nboot,alpha,conf,seed+2)
        tests.append({"comparison":f"B5_vs_{k}","delta_Z_{k}_minus_B5":dz[0],"Z_ci_lo":dz[1],"Z_ci_hi":dz[2],"delta_Q_{k}_minus_B5":dq[0],"Q_ci_lo":dq[1],"Q_ci_hi":dq[2],"delta_CVaR_{k}_minus_B5":dc[0],"CVaR_ci_lo":dc[1],"CVaR_ci_hi":dc[2]})
    pd.DataFrame(tests).to_csv(paths.eval/"paired_bootstrap_policy_tests.csv",index=False)

    # Algorithmic robustness: B5 episodic PG versus same graph policy optimized by PPO.
    if "B5_PPO" in set(res.spec):
        pg=res[(res.spec=="B5")&(res.split=="test")].groupby("scenario_id").agg(Z=("Z","mean"),Q=("Q","mean"))
        pp=res[(res.spec=="B5_PPO")&(res.split=="test")].groupby("scenario_id").agg(Z=("Z","mean"),Q=("Q","mean"))
        ids=pg.index.intersection(pp.index)
        dz=paired_bootstrap_diff(pp.loc[ids,"Z"],pg.loc[ids,"Z"],nboot,conf,seed+20)
        dq=paired_bootstrap_diff(pp.loc[ids,"Q"],pg.loc[ids,"Q"],nboot,conf,seed+21)
        dc=bootstrap_cvar_diff(pp.loc[ids,"Z"],pg.loc[ids,"Z"],nboot,alpha,conf,seed+22)
        # Full J is recomputed inside each paired bootstrap resample because CVaR is nonlinear.
        rng=np.random.default_rng(seed+23); n=len(ids); jb=np.empty(nboot,float)
        zpp=pp.loc[ids,"Z"].to_numpy();qpp=pp.loc[ids,"Q"].to_numpy()
        zpg=pg.loc[ids,"Z"].to_numpy();qpg=pg.loc[ids,"Q"].to_numpy()
        def _J(z,q):
            return float(np.mean(z)+le*np.mean(q)+lr*cvar_empirical(z,alpha))
        for bi in range(nboot):
            ii=rng.integers(0,n,size=n)
            jb[bi]=_J(zpp[ii],qpp[ii])-_J(zpg[ii],qpg[ii])
        j_est=_J(zpp,qpp)-_J(zpg,qpg)
        j_lo,j_hi=np.quantile(jb,[(1-conf)/2,1-(1-conf)/2])
        pd.DataFrame([{
            "comparison":"B5_PPO_minus_B5_PG","n_scenarios":n,
            "delta_Z":dz[0],"Z_ci_lo":dz[1],"Z_ci_hi":dz[2],
            "delta_Q":dq[0],"Q_ci_lo":dq[1],"Q_ci_hi":dq[2],
            "delta_CVaR":dc[0],"CVaR_ci_lo":dc[1],"CVaR_ci_hi":dc[2],
            "delta_J":j_est,"J_ci_lo":float(j_lo),"J_ci_hi":float(j_hi)
        }]).to_csv(paths.eval/"algorithmic_robustness_PG_vs_PPO.csv",index=False)
        summary[(summary.spec.isin(["B5","B5_PPO"]))].to_csv(
            paths.figure_data/"graph_optimizer_robustness_PG_vs_PPO.csv",index=False
        )
    log.log("Exported figure/table source data, paired bootstrap comparisons, and PG/PPO robustness")


# -----------------------------------------------------------------------------
# Tables 1--2 and six publication figures
# -----------------------------------------------------------------------------

def _publication_dirs(paths):
    """Versioned publication assets; never mix them with legacy v1.0.52 outputs."""
    fd=paths.figure_data/"q1pp_v1_1_0"
    td=paths.table_data/"q1pp_v1_1_0"
    fg=paths.figures/"main_v1_1_0"
    tb=paths.tables/"main_v1_1_0"
    for p in (fd,td,fg,tb): p.mkdir(parents=True,exist_ok=True)
    return fd,td,fg,tb


def _publication_require(path: Path,label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}. Complete --stage evaluate first.")
    return path


def _publication_load_exact(cfg,paths):
    """Load the frozen exact-evaluation outputs used by publication assets.

    v1.0.55 aligns the publication reader with the filenames and schema written
    by evaluate_all(): exact_evaluation_completion.json,
    policy_trajectory_results.*, and policy_step_results.*.
    Step-level ``split`` is recovered from the authoritative trajectory mapping
    because the final step parquet intentionally stores no redundant split.
    This function is read-only with respect to evaluation results.
    """
    comp_path = paths.eval / "exact_evaluation_completion.json"
    traj_parq = paths.eval / "policy_trajectory_results.parquet"
    traj_csv = paths.eval / "policy_trajectory_results.csv"
    step_parq = paths.eval / "policy_step_results.parquet"
    step_csv = paths.eval / "policy_step_results.csv"

    if not comp_path.exists():
        raise RuntimeError(
            f"Missing completed exact-evaluation manifest: {comp_path}. "
            "The publication stage requires the successful final --stage evaluate run."
        )
    comp = read_json(comp_path)

    expected = {
        "n_trajectory_rows": 15120,
        "n_step_rows": 90720,
        "n_completed_model_blocks": 84,
    }
    for key,val in expected.items():
        got=int(comp.get(key,-1))
        if got != val:
            raise RuntimeError(f"Exact-evaluation completion gate failed: {key}={got}, expected {val}.")
    if str(comp.get("evaluation_accessibility","")) != "exact_full_network":
        raise RuntimeError("Publication assets require evaluation_accessibility='exact_full_network'.")
    if int(comp.get("integrity_duplicate_trajectory_keys",-1)) != 0:
        raise RuntimeError("Exact-evaluation completion manifest reports duplicate trajectory keys.")
    if comp.get("integrity_all_ZQ_finite") is not True:
        raise RuntimeError("Exact-evaluation completion manifest does not certify finite Z/Q.")

    def _load(parq_path,csv_path,label):
        if parq_path.exists():
            return pd.read_parquet(parq_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise RuntimeError(
            f"Missing {label}: neither {parq_path.name} nor {csv_path.name} exists."
        )

    res=_load(traj_parq,traj_csv,"trajectory-level exact evaluation")
    steps=_load(step_parq,step_csv,"step-level exact evaluation")

    # Canonical schema actually written by evaluate_all().
    need_r={"scenario_id","spec","seed","split","Z","Q"}
    if not need_r.issubset(res.columns):
        raise RuntimeError(f"Trajectory columns missing: {sorted(need_r-set(res.columns))}")

    need_step_base={"scenario_id","spec","seed","t","action"}
    if not need_step_base.issubset(steps.columns):
        raise RuntimeError(f"Step columns missing: {sorted(need_step_base-set(steps.columns))}")

    if "split" not in steps.columns:
        scenario_split=res[["scenario_id","split"]].drop_duplicates().copy()
        counts=scenario_split.groupby("scenario_id")["split"].nunique(dropna=False)
        bad=counts[counts != 1]
        if not bad.empty:
            raise RuntimeError(
                "Non-unique scenario_id -> split mapping in exact trajectory results: "
                f"{bad.index[:10].tolist()}"
            )
        steps=steps.merge(
            scenario_split,on="scenario_id",how="left",validate="many_to_one"
        )
        if steps["split"].isna().any():
            missing=(steps.loc[steps["split"].isna(),"scenario_id"]
                     .drop_duplicates().head(10).tolist())
            raise RuntimeError(f"Unable to recover split for step scenario_id(s): {missing}")

    if len(res) != expected["n_trajectory_rows"]:
        raise RuntimeError(f"Trajectory row-count mismatch: {len(res)} != 15120")
    if len(steps) != expected["n_step_rows"]:
        raise RuntimeError(f"Step row-count mismatch: {len(steps)} != 90720")

    dup=int(res.duplicated(["spec","seed","scenario_id"]).sum())
    if dup:
        raise RuntimeError(f"Publication reader found {dup} duplicate trajectory keys.")
    if not np.isfinite(res[["Z","Q"]].to_numpy(float)).all():
        raise RuntimeError("Non-finite Z/Q in exact trajectory results.")

    return comp,res,steps

def _publication_scenario(res,spec,split):
    g=res[(res.spec==spec)&(res.split==split)].groupby("scenario_id",as_index=True).agg(Z=("Z","mean"),Q=("Q","mean"))
    return g.sort_index()


def _publication_J(z,q,cfg):
    z=np.asarray(z,float);q=np.asarray(q,float)
    return float(np.mean(z)+float(cfg["risk"]["lambda_equity"])*np.mean(q)+float(cfg["risk"]["lambda_tail"])*cvar_empirical(z,float(cfg["risk"]["cvar_alpha"])))


def _publication_pair(cfg,res,reference,candidate,split,offset=0):
    """Paired scenario inference. Positive reference-minus-candidate favors candidate."""
    a=_publication_scenario(res,reference,split); b=_publication_scenario(res,candidate,split); ids=a.index.intersection(b.index)
    if len(ids)==0: raise RuntimeError(f"No paired scenarios: {reference}, {candidate}, {split}")
    a=a.loc[ids]; b=b.loc[ids]
    nboot=int(cfg["statistics"]["bootstrap_resamples"]); conf=float(cfg["statistics"]["confidence_level"])
    nperm=int(cfg["statistics"].get("signflip_resamples",100000)); base=int(active_numerical_seeds(cfg)[0])+int(offset)
    dz=paired_bootstrap_diff(a.Z,b.Z,nboot,conf,base+1); dq=paired_bootstrap_diff(a.Q,b.Q,nboot,conf,base+2)
    dc=bootstrap_cvar_diff(a.Z,b.Z,nboot,float(cfg["risk"]["cvar_alpha"]),conf,base+3)
    dj=paired_full_criterion_bootstrap(a,b,cfg,nboot,conf,base+4)
    rawz=(a.Z-b.Z).to_numpy(float)
    return {
        "split":split,"reference":reference,"candidate":candidate,"comparison":f"{reference}_minus_{candidate}","n_scenarios":int(len(ids)),
        "delta_Z":float(dz[0]),"Z_ci_lo":float(dz[1]),"Z_ci_hi":float(dz[2]),
        "delta_Q":float(dq[0]),"Q_ci_lo":float(dq[1]),"Q_ci_hi":float(dq[2]),
        "delta_CVaR":float(dc[0]),"CVaR_ci_lo":float(dc[1]),"CVaR_ci_hi":float(dc[2]),
        "delta_J":float(dj[0]),"J_ci_lo":float(dj[1]),"J_ci_hi":float(dj[2]),
        "p_signflip_Z":float(paired_signflip_pvalue(rawz,nperm,base+5)),
        "p_wilcoxon_Z":float(paired_wilcoxon_pvalue(rawz)),
        "fraction_candidate_lower_Z":float(np.mean(rawz>0)),
    }


def _publication_action_agreement(steps,other,split="test"):
    cols=["seed","scenario_id","split","t"]
    a=steps[(steps.spec=="B5")&(steps.split==split)][cols+["action"]].rename(columns={"action":"a_B5"})
    b=steps[(steps.spec==other)&(steps.split==split)][cols+["action"]].rename(columns={"action":"a_other"})
    m=a.merge(b,on=cols,how="inner",validate="one_to_one")
    if m.empty:return {"step_action_agreement":np.nan,"trajectory_action_agreement":np.nan,"n_paired_steps":0,"n_paired_trajectories":0}
    m["same"]=m.a_B5.astype(int)==m.a_other.astype(int)
    tr=m.groupby(["seed","scenario_id","split"],as_index=False).same.all()
    return {"step_action_agreement":float(m.same.mean()),"trajectory_action_agreement":float(tr.same.mean()),"n_paired_steps":int(len(m)),"n_paired_trajectories":int(len(tr))}


def _publication_write_spatial_csv(paths,fd):
    """CSV-only spatial sources for Fig. 1, with an explicit reduction audit.

    Population and vulnerability are mapped on the full populated Filosofi grid,
    not only on the 1,800-unit computational sample. Model-sample membership is
    exported separately through ``in_model_sample``. This prevents the map from
    visually presenting the population-weighted computational reduction as if it
    were the underlying socioeconomic partition.
    """
    gpd,_,_=require_geospatial()
    edges=gpd.read_parquet(_publication_require(paths.processed/"network_edges.parquet","processed road edges"))
    full=gpd.read_file(_publication_require(paths.processed/"spatial_units_filosofi_2021.geojson","full Filosofi spatial units"))
    model=gpd.read_file(_publication_require(paths.processed/"spatial_units_model_snapped.geojson","model zones"))
    serv=gpd.read_file(_publication_require(paths.processed/"essential_health_services_snapped.geojson","health services"))

    # Deterministic cartographic thinning of road edges only.
    nmax=18000
    if len(edges)>nmax:
        idx=np.linspace(0,len(edges)-1,nmax,dtype=int)
        ep=edges.iloc[idx].copy()
    else:
        ep=edges.copy()
    pd.DataFrame({
        "edge_id":pd.to_numeric(ep.get("edge_id",pd.Series(np.arange(len(ep)))),errors="coerce"),
        "geometry_wkt":ep.geometry.to_wkt()
    }).to_csv(fd/"fig1_network_edges.csv",index=False)

    # source_zone_id in the reduced model refers to the pre-reduction full-grid zone_id.
    sampled=set()
    if "source_zone_id" in model.columns:
        sampled=set(pd.to_numeric(model["source_zone_id"],errors="coerce").dropna().astype(int).tolist())
    elif "zone_id" in model.columns and len(model)==len(full):
        sampled=set(pd.to_numeric(model["zone_id"],errors="coerce").dropna().astype(int).tolist())

    zcols=[c for c in ["zone_id","P_i","is_vulnerable"] if c in full.columns]
    z=pd.DataFrame(full[zcols].copy())
    if "zone_id" in z.columns:
        zid=pd.to_numeric(z["zone_id"],errors="coerce")
        z["in_model_sample"]=zid.isin(sampled) if sampled else False
    else:
        z["in_model_sample"]=False
    z["geometry_wkt"]=full.geometry.to_wkt()
    z.to_csv(fd/"fig1_zones.csv",index=False)

    scols=[c for c in ["service_id","name","O_j"] if c in serv.columns]
    q=pd.DataFrame(serv[scols].copy())
    q["geometry_wkt"]=serv.geometry.to_wkt()
    q.to_csv(fd/"fig1_services.csv",index=False)

    # Publication-only audit. This is diagnostic metadata, not a sixth main table.
    def _stats(g,label):
        P=pd.to_numeric(g["P_i"],errors="coerce").fillna(0.0) if "P_i" in g.columns else pd.Series(np.zeros(len(g)))
        V=g["is_vulnerable"].astype(bool) if "is_vulnerable" in g.columns else pd.Series(False,index=g.index)
        pop=float(P.sum())
        vpop=float(P[V].sum())
        return {
            "population_set":label,
            "n_units":int(len(g)),
            "n_vulnerable_units":int(V.sum()),
            "vulnerable_unit_share":float(V.mean()) if len(g) else np.nan,
            "population":pop,
            "vulnerable_population":vpop,
            "vulnerable_population_share":vpop/pop if pop>0 else np.nan,
        }
    audit=pd.DataFrame([_stats(full,"full_populated_filosofi_grid"),_stats(model,"computational_model_sample")])
    full_pop=float(audit.loc[audit.population_set=="full_populated_filosofi_grid","population"].iloc[0])
    model_pop=float(audit.loc[audit.population_set=="computational_model_sample","population"].iloc[0])
    audit["model_population_coverage_of_full"]=model_pop/full_pop if full_pop>0 else np.nan
    audit_path=paths.manifests/"publication_spatial_reduction_audit_v1_1_0.csv"
    audit.to_csv(audit_path,index=False)
    return audit

def build_publication_data(cfg,paths,log):
    """Create all publication table/figure source CSVs from frozen exact outputs."""
    fd,td,_,_=_publication_dirs(paths); comp,res,steps=_publication_load_exact(cfg,paths)
    seeds=active_numerical_seeds(cfg)
    # ---------- Table 1: urban system + design ----------
    gpd,_,_=require_geospatial()
    zones_full=gpd.read_file(paths.processed/"spatial_units_filosofi_2021.geojson")
    zones=gpd.read_file(paths.processed/"spatial_units_model_snapped.geojson")
    serv=gpd.read_file(paths.processed/"essential_health_services_snapped.geojson")
    edges=gpd.read_parquet(paths.processed/"network_edges.parquet")
    pairs_path=paths.processed/"zone_service_pairs.parquet"
    path_json=paths.processed/"route_choice_paths.json"
    n_od=int(len(pd.read_parquet(pairs_path))) if pairs_path.exists() else np.nan
    n_paths=int(sum(len(x) for x in read_json(path_json).get("paths",[]))) if path_json.exists() else np.nan
    t1=[
        ("Urban system","Study area","Métropole Européenne de Lille"),
        ("Urban system","Populated Filosofi cells",len(zones_full)),
        ("Urban system","Computational model zones",len(zones)),
        ("Urban system","Essential-health service opportunities",len(serv)),
        ("Urban system","Directed road edges",len(edges)),
        ("Routing","OD pairs",n_od),("Routing","Candidate paths",n_paths),("Routing","Paths per OD",cfg["network"]["candidate_paths_per_od"]),
        ("Disruptions","Training scenarios",cfg["disruptions"]["n_train"]),("Disruptions","Validation scenarios",cfg["disruptions"]["n_validation"]),("Disruptions","Ordinary test scenarios",cfg["disruptions"]["n_test"]),("Disruptions","Structural holdout scenarios",cfg["disruptions"]["n_structural_test"]),("Disruptions","Decision horizon (T+1)",int(cfg["disruptions"]["horizon_T"])+1),
        ("Intervention","Candidate rule","R3: absolute capacity deficit × fixed candidate-path incidence"),("Intervention","Maximum candidates per state",cfg["intervention"]["max_candidate_edges_per_step"]),("Intervention","Restoration fraction",cfg["intervention"]["restoration_fraction_of_remaining_deficit"]),
        ("Risk","Discount factor gamma",cfg["accessibility"]["gamma"]),("Risk","Equity loading lambda_E",cfg["risk"]["lambda_equity"]),("Risk","Tail loading lambda_R",cfg["risk"]["lambda_tail"]),("Risk","CVaR level alpha",cfg["risk"]["cvar_alpha"]),
        ("Evaluation","Numerical seeds",len(seeds)),("Evaluation","Reported accessibility","Exact full network"),("Evaluation","Exact trajectories",comp["n_trajectory_rows"]),("Evaluation","Exact decision steps",comp["n_step_rows"]),
    ]
    pd.DataFrame(t1,columns=["Block","Item","Value"]).to_csv(td/"table1_open_data_empirical_design.csv",index=False)
    # ---------- Table 2: specifications ----------
    rows=[
        ["B0","Random feasible intervention","Yes","No","No","No","No","Random benchmark"],
        ["B1","Directed betweenness ranking","Yes","No","No","No","No","Topology benchmark"],
        ["B2","Myopic accessibility-loss rule","Yes","No","No","No","No","Candidate-path evaluator for action selection; exact final outcomes"],
        ["B3","Training-only surrogate CAR ranking","Yes","No","Ranking","No","No","No test-outcome information in ranking"],
        ["B4","Open-loop rank plan","Fixed rank plan","No","No","No","No","Rank mapped through state-dependent admissible R3 correspondence"],
        ["B5","Graph policy; episodic policy gradient","Yes","Yes","Yes","Yes","Yes","Primary learned specification"],
        ["A1","B5 without message passing","Yes","No","Yes","Yes","Yes","Controlled representation ablation"],
        ["A2","B5 without CAR node feature","Yes","Yes","No","Yes","Yes","Controlled feature ablation"],
        ["A3","B5 without equity loading","Yes","Yes","Yes","No","Yes","Controlled objective ablation"],
        ["A4","B5 without tail-risk loading","Yes","Yes","Yes","Yes","No","Controlled objective ablation"],
        ["A5","B5 with frozen sequential scores","No score updating","Yes","Yes","Yes","Yes","Controlled sequential-adaptation ablation"],
        ["B5-PPO","Same graph policy; PPO-style optimizer","Yes","Yes","Yes","Yes","Yes","Algorithmic robustness only"],
    ]
    pd.DataFrame(rows,columns=["Spec","Decision rule","Sequential updating","Graph message passing","CAR feature/use","Equity term","Tail term","Role"]).to_csv(td/"table2_policy_specifications.csv",index=False)
    # ---------- Table 3: absolute exact performance by regime ----------
    specs=["B0","B1","B2","B3","B4","B5"]
    perf=[]
    for split in ["test","test_structural"]:
        for sp in specs:
            g=_publication_scenario(res,sp,split)
            if g.empty: continue
            seedJ=[]
            for _,sg in res[(res.spec==sp)&(res.split==split)].groupby("seed"):
                seedJ.append(_publication_J(sg.Z.to_numpy(),sg.Q.to_numpy(),cfg))
            perf.append({"split":split,"spec":sp,"n_scenarios":len(g),"mean_Z":float(g.Z.mean()),"mean_Q":float(g.Q.mean()),"CVaR_Z":float(cvar_empirical(g.Z.to_numpy(),float(cfg["risk"]["cvar_alpha"]))),"J":_publication_J(g.Z.to_numpy(),g.Q.to_numpy(),cfg),"J_seed_sd":float(np.std(seedJ,ddof=1)) if len(seedJ)>1 else np.nan,"n_seeds":len(seedJ)})
    t3=pd.DataFrame(perf); t3.to_csv(td/"table3_exact_performance_by_regime.csv",index=False)
    # ---------- Table 4: paired inference ----------
    inf=[]
    refs=["B0","B1","B2","B3","B4"]
    # Identical paired protocol in both held-out regimes.
    # Delta = benchmark - B5, hence Delta_J > 0 favors B5.
    for split,offset in [("test",0),("test_structural",1000)]:
        block_inf=[]
        for i,k in enumerate(refs):
            block_inf.append(_publication_pair(cfg,res,k,"B5",split,offset+100*i))
        # Holm correction is applied separately within each prespecified family
        # of five benchmark comparisons, avoiding cross-regime multiplicity mixing.
        sf=holm_adjust([r["p_signflip_Z"] for r in block_inf])
        wi=holm_adjust([r["p_wilcoxon_Z"] for r in block_inf])
        for j,r in enumerate(block_inf):
            r["p_signflip_Z_holm"]=float(sf[j])
            r["p_wilcoxon_Z_holm"]=float(wi[j])
        inf.extend(block_inf)
    t4=pd.DataFrame(inf)
    required_inference_cols={
        "split","reference","candidate","delta_Z","Z_ci_lo","Z_ci_hi",
        "delta_Q","Q_ci_lo","Q_ci_hi","delta_CVaR","CVaR_ci_lo","CVaR_ci_hi",
        "delta_J","J_ci_lo","J_ci_hi","p_signflip_Z","p_wilcoxon_Z"
    }
    missing_inference=required_inference_cols-set(t4.columns)
    if missing_inference:
        raise RuntimeError(
            f"Paired-inference schema mismatch: missing {sorted(missing_inference)}"
        )
    if len(t4)!=10:
        raise RuntimeError(f"Expected 10 paired-inference rows, got {len(t4)}.")
    for split in ["test","test_structural"]:
        got=set(t4.loc[t4.split==split,"reference"].astype(str))
        if got!=set(refs):
            raise RuntimeError(f"Incomplete paired inference for {split}: {sorted(got)}")
    t4["direction_J"]=np.where(t4["delta_J"]>0,"B5 lower J",
                               np.where(t4["delta_J"]<0,"benchmark lower J","tie"))
    t4.to_csv(td/"table4_paired_inference.csv",index=False)
    b2s=t4[(t4.split=="test_structural")&(t4.reference=="B2")].iloc[0]
    log.log(
        "Structural B2-B5 paired inference: "
        f"DeltaJ={float(b2s['delta_J']):.8g}, "
        f"95% bootstrap CI=[{float(b2s['J_ci_lo']):.8g}, {float(b2s['J_ci_hi']):.8g}]; "
        f"DeltaZ={float(b2s['delta_Z']):.8g}, "
        f"Z 95% CI=[{float(b2s['Z_ci_lo']):.8g}, {float(b2s['Z_ci_hi']):.8g}], "
        f"Z sign-flip p={float(b2s['p_signflip_Z']):.6g}, "
        f"Z Wilcoxon p={float(b2s['p_wilcoxon_Z']):.6g}"
    )
    # ---------- Table 5: ablations + optimizer robustness ----------
    rob=[]
    for i,sp in enumerate(["A1","A2","A3","A4","A5","B5_PPO"]):
        r=_publication_pair(cfg,res,sp,"B5","test",1200+100*i)  # other - B5: positive means B5 lower risk
        ag=_publication_action_agreement(steps,sp); r.update(ag); r["spec"]=sp; r["type"]="optimizer" if sp=="B5_PPO" else "ablation"; rob.append(r)
    t5=pd.DataFrame(rob); t5.to_csv(td/"table5_ablation_optimizer_robustness.csv",index=False)
    # ---------- Figure CSVs ----------
    spatial_audit=_publication_write_spatial_csv(paths,fd)
    fa=spatial_audit.loc[spatial_audit.population_set=="full_populated_filosofi_grid"].iloc[0]
    ma=spatial_audit.loc[spatial_audit.population_set=="computational_model_sample"].iloc[0]
    log.log(
        "Publication spatial audit: "
        f"full n={int(fa.n_units):,}, vulnerable-unit share={float(fa.vulnerable_unit_share):.3f}, "
        f"population={float(fa.population):,.1f}; "
        f"model n={int(ma.n_units):,}, vulnerable-unit share={float(ma.vulnerable_unit_share):.3f}, "
        f"population={float(ma.population):,.1f}, "
        f"population coverage={float(ma.model_population_coverage_of_full):.3f}"
    )
    # Fig. 2: signed frozen-load diagnostic, copied and enriched only from existing source data.
    old=_publication_require(paths.figure_data/"fig3_cascade_decomposition.csv","frozen-load cascade source")
    f2=pd.read_csv(old)
    zfull=pd.to_numeric(f2["Z_FULL"],errors="coerce").to_numpy(float)
    delta=pd.to_numeric(f2["Delta_flow"],errors="coerce").to_numpy(float)
    f2["flow_share"]=np.where(np.abs(zfull)>1e-12,delta/zfull,np.nan)
    tol=1e-12
    f2["flow_response_sign"]=np.where(delta>tol,"amplification",np.where(delta<-tol,"attenuation","approximately_zero"))
    f2.to_csv(fd/"fig2_cascade_diagnostic.csv",index=False)
    finite=np.isfinite(delta)
    if finite.any():
        log.log(
            "Signed frozen-load audit: "
            f"n={int(finite.sum())}, amplification share={float(np.mean(delta[finite]>tol)):.4f}, "
            f"attenuation share={float(np.mean(delta[finite]<-tol)):.4f}, "
            f"median Delta_flow={float(np.median(delta[finite])):.6g}"
        )
    # Fig. 3: forest-plot source is exactly Table 4, duplicated deliberately for figure independence.
    t4.to_csv(fd/"fig3_policy_regime_forest.csv",index=False)
    # Fig. 4: scenario-level B4-B5 heterogeneity; one row per held-out scenario.
    f4=[]
    for split in ["test","test_structural"]:
        a=_publication_scenario(res,"B4",split); b=_publication_scenario(res,"B5",split); ids=a.index.intersection(b.index)
        for sid in ids:
            f4.append({"split":split,"scenario_id":sid,"delta_Z_B4_minus_B5":float(a.loc[sid,"Z"]-b.loc[sid,"Z"]),"delta_Q_B4_minus_B5":float(a.loc[sid,"Q"]-b.loc[sid,"Q"])})
    pd.DataFrame(f4).to_csv(fd/"fig4_scenario_value_sequential_adaptation.csv",index=False)
    # Fig. 5: ablation + optimizer outcome/decision robustness.
    t5.to_csv(fd/"fig5_ablation_optimizer.csv",index=False)
    # Human/machine-readable provenance manifest for all publication CSVs.
    csvs=sorted(list(fd.glob("*.csv"))+list(td.glob("*.csv")))
    write_json(paths.manifests/"publication_assets_v1_1_0.json",{"script_version":SCRIPT_VERSION,"evaluation_accessibility":"exact_full_network","source_completion_manifest":str(paths.eval/"exact_evaluation_completion.json"),"figure_csvs":[{"path":str(p.relative_to(paths.root)),"sha256":sha256_file(p),"rows":int(len(pd.read_csv(p)))} for p in sorted(fd.glob("*.csv"))],"table_csvs":[{"path":str(p.relative_to(paths.root)),"sha256":sha256_file(p),"rows":int(len(pd.read_csv(p)))} for p in sorted(td.glob("*.csv"))],"diagnostic_csvs":[{"path":str((paths.manifests/"publication_spatial_reduction_audit_v1_1_0.csv").relative_to(paths.root)),"sha256":sha256_file(paths.manifests/"publication_spatial_reduction_audit_v1_1_0.csv"),"rows":int(len(pd.read_csv(paths.manifests/"publication_spatial_reduction_audit_v1_1_0.csv")))}]})
    log.log(f"Publication-data v1.1.0 COMPLETE: {len(list(fd.glob('*.csv')))} figure CSVs, {len(list(td.glob('*.csv')))} table CSVs; no evaluation rerun")


def _publication_savefig(fig,pathbase):
    fig.savefig(pathbase.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(pathbase.with_suffix(".png"),dpi=600,bbox_inches="tight")


def make_figures(cfg,paths,log):
    """Render exactly five main figures, strictly from saved CSV source data."""
    import matplotlib.pyplot as plt
    fd,_,fg,_=_publication_dirs(paths)
    required=["fig1_network_edges.csv","fig1_zones.csv","fig1_services.csv","fig2_cascade_diagnostic.csv","fig3_policy_regime_forest.csv","fig4_scenario_value_sequential_adaptation.csv","fig5_ablation_optimizer.csv"]
    missing=[x for x in required if not (fd/x).exists()]
    if missing: raise RuntimeError(f"Missing v1.1.0 figure CSVs {missing}; run --stage publication-data once. Figures never rebuild analytical data.")
    # Fig. 1: full socioeconomic support, computational sample, and services.
    gpd,_,_=require_geospatial(); from shapely import wkt
    e=pd.read_csv(fd/"fig1_network_edges.csv"); z=pd.read_csv(fd/"fig1_zones.csv"); q=pd.read_csv(fd/"fig1_services.csv")
    eg=gpd.GeoDataFrame(e,geometry=e.geometry_wkt.map(wkt.loads),crs=cfg["project"]["crs_metric"])
    zg=gpd.GeoDataFrame(z,geometry=z.geometry_wkt.map(wkt.loads),crs=cfg["project"]["crs_metric"])
    qg=gpd.GeoDataFrame(q,geometry=q.geometry_wkt.map(wkt.loads),crs=cfg["project"]["crs_metric"])
    fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    eg.plot(ax=axs[0],linewidth=.32,alpha=.78)
    qg.plot(ax=axs[0],markersize=24)
    axs[0].set_title("A. Road network and essential-health services")
    if "P_i" in zg:
        zg.plot(column="P_i",ax=axs[1],markersize=2.8,legend=True)
    else:
        zg.plot(ax=axs[1],markersize=2.8)
    axs[1].set_title("B. Population on the full Filosofi support")
    if "is_vulnerable" in zg:
        zg.plot(column="is_vulnerable",ax=axs[2],markersize=2.8,categorical=True,legend=True)
        leg=axs[2].get_legend()
        if leg is not None:
            texts=leg.get_texts()
            if len(texts)>=2:
                texts[0].set_text("Non-vulnerable")
                texts[1].set_text("Vulnerable")
    else:
        zg.plot(ax=axs[2],markersize=2.8)
    axs[2].set_title("C. Prespecified vulnerability partition")
    for ax in axs: ax.set_axis_off()
    fig.suptitle("Urban service-access system in Métropole Européenne de Lille",y=1.01)
    fig.tight_layout()
    _publication_savefig(fig,fg/"fig1_urban_service_access_system")
    plt.close(fig)
    # Fig. 2: signed frozen-load diagnostic; no amplification assumption is imposed.
    d=pd.read_csv(fd/"fig2_cascade_diagnostic.csv").replace([np.inf,-np.inf],np.nan)
    fig,axs=plt.subplots(1,3,figsize=(14.8,4.5))
    axs[0].scatter(d.Z_FL,d.Z_FULL,s=13,alpha=.62)
    lo=float(np.nanmin([d.Z_FL.min(),d.Z_FULL.min()]))
    hi=float(np.nanmax([d.Z_FL.max(),d.Z_FULL.max()]))
    axs[0].plot([lo,hi],[lo,hi],ls="--",linewidth=1)
    axs[0].set(xlabel=r"$Z^{FL}$",ylabel=r"$Z^{FULL}$",title="A. Full versus frozen-load loss")

    x=np.sort(pd.to_numeric(d.Delta_flow,errors="coerce").dropna().to_numpy(float))
    rank=100.0*(np.arange(len(x))+0.5)/max(len(x),1)
    axs[1].plot(rank,x,linewidth=1.4)
    axs[1].axhline(0,ls="--",linewidth=1)
    axs[1].set(
        xlabel="Scenario percentile rank",
        ylabel=r"$\Delta_{\mathrm{flow}}=Z^{FULL}-Z^{FL}$",
        title="B. Ordered signed flow-response contribution"
    )

    # Preserve sign and expose regime heterogeneity rather than absolute magnitudes.
    if "split" in d.columns:
        labels=[]
        data=[]
        for sp,label in [("train","Train"),("validation","Validation"),("test","Ordinary test"),("test_structural","Structural")]:
            vals=pd.to_numeric(d.loc[d["split"]==sp,"Delta_flow"],errors="coerce").dropna().to_numpy(float)
            if len(vals):
                data.append(vals); labels.append(label)
        if data:
            axs[2].boxplot(data,labels=labels,showfliers=False)
            axs[2].axhline(0,ls="--",linewidth=1)
            axs[2].tick_params(axis="x",rotation=25)
            axs[2].set(ylabel=r"$\Delta_{\mathrm{flow}}$",title="C. Contribution across disruption regimes")
        else:
            rr=d.flow_share.dropna().to_numpy(float)
            axs[2].hist(rr,bins=28); axs[2].axvline(0,ls="--",linewidth=1)
            axs[2].set(xlabel=r"$\Delta_{\mathrm{flow}}/Z^{FULL}$",title="C. Signed relative contribution")
    else:
        rr=d.flow_share.dropna().to_numpy(float)
        axs[2].hist(rr,bins=28); axs[2].axvline(0,ls="--",linewidth=1)
        axs[2].set(xlabel=r"$\Delta_{\mathrm{flow}}/Z^{FULL}$",title="C. Signed relative contribution")

    fig.suptitle("Endogenous network response can amplify or attenuate disruption losses",y=1.01)
    fig.tight_layout()
    _publication_savefig(fig,fg/"fig2_signed_flow_response_diagnostic")
    plt.close(fig)
    # Fig. 3: paired forest plot, ordinary vs structural.
    f=pd.read_csv(fd/"fig3_policy_regime_forest.csv"); fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),sharex=False)
    panels=[("test","A. Ordinary held-out disruptions"),("test_structural","B. Structural out-of-distribution holdout")]
    for ax,(split,title) in zip(axs,panels):
        g=f[f.split==split].copy(); labels=[x.replace("_minus_B5","") for x in g.comparison]; y=np.arange(len(g)); ax.errorbar(g.delta_J,y,xerr=np.vstack([g.delta_J-g.J_ci_lo,g.J_ci_hi-g.delta_J]),fmt="o",capsize=3); ax.axvline(0,ls="--",linewidth=1); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel(r"$\Delta J=J_{benchmark}-J_{B5}$"); ax.set_title(title)
    fig.suptitle("Held-out performance of B5 relative to canonical benchmarks",y=1.01); fig.tight_layout(); _publication_savefig(fig,fg/"fig3_policy_performance_by_regime"); plt.close(fig)
    # Fig. 4: paired scenario heterogeneity as ECDFs, zero has direct interpretation.
    f=pd.read_csv(fd/"fig4_scenario_value_sequential_adaptation.csv"); fig,ax=plt.subplots(figsize=(7.4,5.2))
    for split,label in [("test","Ordinary test"),("test_structural","Structural holdout")]:
        x=np.sort(f.loc[f.split==split,"delta_Z_B4_minus_B5"].to_numpy(float)); y=np.arange(1,len(x)+1)/len(x); ax.step(x,y,where="post",label=label)
    ax.axvline(0,ls="--",linewidth=1); ax.set(xlabel=r"Scenario-level $Z_{B4}-Z_{B5}$",ylabel="Empirical cumulative probability",title="Scenario-level B5 advantage relative to the open-loop B4 benchmark"); ax.legend(); fig.tight_layout(); _publication_savefig(fig,fg/"fig4_scenario_level_sequential_value"); plt.close(fig)
    # Fig. 5: outcome effect and action agreement for ablations/optimizer.
    f=pd.read_csv(fd/"fig5_ablation_optimizer.csv")
    order=[x for x in ["A1","A2","A3","A4","A5","B5_PPO"] if x in set(f.spec)]
    f=f.set_index("spec").loc[order].reset_index()
    display_labels=[x.replace("_","-") for x in f.spec.astype(str)]
    y=np.arange(len(f))
    fig,axs=plt.subplots(1,2,figsize=(12.2,5.0))
    axs[0].errorbar(f.delta_J,y,xerr=np.vstack([f.delta_J-f.J_ci_lo,f.J_ci_hi-f.delta_J]),fmt="o",capsize=3)
    axs[0].axvline(0,ls="--",linewidth=1)
    axs[0].set_yticks(y,display_labels)
    axs[0].invert_yaxis()
    axs[0].set(xlabel=r"$J_{variant}-J_{B5}$",title="A. Exact held-out outcome difference")
    axs[1].barh(y,f.step_action_agreement)
    axs[1].set_yticks(y,display_labels)
    axs[1].invert_yaxis()
    axs[1].set_xlim(0,1)
    axs[1].set(xlabel="Step-level action agreement with B5",title="B. Decision agreement")
    fig.suptitle("Controlled ablations and optimizer robustness",y=1.01); fig.tight_layout(); _publication_savefig(fig,fg/"fig5_ablation_optimizer_robustness"); plt.close(fig)
    log.log("Created exactly five Q1++ main figures (PDF + 600-dpi PNG) strictly from v1.1.0 CSV sources")


def make_tables(cfg,paths,log):
    """Export exactly five main tables from saved v1.1.0 table CSV sources."""
    _,td,_,tb=_publication_dirs(paths)
    names=["table1_open_data_empirical_design.csv","table2_policy_specifications.csv","table3_exact_performance_by_regime.csv","table4_paired_inference.csv","table5_ablation_optimizer_robustness.csv"]
    missing=[n for n in names if not (td/n).exists()]
    if missing: raise RuntimeError(f"Missing v1.1.0 table CSVs {missing}; run --stage publication-data once.")
    for name in names:
        src=td/name; df=pd.read_csv(src); shutil.copy2(src,tb/name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); tex=df.to_latex(index=False,escape=True,float_format=lambda x:f"{x:.6g}")
        (tb/(Path(name).stem+".tex")).write_text(tex,encoding="utf-8")
    log.log("Created exactly five Q1++ main tables (CSV + LaTeX) from saved v1.1.0 CSV sources")


# -----------------------------------------------------------------------------
# Run manifest, CLI
# -----------------------------------------------------------------------------

def write_run_manifest(cfg,paths,stage,started,log):
    files=[]
    for folder in [paths.processed,paths.scenarios,paths.models,paths.eval,paths.figure_data,paths.table_data]:
        for p in folder.glob("*"):
            if p.is_file() and p.stat().st_size<2_000_000_000:
                with contextlib.suppress(Exception): files.append({"path":str(p.relative_to(paths.root)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    torch_info={}
    with contextlib.suppress(Exception):
        torch=require_torch();torch_info={"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    man={"script_version":SCRIPT_VERSION,"config_schema":CONFIG_SCHEMA_VERSION,"stage":stage,"started_utc":started,"finished_utc":utc_now(),"config_sha256":sha256_file(paths.config) if paths.config.exists() else None,"python":sys.version,"platform":platform.platform(),"packages":package_versions(),"torch":torch_info,"outputs":files}
    write_json(paths.manifests/f"run_{stage}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",man)


def init_config(paths: Paths, force: bool=False):
    paths.mkdirs()
    if paths.config.exists() and not force:
        print(f"Config already exists: {paths.config}")
        return
    cfg=json.loads(json.dumps(DEFAULT_CONFIG));cfg["project"]["retrieval_date"]=dt.date.today().isoformat();write_json(paths.config,cfg);print(f"Wrote {paths.config}")


def load_cfg(paths: Paths, config_override: Optional[Path]) -> Dict[str,Any]:
    if not paths.config.exists(): init_config(paths)
    cfg=deep_update(DEFAULT_CONFIG,read_json(paths.config))
    if config_override:
        cfg=deep_update(cfg,read_json(config_override))
    if len(cfg["compute"]["seeds"]) < int(cfg["compute"].get("numerical_seed_limit",10)):
        warnings.warn("Configuration seed pool is smaller than compute.numerical_seed_limit.")
    return cfg



def apply_training_protocol_lock(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze the final R3/Top-160 learning/evaluation protocol after all config merging."""
    cfg=json.loads(json.dumps(cfg))
    cfg["compute"]["seeds"]=list(CANONICAL_SEEDS)
    cfg["compute"]["numerical_seed_limit"]=10
    # Final action correspondence locked after v1.0.27--v1.0.29 audits.
    cfg["intervention"]["max_candidate_edges_per_step"]=160
    cfg["intervention"]["candidate_screening_rule"]="r3_abs_deficit_x_candidate_path_use"
    cfg["intervention"]["candidate_deficit_tolerance"]=1e-12
    gp=cfg["graph_policy"]
    gp.update({"epochs":30,"episodes_per_epoch":6,"validation_every":5,"early_stopping_patience":3,"validation_scenarios_training":8,"training_accessibility":"candidate_path_surrogate","final_evaluation_accessibility":"exact_full_network","surrogate_audit_scenarios":4,"profile_first_epoch":True,"runtime_guard_seconds_per_epoch":60.0})
    pp=cfg["graph_ppo"]
    pp.update({"epochs":30,"episodes_per_epoch":6,"validation_every":5,"early_stopping_patience":3,"validation_scenarios_training":8})
    return cfg



def _repro_core_processed_files(paths: Paths) -> List[Path]:
    return [
        paths.manifests / "data_manifest.json",
        paths.processed / "mel_drive_network_projected.graphml",
        paths.processed / "network_nodes.parquet",
        paths.processed / "network_edges.parquet",
        paths.processed / "spatial_units_filosofi_2021.geojson",
        paths.processed / "spatial_units_model_snapped.geojson",
        paths.processed / "essential_health_services_snapped.geojson",
        paths.processed / "edge_key_to_id.json",
    ]


def runtime_reproducibility_preflight(
    cfg: Dict[str, Any],
    paths: Paths,
    log: Logger,
    strict_reference_env: bool = False,
) -> Dict[str, Any]:
    """Validate the frozen numerical design and record the runtime environment."""
    seeds = active_numerical_seeds(cfg)
    if seeds != CANONICAL_SEEDS:
        raise RuntimeError(f"Canonical seed check failed: {seeds}")

    torch = require_torch()
    cuda_available = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    checks = {
        "seeds": {"expected": CANONICAL_SEEDS, "actual": seeds, "pass": seeds == CANONICAL_SEEDS},
        "torch": {
            "expected": REFERENCE_TORCH_VERSION,
            "actual": str(torch.__version__),
            "pass": str(torch.__version__) == REFERENCE_TORCH_VERSION,
        },
        "cuda": {
            "expected": REFERENCE_CUDA_AVAILABLE,
            "actual": cuda_available,
            "pass": cuda_available == REFERENCE_CUDA_AVAILABLE,
        },
        "gpu": {
            "expected": REFERENCE_GPU_NAME,
            "actual": gpu,
            "pass": gpu == REFERENCE_GPU_NAME,
        },
    }

    log.log(f"REPRO CHECK seeds: {'PASS' if checks['seeds']['pass'] else 'FAIL'} | {seeds}")
    log.log(
        "REPRO CHECK runtime: "
        f"torch={torch.__version__} ({'PASS' if checks['torch']['pass'] else 'DIFF'}), "
        f"CUDA={cuda_available} ({'PASS' if checks['cuda']['pass'] else 'DIFF'}), "
        f"GPU={gpu} ({'PASS' if checks['gpu']['pass'] else 'DIFF'})"
    )
    if strict_reference_env and not all(v["pass"] for v in checks.values()):
        raise RuntimeError(
            "Strict reference-environment check failed. Re-run without "
            "--strict-reference-env for portable reproduction, or use the "
            "reference PyTorch/CUDA/GPU environment."
        )

    env = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "checks": checks,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch_cuda_version": torch.version.cuda,
    }
    write_json(paths.manifests / "reproducibility_environment_v1_1_0.json", env)
    return env


def ensure_reproducibility_inputs(cfg: Dict[str, Any], paths: Paths, log: Logger) -> None:
    """Use validated local processed data when present; download/process only if needed."""
    core = _repro_core_processed_files(paths)
    missing = [p for p in core if not p.exists()]
    if not missing:
        log.log("DATA CHECK PASS: all core processed public-data artifacts are available locally")
        G = load_graphml_compat(paths.processed / "mel_drive_network_projected.graphml", log)
        edges = pd.read_parquet(paths.processed / "network_edges.parquet")
        if G.number_of_edges() != len(edges):
            raise RuntimeError(
                "Processed network cache failed edge-count consistency check: "
                f"GraphML={G.number_of_edges():,}, GeoParquet={len(edges):,}."
            )
        log.log(
            "GRAPH CHECK PASS: projected GraphML is available and compatible; "
            "network reconstruction/download is skipped"
        )
        return

    log.log(
        "DATA CHECK: processed cache incomplete; missing "
        + ", ".join(str(p.relative_to(paths.root)) for p in missing)
    )
    log.log("DATA ACQUISITION START: check cached raw sources, download missing public data, then process")
    download_public_data(cfg, paths, log)
    process_all(cfg, paths, log)

    still_missing = [p for p in core if not p.exists()]
    if still_missing:
        raise RuntimeError(
            "Data/process stage completed but required artifacts are still missing: "
            + ", ".join(str(p) for p in still_missing)
        )
    log.log("DATA ACQUISITION/PROCESSING COMPLETE")


def ensure_reproducibility_scenarios(cfg: Dict[str, Any], paths: Paths, log: Logger) -> None:
    """Validate the frozen 420-scenario split; regenerate deterministically if incomplete."""
    manifest = paths.scenarios / "scenario_manifest.csv"
    expected = (
        int(cfg["disruptions"]["n_train"])
        + int(cfg["disruptions"]["n_validation"])
        + int(cfg["disruptions"]["n_test"])
        + int(cfg["disruptions"]["n_structural_test"])
    )
    valid = False
    if manifest.exists():
        try:
            meta = pd.read_csv(manifest)
            split_counts = meta["split"].value_counts().to_dict()
            valid = (
                len(meta) == expected
                and int(split_counts.get("train", 0)) == int(cfg["disruptions"]["n_train"])
                and int(split_counts.get("validation", 0)) == int(cfg["disruptions"]["n_validation"])
                and int(split_counts.get("test", 0)) == int(cfg["disruptions"]["n_test"])
                and int(split_counts.get("test_structural", 0)) == int(cfg["disruptions"]["n_structural_test"])
                and all((paths.scenarios / f"scenario_{int(sid):05d}.npz").exists()
                        for sid in meta["scenario_id"])
            )
        except Exception:
            valid = False

    if valid:
        log.log(
            "SCENARIO CHECK PASS: reuse frozen deterministic disruption design "
            f"({expected} trajectories)"
        )
    else:
        log.log("SCENARIO CHECK: cache absent/incomplete; regenerate deterministic disruption design")
        generate_disruptions(cfg, paths, log)


def _remove_if_exists(path: Path, removed: List[str]) -> None:
    if path.exists():
        path.unlink()
        removed.append(str(path))


def initialize_reproducibility_run(
    cfg: Dict[str, Any],
    paths: Paths,
    log: Logger,
    force_fresh: bool = False,
) -> Path:
    """Start once, then resume safely after interruption without deleting progress."""
    bundle = paths.root / REPRO_BUNDLE_NAME
    bundle.mkdir(parents=True, exist_ok=True)
    state_path = bundle / "run_state.json"

    if state_path.exists() and not force_fresh:
        state = read_json(state_path)
        log.log(
            "REPRO RUN RESUME: existing v1.1.0 run state found; "
            "completed checkpoints/work files will be reused"
        )
        return bundle

    removed: List[str] = []
    # Fresh learned models: B5-PG, B5-PPO and all five trained ablations.
    for pat in (
        "B5_seed_*.pt", "B5_PPO_seed_*.pt",
        "A1_seed_*.pt", "A2_seed_*.pt", "A3_seed_*.pt",
        "A4_seed_*.pt", "A5_seed_*.pt",
    ):
        for fp in paths.models.glob(pat):
            _remove_if_exists(fp, removed)

    _remove_if_exists(paths.models / "final_action_protocol.json", removed)
    # A genuinely fresh reproduction must not inherit a B2 sequence file from
    # another training realization. Resume runs preserve and audit it instead.
    _remove_if_exists(paths.eval / "b2_frozen_sequences_v1_0_51.json", removed)
    _remove_if_exists(paths.eval / "b2_frozen_sequences_v1_0_51_work.json", removed)

    # Fresh training histories and final evaluation rows, while preserving
    # public data, processed graph, deterministic scenarios, baseline products,
    # B4's frozen audited rank plan, and the expensive exact-state cache.
    for name in [
        "training_history_B5.csv",
        "training_history_B5_PPO.csv",
        "training_history_ablations.csv",
        "training_surrogate_audit.csv",
        "training_surrogate_audit_summary.json",
        "policy_trajectory_results_work.csv",
        "policy_step_results_work.csv",
        "policy_evaluation_completed_models.json",
        "policy_trajectory_results.csv",
        "policy_trajectory_results.parquet",
        "policy_step_results.csv",
        "policy_step_results.parquet",
        "exact_evaluation_completion.json",
    ]:
        _remove_if_exists(paths.eval / name, removed)
    for fp in paths.eval.glob("training_profile_*_seed_*.csv"):
        _remove_if_exists(fp, removed)

    # New publication namespace is always rebuilt.
    for d in [
        paths.figure_data / "q1pp_v1_1_0",
        paths.table_data / "q1pp_v1_1_0",
        paths.figures / "main_v1_1_0",
        paths.tables / "main_v1_1_0",
    ]:
        if d.exists():
            shutil.rmtree(d)

    state = {
        "script_version": SCRIPT_VERSION,
        "started_utc": utc_now(),
        "status": "started",
        "fresh_reset": True,
        "removed_files": len(removed),
        "canonical_seeds": CANONICAL_SEEDS,
    }
    write_json(state_path, state)
    log.log(
        f"REPRO RUN START: clean v1.1.0 training/evaluation state initialized; "
        f"removed {len(removed)} prior learned/evaluation artifact(s)"
    )
    return bundle


def snapshot_reproducibility_bundle(
    cfg: Dict[str, Any],
    paths: Paths,
    log: Logger,
    bundle: Path,
) -> None:
    """Collect publication-facing outputs and machine-readable provenance in one folder."""
    bundle.mkdir(parents=True, exist_ok=True)
    targets = {
        "figures": paths.figures / "main_v1_1_0",
        "tables": paths.tables / "main_v1_1_0",
        "figure_data": paths.figure_data / "q1pp_v1_1_0",
        "table_data": paths.table_data / "q1pp_v1_1_0",
    }
    for name, source in targets.items():
        dest = bundle / name
        if dest.exists():
            shutil.rmtree(dest)
        if not source.exists():
            raise RuntimeError(f"Missing publication output directory: {source}")
        shutil.copytree(source, dest)

    (bundle / "logs").mkdir(exist_ok=True)
    if (paths.logs / "pipeline.log").exists():
        shutil.copy2(paths.logs / "pipeline.log", bundle / "logs" / "pipeline.log")

    (bundle / "manifests").mkdir(exist_ok=True)
    for fp in [
        paths.manifests / "reproducibility_environment_v1_1_0.json",
        paths.manifests / "publication_assets_v1_1_0.json",
        paths.manifests / "publication_spatial_reduction_audit_v1_1_0.csv",
        paths.eval / "exact_evaluation_completion.json",
        paths.eval / "evaluation_model_inventory.csv",
    ]:
        if fp.exists():
            shutil.copy2(fp, bundle / "manifests" / fp.name)

    # Checksum inventory for every trained checkpoint, without duplicating model binaries.
    model_rows = []
    for spec in ["B5", "B5_PPO", "A1", "A2", "A3", "A4", "A5"]:
        pat = "B5_PPO_seed_*.pt" if spec == "B5_PPO" else f"{spec}_seed_*.pt"
        for fp in sorted(paths.models.glob(pat)):
            model_rows.append({
                "spec": spec,
                "checkpoint": str(fp.relative_to(paths.root)),
                "bytes": fp.stat().st_size,
                "sha256": sha256_file(fp),
            })
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(bundle / "model_checkpoint_checksums.csv", index=False)

    # Final bundle manifest.
    files = []
    for fp in sorted(bundle.rglob("*")):
        if fp.is_file() and fp.name != "reproducibility_manifest.json":
            files.append({
                "path": str(fp.relative_to(bundle)),
                "bytes": int(fp.stat().st_size),
                "sha256": sha256_file(fp),
            })
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "canonical_seeds": CANONICAL_SEEDS,
        "reference_environment": {
            "torch": REFERENCE_TORCH_VERSION,
            "cuda": REFERENCE_CUDA_AVAILABLE,
            "gpu": REFERENCE_GPU_NAME,
        },
        "n_model_checkpoints": int(len(model_df)),
        "files": files,
    }
    write_json(bundle / "reproducibility_manifest.json", manifest)

    state_path = bundle / "run_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    state.update({"status": "complete", "finished_utc": utc_now()})
    write_json(state_path, state)
    log.log(
        f"REPRO BUNDLE COMPLETE: {bundle} | "
        f"{len(files)} checksummed output/provenance file(s), "
        f"{len(model_df)} trained checkpoint checksum(s)"
    )


def exact_precompute_is_complete(paths: Paths, log: Optional[Logger]=None) -> bool:
    """Return True only for a completed, populated exact-cache precompute."""
    manifest=paths.eval/"parallel_exact_cache_precompute_v1_0_47.json"
    db=paths.eval/"exact_state_cache_v1_0_41.sqlite"
    if not manifest.exists() or not db.exists():
        return False
    try:
        meta=json.loads(manifest.read_text(encoding="utf-8"))
        if int(meta.get("requests",-1)) != 90720:
            return False
        if int(meta.get("unique_seen_states",-1)) <= 0:
            return False
        protocol=str(meta.get("protocol_hash",""))
        if not protocol:
            return False
        conn=sqlite3.connect(str(db))
        try:
            n=int(conn.execute(
                "SELECT COUNT(*) FROM exact_state_loss WHERE protocol=?",
                (protocol,)
            ).fetchone()[0])
        finally:
            conn.close()
        if n <= 0:
            return False
        if log is not None:
            log.log(
                "EXACT PRECOMPUTE CHECK PASS: completed manifest found "
                f"(requests=90,720, unique seen="
                f"{int(meta['unique_seen_states']):,}, cached rows={n:,}); "
                "21-hour precompute will NOT be repeated"
            )
        return True
    except Exception as exc:
        if log is not None:
            log.log(
                f"EXACT PRECOMPUTE CHECK INCOMPLETE: "
                f"{type(exc).__name__}: {exc}"
            )
        return False


def run_reproducibility_workflow(
    cfg: Dict[str, Any],
    paths: Paths,
    log: Logger,
    strict_reference_env: bool = False,
    fresh_run: bool = False,
) -> None:
    """End-to-end, resumable public reproduction workflow for the paper."""
    log.log("=" * 78)
    log.log("CEUS LILLE PUBLIC REPRODUCTION WORKFLOW v1.1.0 START")
    log.log("=" * 78)

    runtime_reproducibility_preflight(cfg, paths, log, strict_reference_env)
    ensure_reproducibility_inputs(cfg, paths, log)
    ensure_reproducibility_scenarios(cfg, paths, log)
    bundle = initialize_reproducibility_run(cfg, paths, log, force_fresh=fresh_run)

    log.log("STAGE 1/7 — BASELINE / REFERENCE ENGINE")
    baseline_stage(cfg, paths, log)

    log.log("STAGE 2/7 — PRIMARY GRAPH-POLICY TRAINING (B5-PG + B5-PPO)")
    train_graph_policies(cfg, paths, log)

    log.log("STAGE 3/7 — CONTROLLED ABLATION TRAINING (A1–A5)")
    train_ablation_models(cfg, paths, log)

    log.log("STAGE 4/7 — EVALUATION BENCHMARKS / EXACT-STATE CACHE")
    engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
    b4_open_loop_search(cfg, paths, engine, static, edges, log)
    if exact_precompute_is_complete(paths, log):
        log.log(
            "Resume path: reuse completed exact-state precompute; "
            "proceed directly to frozen-B2 cache repair/audit"
        )
    else:
        evaluation_exact_cache_parallel_precompute(cfg, paths, log)
        if not exact_precompute_is_complete(paths, log):
            raise RuntimeError(
                "Exact-state precompute returned without a valid completion manifest/cache."
            )
    evaluation_b2_freeze_precompute_v151(cfg, paths, log)

    log.log("STAGE 5/7 — EXACT HELD-OUT EVALUATION")
    evaluation_preflight(cfg, paths, log)
    evaluate_all(cfg, paths, log)

    log.log("STAGE 6/7 — Q1++ PUBLICATION TABLE/FIGURE DATA")
    build_publication_data(cfg, paths, log)

    log.log("STAGE 7/7 — Q1++ FIGURES, TABLES, AND CLEAN REPRODUCTION BUNDLE")
    make_figures(cfg, paths, log)
    make_tables(cfg, paths, log)
    snapshot_reproducibility_bundle(cfg, paths, log, bundle)

    log.log("=" * 78)
    log.log("CEUS LILLE PUBLIC REPRODUCTION WORKFLOW v1.1.0 COMPLETE")
    log.log("=" * 78)


def main(argv=None):
    stages = [
        "reproduce",
        "init-config", "download", "process", "disruptions", "baseline",
        "train", "train-ablations", "evaluate",
        "publication-data", "figures", "tables", "publication-assets",
        # Expert/resume/audit stages retained for transparent long-run recovery.
        "learning-audit", "intervention-audit", "hazard-capacity-audit",
        "candidate-redesign-audit", "candidate-cap-audit",
        "candidate-cap-extended-audit", "final-action-smoke",
        "evaluation-precompute", "evaluation-preflight",
        "evaluation-cache-audit", "evaluation-prefix-audit",
        "action-ablation-audit", "exact-speed-audit", "exact-fast-audit",
        "exact-process-audit", "exact-production-audit",
        "b4-rank-precompute", "b4-rank-audit",
        "evaluation-cache-precompute-parallel", "b2-freeze-precompute",
        "all",
    ]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "CEUS Lille Q1++ public reproducibility pipeline v1.1.0-B\n\n"
            "Recommended reviewer command:\n"
            "  python ceus_lille_q1pp_pipeline_v1_1_0.py --stage reproduce\n\n"
            "The reproduce stage checks local/open data, validates the frozen "
            "10-seed design and runtime environment, reuses the projected GraphML "
            "when available, trains B5/PPO/A1-A5, performs exact held-out evaluation, "
            "and exports five Q1++ figures and five tables into a clean bundle."
        ),
    )
    parser.add_argument(
        "--root", default=DATA_ROOT_NAME,
        help=f"Data/cache root (default: {DATA_ROOT_NAME})"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON override merged before the frozen publication lock"
    )
    parser.add_argument("--stage", choices=stages, default="reproduce")
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument(
        "--fresh-run", action="store_true",
        help=(
            "Restart v1.1.0 learned models and final evaluation from a clean state. "
            "Processed public data, scenarios, baseline products and exact-state cache "
            "are preserved. Without this flag, an interrupted v1.1.0 run resumes."
        ),
    )
    parser.add_argument(
        "--strict-reference-env", action="store_true",
        help=(
            "Require exact reference runtime: torch 2.12.0+cu126, CUDA available, "
            "NVIDIA GeForce RTX 3060 Laptop GPU. Without this flag, differences are "
            "recorded but portable reproduction is allowed."
        ),
    )
    parser.add_argument(
        "--reset-training", action="store_true",
        help="Expert single-stage reset for --stage train only."
    )
    args = parser.parse_args(argv)

    paths = Paths.build(Path(args.root))
    paths.mkdirs()
    if args.stage == "init-config":
        init_config(paths, args.force_config)
        return 0

    cfg = load_cfg(paths, args.config)
    scientific_stages = set(stages) - {"init-config", "download", "process", "disruptions", "baseline"}
    if args.stage in scientific_stages:
        cfg = apply_training_protocol_lock(cfg)

    log = Logger(paths.logs / "pipeline.log")
    started = utc_now()
    log.log(f"Start stage={args.stage} script={SCRIPT_VERSION}")

    # Threading/CUDA setup is recorded, not hidden.
    with contextlib.suppress(Exception):
        torch = require_torch()
        torch.set_num_threads(int(cfg["compute"]["cpu_workers"]))
        torch.set_num_interop_threads(max(1, min(4, int(cfg["compute"]["cpu_workers"]))))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    seeds = active_numerical_seeds(apply_training_protocol_lock(cfg))
    log.log(f"10-seed numerical design: {seeds}")
    with contextlib.suppress(Exception):
        torch = require_torch()
        log.log(
            f"torch={torch.__version__}, CUDA={torch.cuda.is_available()}, "
            f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
        )

    try:
        if args.stage in {"reproduce", "all"}:
            run_reproducibility_workflow(
                apply_training_protocol_lock(cfg),
                paths,
                log,
                strict_reference_env=bool(args.strict_reference_env),
                fresh_run=bool(args.fresh_run),
            )
        elif args.stage == "download":
            download_public_data(cfg, paths, log)
        elif args.stage == "process":
            process_all(cfg, paths, log)
        elif args.stage == "disruptions":
            generate_disruptions(cfg, paths, log)
        elif args.stage == "baseline":
            baseline_stage(cfg, paths, log)
        elif args.stage == "train":
            if args.reset_training:
                removed = []
                for pat in ("B5_seed_*.pt", "B5_PPO_seed_*.pt"):
                    for fp in paths.models.glob(pat):
                        _remove_if_exists(fp, removed)
                _remove_if_exists(paths.models / "final_action_protocol.json", removed)
                log.log(f"Expert training reset: removed {len(removed)} file(s)")
            train_graph_policies(cfg, paths, log)
        elif args.stage == "train-ablations":
            train_ablation_models(cfg, paths, log)
        elif args.stage == "evaluate":
            evaluate_all(cfg, paths, log)
        elif args.stage == "publication-data":
            build_publication_data(cfg, paths, log)
        elif args.stage == "figures":
            make_figures(cfg, paths, log)
        elif args.stage == "tables":
            make_tables(cfg, paths, log)
        elif args.stage == "publication-assets":
            build_publication_data(cfg, paths, log)
            make_figures(cfg, paths, log)
            make_tables(cfg, paths, log)
        elif args.stage == "learning-audit":
            learning_integrity_audit(cfg, paths, log)
        elif args.stage == "intervention-audit":
            intervention_scale_audit(cfg, paths, log)
        elif args.stage == "hazard-capacity-audit":
            hazard_capacity_candidate_audit(cfg, paths, log)
        elif args.stage == "candidate-redesign-audit":
            candidate_action_redesign_audit(cfg, paths, log)
        elif args.stage == "candidate-cap-audit":
            candidate_cap_signal_audit(cfg, paths, log)
        elif args.stage == "candidate-cap-extended-audit":
            candidate_cap_extended_audit(cfg, paths, log)
        elif args.stage == "final-action-smoke":
            final_action_space_smoke_test(cfg, paths, log)
        elif args.stage == "evaluation-precompute":
            precompute_evaluation_benchmarks(cfg, paths, log)
        elif args.stage == "evaluation-preflight":
            evaluation_preflight(cfg, paths, log)
        elif args.stage == "evaluation-cache-audit":
            evaluation_cache_audit(cfg, paths, log)
        elif args.stage == "evaluation-prefix-audit":
            evaluation_prefix_audit(cfg, paths, log)
        elif args.stage == "action-ablation-audit":
            action_ablation_audit(cfg, paths, log)
        elif args.stage == "exact-speed-audit":
            exact_accessibility_speed_audit(cfg, paths, log)
        elif args.stage == "exact-fast-audit":
            exact_accessibility_fast_audit(cfg, paths, log)
        elif args.stage == "exact-process-audit":
            exact_state_process_parallel_audit(cfg, paths, log)
        elif args.stage == "exact-production-audit":
            exact_process_production_audit(cfg, paths, log)
        elif args.stage == "b4-rank-precompute":
            engine, static, edges, zones, _ = build_reference_engine(cfg, paths, log)
            b4_open_loop_search(cfg, paths, engine, static, edges, log)
        elif args.stage == "b4-rank-audit":
            b4_rank_policy_audit(cfg, paths, log)
        elif args.stage == "evaluation-cache-precompute-parallel":
            evaluation_exact_cache_parallel_precompute(cfg, paths, log)
        elif args.stage == "b2-freeze-precompute":
            evaluation_b2_freeze_precompute_v151(cfg, paths, log)
        else:
            raise RuntimeError(f"Unhandled stage: {args.stage}")
    finally:
        write_run_manifest(cfg, paths, args.stage, started, log)

    log.log("DONE")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    # Explicit spawn avoids Windows fork assumptions.
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

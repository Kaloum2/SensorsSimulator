"""
Chargement et fusion de fichiers de configuration (YAML, JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from sensor_noise_sim.pipeline import PipelineConfig


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> dict[str, Any]:
    """Charge un fichier YAML et retourne un dictionnaire."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("La racine YAML doit être un mapping (dictionnaire).")
    return data


def load_json(path: PathLike) -> dict[str, Any]:
    """Charge un fichier JSON."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("La racine JSON doit être un objet.")
    return data


def load_config(path: PathLike) -> dict[str, Any]:
    """
    Charge un fichier selon l'extension : ``.yaml`` / ``.yml`` ou ``.json``.

    Examples
    --------
    >>> cfg = load_config("config.yaml")
    >>> pipe = build_pipeline_from_config(PipelineConfig(**cfg))
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".yaml", ".yml"):
        return load_yaml(p)
    if suf == ".json":
        return load_json(p)
    raise ValueError(f"Extension non supportée: {suf}")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : les clés de ``override`` remplacent ``base``."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_config_with_overrides(
    path: PathLike,
    path_override: Optional[PathLike] = None,
) -> PipelineConfig:
    """
    Charge la configuration principale et une optionnelle ``override`` (même format).

    Retourne un :class:`PipelineConfig` prêt pour :func:`build_pipeline_from_config`.
    """
    main = load_config(path)
    if path_override is not None:
        over = load_config(path_override)
        main = merge_dicts(main, over)
    return PipelineConfig(
        imu=main.get("imu"),
        barometer=main.get("barometer"),
        odometry=main.get("odometry"),
        seed=main.get("seed"),
        odometry_dim_linear=main.get("odometry_dim_linear", 2),
    )

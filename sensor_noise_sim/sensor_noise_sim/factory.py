"""
Construction de pipelines et de couches d'artefacts à partir de dictionnaires (YAML/JSON).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from sensor_noise_sim.artifacts import (
    Dropout,
    DropoutConfig,
    ShockConfig,
    ShockImpulse,
    SpikeConfig,
    SpikeNoise,
    VibrationConfig,
    VibrationOverlay,
)
from sensor_noise_sim.base import NoiseModel
from sensor_noise_sim.pipeline import PipelineConfig, SensorNoisePipeline, build_pipeline_from_config


def _as_post_layer(
    model: NoiseModel,
) -> Callable[[np.ndarray, float, float, Any], np.ndarray]:
    """Transforme un :class:`NoiseModel` en fonction compatible ``post_*`` du pipeline."""

    def layer(x: np.ndarray, dt: float, t: float, state: Any) -> np.ndarray:
        return model.apply(x, dt, state=state, t=t)

    return layer


def _shock_config_from_mapping(m: dict[str, Any]) -> ShockConfig:
    d = dict(m)
    aw = d.get("axis_weights")
    if isinstance(aw, (list, tuple)) and len(aw) == 3:
        d["axis_weights"] = (float(aw[0]), float(aw[1]), float(aw[2]))
    return ShockConfig(**d)


def _vibration_config_from_mapping(m: dict[str, Any]) -> VibrationConfig:
    d = dict(m)
    for key in ("frequencies_hz", "amplitudes_m_s2"):
        v = d.get(key)
        if isinstance(v, (list, tuple)):
            d[key] = tuple(float(x) for x in v)
    ax = d.get("axis")
    if isinstance(ax, (list, tuple)) and len(ax) == 3:
        d["axis"] = (float(ax[0]), float(ax[1]), float(ax[2]))
    return VibrationConfig(**d)


def build_accel_artifact_layers(
    cfg: dict[str, Any],
    *,
    seed: Optional[int] = None,
) -> list[Callable[[np.ndarray, float, float, Any], np.ndarray]]:
    """
    Construit les couches post-accéléromètre à partir d'une section ``artifacts`` de config.

    Clés reconnues :

    - ``shock`` : paramètres :class:`ShockConfig`
    - ``vibration`` : paramètres :class:`VibrationConfig`
    - ``spike`` : ``{ "amplitude_std": float, "probability_per_step": float }``
    - ``dropout`` : ``{ "probability_per_step": float, "mode": "nan"|"hold" }``
    """
    layers: list[Callable[[np.ndarray, float, float, Any], np.ndarray]] = []
    if not cfg:
        return layers

    if "shock" in cfg and cfg["shock"]:
        shock = ShockImpulse(_shock_config_from_mapping(cfg["shock"]), seed=seed)
        layers.append(_as_post_layer(shock))

    if "vibration" in cfg and cfg["vibration"]:
        vib = VibrationOverlay(_vibration_config_from_mapping(cfg["vibration"]), seed=seed)
        layers.append(_as_post_layer(vib))

    if "spike" in cfg and cfg["spike"]:
        raw = cfg["spike"]
        sp_cfg = SpikeConfig(
            probability_per_step=float(raw.get("probability_per_step", 0.0)),
            amplitude_std=float(raw.get("amplitude_std", 1.0)),
        )
        sp = SpikeNoise(sp_cfg, dim=3, seed=seed)
        layers.append(_as_post_layer(sp))

    if "dropout" in cfg and cfg["dropout"]:
        dr = Dropout(DropoutConfig(**cfg["dropout"]), dim=3, seed=seed)
        layers.append(_as_post_layer(dr))

    return layers


def build_gyro_artifact_layers(
    cfg: dict[str, Any],
    *,
    seed: Optional[int] = None,
) -> list[Callable[[np.ndarray, float, float, Any], np.ndarray]]:
    """Comme :func:`build_accel_artifact_layers` mais pour le gyro (chocs/spikes/dropout)."""
    layers: list[Callable[[np.ndarray, float, float, Any], np.ndarray]] = []
    if not cfg:
        return layers
    if "shock" in cfg and cfg["shock"]:
        shock = ShockImpulse(_shock_config_from_mapping(cfg["shock"]), seed=seed)
        layers.append(_as_post_layer(shock))
    if "spike" in cfg and cfg["spike"]:
        raw = cfg["spike"]
        sp_cfg = SpikeConfig(
            probability_per_step=float(raw.get("probability_per_step", 0.0)),
            amplitude_std=float(raw.get("amplitude_std", 1.0)),
        )
        sp = SpikeNoise(sp_cfg, dim=3, seed=seed)
        layers.append(_as_post_layer(sp))
    if "dropout" in cfg and cfg["dropout"]:
        dr = Dropout(DropoutConfig(**cfg["dropout"]), dim=3, seed=seed)
        layers.append(_as_post_layer(dr))
    return layers


def build_sensor_pipeline_from_dict(
    data: dict[str, Any],
) -> SensorNoisePipeline:
    """
    Construit un :class:`SensorNoisePipeline` à partir du dictionnaire racine du fichier YAML.

    Structure attendue :

    .. code-block:: yaml

        seed: 42
        imu: { ... }
        barometer: { ... }
        odometry: { ... }
        odometry_dim_linear: 2
        artifacts:
          accel: { shock: {...}, vibration: {...} }
          gyro: { spike: {...} }
    """
    seed = data.get("seed")
    pc = PipelineConfig(
        imu=data.get("imu"),
        barometer=data.get("barometer"),
        odometry=data.get("odometry"),
        seed=seed,
        odometry_dim_linear=data.get("odometry_dim_linear", 2),
    )
    pipe = build_pipeline_from_config(pc)

    art = data.get("artifacts") or {}
    accel_cfg = art.get("accel") or {}
    gyro_cfg = art.get("gyro") or {}

    pipe.post_imu_accel.extend(build_accel_artifact_layers(accel_cfg, seed=seed))
    pipe.post_gyro.extend(build_gyro_artifact_layers(gyro_cfg, seed=seed))
    return pipe

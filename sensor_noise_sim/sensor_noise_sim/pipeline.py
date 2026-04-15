"""
Pipeline unifié : IMU + baromètre + odométrie + couches d'artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from sensor_noise_sim.barometer import BarometerNoise, BarometerNoiseConfig
from sensor_noise_sim.base import NoiseState
from sensor_noise_sim.imu import IMUNoise, IMUNoiseConfig
from sensor_noise_sim.odometry import OdometryNoise, OdometryNoiseConfig


@dataclass
class PipelineConfig:
    """
    Agrégat de configuration pour :class:`SensorNoisePipeline`.

    Tous les champs sont optionnels : seuls les simulateurs demandés sont construits.
    """

    imu: Optional[dict[str, Any]] = None
    barometer: Optional[dict[str, Any]] = None
    odometry: Optional[dict[str, Any]] = None
    seed: Optional[int] = None
    odometry_dim_linear: int = 2


class SensorNoisePipeline:
    """
    Orchestre les simulateurs de capteurs et met à jour un :class:`NoiseState` temps.

    Ordre d'application pour l'IMU : modèle interne IMU → couches ``post_imu`` (accélé 3D).

    Parameters
    ----------
    imu : IMUNoise, optional
    baro : BarometerNoise, optional
    odo : OdometryNoise, optional
    post_imu_accel : list of callable
        Chaque fonction ``(accel: ndarray, dt, t, state) -> ndarray`` pour chocs, vibrations, etc.
    post_gyro : même principe pour le gyroscope (3D).
    post_baro : ``(pressure: float, dt, t, state) -> float``
    post_odom : ``(v, w, dt, t, state) -> (v, w)`` ou None pour laisser tel quel.

    Examples
    --------
    >>> from sensor_noise_sim import SensorNoisePipeline, IMUNoiseConfig
    >>> pipe = SensorNoisePipeline(imu=IMUNoise(IMUNoiseConfig(gyro_white_noise_std=0.001)))
    >>> g, a = pipe.step_imu(np.zeros(3), np.array([0,0,9.81]), dt=0.01)
    """

    def __init__(
        self,
        *,
        imu: Optional[IMUNoise] = None,
        baro: Optional[BarometerNoise] = None,
        odo: Optional[OdometryNoise] = None,
        post_imu_accel: Optional[list[Callable[..., np.ndarray]]] = None,
        post_gyro: Optional[list[Callable[..., np.ndarray]]] = None,
        post_baro: Optional[list[Callable[..., float]]] = None,
        post_odom: Optional[list[Callable[..., tuple[np.ndarray, np.ndarray]]]] = None,
        state: Optional[NoiseState] = None,
        advance_time: bool = True,
    ) -> None:
        self.imu = imu
        self.baro = baro
        self.odo = odo
        self.post_imu_accel = post_imu_accel or []
        self.post_gyro = post_gyro or []
        self.post_baro = post_baro or []
        self.post_odom = post_odom or []
        self.state = state if state is not None else NoiseState()
        self.advance_time = advance_time

    def _tick_time(self, dt: float, t: Optional[float]) -> float:
        if t is not None:
            self.state.time_s = t
            return t
        if self.advance_time:
            self.state.time_s += dt
        return self.state.time_s

    def step_imu(
        self,
        gyro_truth: np.ndarray,
        accel_truth: np.ndarray,
        dt: float,
        *,
        t: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Une étape IMU : gyro + accélé avec couches post-traitement."""
        if self.imu is None:
            raise RuntimeError("IMU non configuré dans ce pipeline.")
        ts = self._tick_time(dt, t)
        g, a = self.imu.apply_gyro_accel(gyro_truth, accel_truth, dt, state=self.state, t=ts)
        for fn in self.post_gyro:
            g = np.asarray(fn(g, dt, ts, self.state), dtype=np.float64).reshape(3)
        for fn in self.post_imu_accel:
            a = np.asarray(fn(a, dt, ts, self.state), dtype=np.float64).reshape(3)
        return g, a

    def step_mag(self, mag_truth: np.ndarray, dt: float, *, t: Optional[float] = None) -> np.ndarray:
        if self.imu is None:
            raise RuntimeError("IMU non configuré.")
        ts = self._tick_time(dt, t)
        return self.imu.apply_magnetometer(mag_truth, dt, state=self.state, t=ts)

    def step_baro(
        self,
        pressure_truth_pa: float,
        dt: float,
        *,
        t: Optional[float] = None,
    ) -> float:
        if self.baro is None:
            raise RuntimeError("Baromètre non configuré.")
        ts = self._tick_time(dt, t)
        p = self.baro.apply_pressure(pressure_truth_pa, dt, state=self.state, t=ts)
        for fn in self.post_baro:
            p = float(fn(p, dt, ts, self.state))
        return p

    def step_odom(
        self,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        dt: float,
        *,
        t: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.odo is None:
            raise RuntimeError("Odométrie non configurée.")
        ts = self._tick_time(dt, t)
        v, w = self.odo.apply_twist(linear_velocity, angular_velocity, dt, state=self.state, t=ts)
        for fn in self.post_odom:
            v, w = fn(v, w, dt, ts, self.state)
        return v, w

    def reset(self, seed: Optional[int] = None) -> None:
        """Réinitialise le temps et tous les sous-simulateurs."""
        self.state.time_s = 0.0
        self.state.extra.clear()
        if self.imu is not None:
            self.imu.reset(seed)
        if self.baro is not None:
            self.baro.reset(seed)
        if self.odo is not None:
            self.odo.reset(seed)


def _normalize_imu_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Convertit listes YAML en tuples pour :class:`IMUNoiseConfig`."""
    out = dict(d)
    for key in ("gyro_misalignment_rad", "accel_misalignment_rad"):
        v = out.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            out[key] = (float(v[0]), float(v[1]), float(v[2]))
    return out


def _normalize_odometry_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Convertit listes YAML en tuples pour :class:`OdometryNoiseConfig`."""
    out = dict(d)
    s = out.get("slip_factor_range")
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        out["slip_factor_range"] = (float(s[0]), float(s[1]))
    return out


def build_pipeline_from_config(
    cfg: PipelineConfig | dict[str, Any],
) -> SensorNoisePipeline:
    """
    Construit un :class:`SensorNoisePipeline` à partir de :class:`PipelineConfig` ou d'un dict.

    ``artifact_builders`` permet d'enregistrer des constructeurs nommés pour les couches
    post-traitement (non sérialisables en YAML) si besoin.
    """
    if isinstance(cfg, dict):
        cfg = PipelineConfig(
            imu=cfg.get("imu"),
            barometer=cfg.get("barometer"),
            odometry=cfg.get("odometry"),
            seed=cfg.get("seed"),
            odometry_dim_linear=cfg.get("odometry_dim_linear", 2),
        )

    rng = np.random.default_rng(cfg.seed)

    imu: Optional[IMUNoise] = None
    if cfg.imu:
        imu = IMUNoise(IMUNoiseConfig(**_normalize_imu_dict(cfg.imu)), seed=cfg.seed, rng=rng)

    baro: Optional[BarometerNoise] = None
    if cfg.barometer:
        baro = BarometerNoise(BarometerNoiseConfig(**cfg.barometer), seed=cfg.seed, rng=rng)

    odo: Optional[OdometryNoise] = None
    if cfg.odometry:
        odo = OdometryNoise(
            OdometryNoiseConfig(**_normalize_odometry_dict(cfg.odometry)),
            dim_linear=cfg.odometry_dim_linear,
            seed=cfg.seed,
            rng=rng,
        )

    return SensorNoisePipeline(imu=imu, baro=baro, odo=odo)

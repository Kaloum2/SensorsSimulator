"""
Erreurs et artefacts de collecte : chocs, pics, pertes de mesure, vibrations, dérives lentes.

Ces modules sont conçus pour être chaînés après les modèles de capteurs ou combinés
via :class:`sensor_noise_sim.pipeline.SensorNoisePipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.random import Generator

from sensor_noise_sim.base import NoiseModel, NoiseState, as_array


@dataclass
class ShockConfig:
    """
    Impulsions de type choc (souvent sur l'accéléromètre).

    Attributes
    ----------
    probability_per_step : float
        Probabilité qu'un choc survienne à chaque appel (pas de temps).
    amplitude_std_m_s2 : float
        Écart-type de l'amplitude du pic (m/s²), tirée en valeur absolue puis signe aléatoire.
    axis_weights : tuple[float, float, float], optional
        Poids relatifs par axe pour la direction du choc (normalisés en interne).
    """

    probability_per_step: float = 0.0
    amplitude_std_m_s2: float = 1.0
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)


class ShockImpulse(NoiseModel):
    """
    Ajoute un vecteur impulsionnel aléatoire à un signal 3D (ex. accélération).

    Examples
    --------
    >>> shock = ShockImpulse(ShockConfig(probability_per_step=0.001, amplitude_std_m_s2=5.0))
    >>> a_noisy = shock.apply(accel_vector, dt=0.01)
    """

    def __init__(
        self,
        config: ShockConfig,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        w = np.array(config.axis_weights, dtype=np.float64)
        self._axis = w / (np.linalg.norm(w) + 1e-12)

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        x = as_array(truth).astype(np.float64).copy()
        if x.size != 3:
            raise ValueError("ShockImpulse attend un vecteur 3D.")
        if self.rng.random() < self.config.probability_per_step:
            amp = abs(float(self.rng.normal(0.0, self.config.amplitude_std_m_s2)))
            sign = 1.0 if self.rng.random() > 0.5 else -1.0
            x += sign * amp * self._axis
        return x


@dataclass
class SpikeConfig:
    """Pics isolés (outliers) sur un signal de dimension arbitraire."""

    probability_per_step: float = 0.0
    amplitude_std: float = 1.0


class SpikeNoise(NoiseModel):
    """Ajoute un pic gaussien sur une ou toutes les composantes avec probabilité faible."""

    def __init__(
        self,
        config: SpikeConfig,
        dim: int,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        self.dim = dim

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        x = as_array(truth).astype(np.float64).copy().reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Dimension {x.size} != {self.dim}.")
        if self.rng.random() < self.config.probability_per_step:
            idx = int(self.rng.integers(0, self.dim))
            x[idx] += float(self.rng.normal(0.0, self.config.amplitude_std))
        return x


@dataclass
class DropoutConfig:
    """
    Perte de mesure : remplace la sortie par NaN ou maintient la dernière valeur.

    mode : "nan" | "hold"
    """

    probability_per_step: float = 0.0
    mode: str = "nan"


class Dropout(NoiseModel):
    """Simule des trous dans les données."""

    def __init__(
        self,
        config: DropoutConfig,
        dim: int,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        self.dim = dim
        self._last: Optional[np.ndarray] = None

    def _reset_state(self) -> None:
        self._last = None

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        x = as_array(truth).astype(np.float64).copy().reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Dimension {x.size} != {self.dim}.")
        if self.rng.random() < self.config.probability_per_step:
            if self.config.mode == "nan":
                return np.full(self.dim, np.nan)
            if self.config.mode == "hold":
                if self._last is None:
                    self._last = x.copy()
                return self._last.copy()
            raise ValueError("mode doit être 'nan' ou 'hold'.")
        self._last = x.copy()
        return x


@dataclass
class VibrationConfig:
    """
    Vibration sinusoïdale + bruit léger (sur accélération ou autre vecteur 3D).

    frequencies_hz : liste de fréquences
    amplitudes_m_s2 : amplitude par fréquence (m/s²)
    phase_noise_rad : bruit de phase optionnel par pas
    """

    frequencies_hz: tuple[float, ...] = ()
    amplitudes_m_s2: tuple[float, ...] = ()
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    phase_noise_rad: float = 0.0


class VibrationOverlay(NoiseModel):
    """
    Ajoute une somme de sinusoïdes ``sum A_i sin(2π f_i t + φ_i)`` sur la direction ``axis``.

    Le temps ``t`` doit être fourni (via ``state.time_s`` ou l'argument ``t``).
    """

    def __init__(
        self,
        config: VibrationConfig,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        ax = np.array(config.axis, dtype=np.float64)
        self._axis = ax / (np.linalg.norm(ax) + 1e-12)
        self._phases = self.rng.uniform(0.0, 2 * np.pi, size=len(config.frequencies_hz))

    def _reset_state(self) -> None:
        self._phases = self.rng.uniform(0.0, 2 * np.pi, size=len(self.config.frequencies_hz))

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        x = as_array(truth).astype(np.float64).copy()
        if x.size != 3:
            raise ValueError("VibrationOverlay attend un vecteur 3D.")
        time_s = t if t is not None else (state.time_s if state is not None else 0.0)
        freqs = self.config.frequencies_hz
        amps = self.config.amplitudes_m_s2
        if len(freqs) != len(amps):
            raise ValueError("frequencies_hz et amplitudes_m_s2 doivent avoir la même longueur.")
        acc = 0.0
        for i, (f, a) in enumerate(zip(freqs, amps)):
            phi = self._phases[i] + float(self.rng.normal(0.0, self.config.phase_noise_rad))
            acc += float(a) * np.sin(2.0 * np.pi * float(f) * time_s + phi)
        x += acc * self._axis
        return x


@dataclass
class SoftIronHardIronConfig:
    """
    Distorsion magnétomètre (soft iron + hard iron simplifiés).

    hard_iron_bias : offset (µT ou T selon votre échelle)
    soft_iron_matrix : matrice 3x3 (identité par défaut)
    """

    hard_iron_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    soft_iron_matrix: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


class SoftIronHardIron(NoiseModel):
    """Applique ``m_meas = S @ m_truth + b`` (erreurs magnétiques déterministes)."""

    def __init__(self, config: SoftIronHardIronConfig) -> None:
        super().__init__(seed=0, rng=np.random.default_rng(0))
        self._b = np.array(config.hard_iron_bias, dtype=np.float64)
        self._S = np.array(config.soft_iron_matrix, dtype=np.float64)

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        m = as_array(truth).astype(np.float64).reshape(3)
        return self._S @ m + self._b


@dataclass
class CustomCallableConfig:
    """Enveloppe pour une fonction utilisateur ``fn(x, dt, t, state) -> array``."""

    fn: Callable[[np.ndarray, float, Optional[float], Optional[NoiseState]], np.ndarray]


class CustomNoise(NoiseModel):
    """
    Délègue à une fonction Python pour des erreurs ad hoc.

    Examples
    --------
    >>> def tilt(x, dt, t, state):
    ...     return x + np.array([0.01, 0, 0])
    >>> custom = CustomNoise(CustomCallableConfig(fn=tilt))
    """

    def __init__(self, config: CustomCallableConfig) -> None:
        super().__init__(seed=None, rng=np.random.default_rng())
        self._fn = config.fn

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        return np.asarray(self._fn(truth, dt, t, state), dtype=np.float64)

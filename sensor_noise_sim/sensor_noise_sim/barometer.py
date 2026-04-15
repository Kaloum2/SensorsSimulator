"""
Modèles de bruit pour baromètre / pression statique (et altitude dérivée).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from sensor_noise_sim.base import NoiseModel, NoiseState, as_array

# Constante typique pour conversion pression -> altitude (modèle isotherme simplifié)
DEFAULT_AIR_SCALE_PA_PER_M = 12.0  # ~ ρg en ordre de grandeur près la surface


@dataclass
class BarometerNoiseConfig:
    """
    Paramètres de bruit barométrique.

    Attributes
    ----------
    pressure_white_noise_std_pa : float
        Écart-type du bruit blanc sur la pression (Pa).
    pressure_bias_instability_std_pa : float
        Écart-type de la marche aléatoire du biais de pression (par sqrt(s) ou par s,
        voir ``use_allan_style_bias``).
    pressure_bias_initial_pa : float, optional
        Biais initial (Pa).
    temperature_white_noise_std_c : float
        Bruit blanc sur la mesure de température du capteur (°C), si utilisée.
    """

    pressure_white_noise_std_pa: float = 0.0
    pressure_bias_instability_std_pa: float = 0.0
    pressure_bias_initial_pa: float = 0.0
    temperature_white_noise_std_c: float = 0.0
    use_allan_style_bias: bool = True


class BarometerNoise(NoiseModel):
    """
    Bruit sur la pression (Pa) avec biais aléatoire optionnel.

    Examples
    --------
    >>> baro = BarometerNoise(BarometerNoiseConfig(pressure_white_noise_std_pa=5.0))
    >>> p = baro.apply_pressure(101325.0, dt=0.02)
    """

    def __init__(
        self,
        config: BarometerNoiseConfig,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        self._pressure_bias = float(config.pressure_bias_initial_pa)
        self._reset_state()

    def _reset_state(self) -> None:
        self._pressure_bias = float(self.config.pressure_bias_initial_pa)

    def _bias_step(self, dt: float) -> float:
        s = self.config.pressure_bias_instability_std_pa
        if self.config.use_allan_style_bias:
            return float(self.rng.normal(0.0, s * np.sqrt(max(dt, 0.0))))
        return float(self.rng.normal(0.0, s * dt))

    def _white_pressure(self, dt: float) -> float:
        sig = self.config.pressure_white_noise_std_pa
        return float(self.rng.normal(0.0, sig * np.sqrt(max(dt, 1e-12))))

    def apply_pressure(
        self,
        pressure_truth_pa: float,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> float:
        """Retourne la pression mesurée (Pa)."""
        self._pressure_bias += self._bias_step(dt)
        n = self._white_pressure(dt)
        return float(pressure_truth_pa) + self._pressure_bias + n

    def apply_temperature(
        self,
        temperature_truth_c: float,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> float:
        """Bruit blanc simple sur la température (°C), si le capteur la fournit."""
        sig = self.config.temperature_white_noise_std_c
        n = float(self.rng.normal(0.0, sig * np.sqrt(max(dt, 1e-12))))
        return float(temperature_truth_c) + n

    def pressure_to_altitude_error_m(
        self,
        pressure_error_pa: float,
        *,
        air_scale_pa_per_m: float = DEFAULT_AIR_SCALE_PA_PER_M,
    ) -> float:
        """
        Convertit une erreur de pression (Pa) en erreur d'altitude approximative (m).

        ``dp ≈ -ρ g dh`` ⇒ ``dh ≈ -dp / (ρg)``. Le paramètre ``air_scale_pa_per_m``
        vaut typiquement ~12 Pa/m près du niveau de la mer (à ajuster selon votre modèle).
        """
        return -float(pressure_error_pa) / max(air_scale_pa_per_m, 1e-9)

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """Si ``truth`` est scalaire ou (1,), retourne la pression bruitée."""
        p = float(as_array(truth).ravel()[0])
        return np.array([self.apply_pressure(p, dt, state=state, t=t)])

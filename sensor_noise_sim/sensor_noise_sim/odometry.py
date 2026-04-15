"""
Modèles de bruit pour odométrie (roues, vitesses linéaires/angulaires, déplacement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from sensor_noise_sim.base import NoiseModel, NoiseState, as_array


@dataclass
class OdometryNoiseConfig:
    """
    Paramètres d'odométrie (2D typique : v, ω ; ou 3D selon usage).

    Attributes
    ----------
    velocity_white_noise_std : float ou liste
        Bruit blanc sur la vitesse linéaire (m/s), par axe ou scalaire.
    angular_velocity_white_noise_std : float ou liste
        Bruit blanc sur la vitesse angulaire (rad/s).
    velocity_bias_instability_std : float ou liste
        Marche aléatoire du biais de vitesse (m/s par sqrt(s) ou par s).
    scale_error_ppm : float ou liste
        Erreur d'échelle en ppm sur la vitesse (diagonale).
    slip_probability : float
        Probabilité à chaque pas qu'un événement de glissement réduise la vitesse.
    slip_factor_range : tuple[float, float]
        Intervalle multiplicateur lors d'un glissement (ex. (0.7, 0.95)).
    encoder_quantization_m : float
        Pas de quantification sur la distance intégrée (0 = désactivé), en mètres.
    """

    velocity_white_noise_std: float | list[float] = 0.0
    angular_velocity_white_noise_std: float | list[float] = 0.0
    velocity_bias_instability_std: float | list[float] = 0.0
    velocity_bias_initial: float | list[float] | None = None
    scale_error_ppm: float | list[float] = 0.0
    slip_probability: float = 0.0
    slip_factor_range: tuple[float, float] = (0.7, 0.95)
    encoder_quantization_m: float = 0.0
    use_allan_style_bias: bool = True


def _vec_n(x: float | list[float], n: int, default: float = 0.0) -> np.ndarray:
    a = as_array(x if x is not None else [default] * n)
    if a.size == 1:
        return np.full(n, float(a[0]), dtype=np.float64)
    if a.size != n:
        raise ValueError(f"Attendu scalaire ou vecteur de taille {n}.")
    return a.astype(np.float64)


class OdometryNoise(NoiseModel):
    """
    Ajoute bruit, biais, échelle, glissement et quantification optionnelle sur l'odométrie.

    La méthode principale est :meth:`apply_twist` pour un torseur (v_x, v_y, omega) ou 3D.

    Examples
    --------
    >>> odo = OdometryNoise(OdometryNoiseConfig(velocity_white_noise_std=0.02, slip_probability=0.01))
    >>> v = np.array([1.0, 0.0])
    >>> w = np.array([0.05])
    >>> vm, wm = odo.apply_twist(v, w, dt=0.05)
    """

    def __init__(
        self,
        config: OdometryNoiseConfig,
        dim_linear: int = 2,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        self.dim_linear = dim_linear
        self._vel_bias: np.ndarray = np.zeros(dim_linear, dtype=np.float64)
        self._integrated_distance_m: float = 0.0
        self._reset_state()

    def _reset_state(self) -> None:
        n = self.dim_linear
        if self.config.velocity_bias_initial is not None:
            self._vel_bias = _vec_n(self.config.velocity_bias_initial, n)
        else:
            self._vel_bias = np.zeros(n, dtype=np.float64)
        self._integrated_distance_m = 0.0

    def _bias_step(self, dt: float) -> np.ndarray:
        sigma = _vec_n(self.config.velocity_bias_instability_std, self.dim_linear)
        if self.config.use_allan_style_bias:
            return self.rng.normal(0.0, 1.0, size=self.dim_linear) * sigma * np.sqrt(
                max(dt, 0.0)
            )
        return self.rng.normal(0.0, 1.0, size=self.dim_linear) * sigma * dt

    def _white_vel(self, dt: float) -> np.ndarray:
        sigma = _vec_n(self.config.velocity_white_noise_std, self.dim_linear)
        return self.rng.normal(0.0, 1.0, size=self.dim_linear) * sigma * np.sqrt(
            max(dt, 1e-12)
        )

    def _white_omega(self, dt: float, dim_angular: int) -> np.ndarray:
        sigma = _vec_n(self.config.angular_velocity_white_noise_std, dim_angular)
        return self.rng.normal(0.0, 1.0, size=dim_angular) * sigma * np.sqrt(
            max(dt, 1e-12)
        )

    def _scale_matrix(self) -> np.ndarray:
        ppm = _vec_n(self.config.scale_error_ppm, self.dim_linear)
        return np.diag(1.0 + ppm * 1e-6)

    def _maybe_slip(self) -> float:
        lo, hi = self.config.slip_factor_range
        if self.rng.random() < self.config.slip_probability:
            return float(self.rng.uniform(lo, hi))
        return 1.0

    def apply_twist(
        self,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applique le modèle à la vitesse linéaire et angulaire.

        Parameters
        ----------
        linear_velocity : (dim_linear,) ndarray
            m/s.
        angular_velocity : (dim_angular,) ndarray
            rad/s.
        dt : float
            Pas de temps (s).

        Returns
        -------
        linear_meas, angular_meas : ndarrays
        """
        v = as_array(linear_velocity).astype(np.float64).reshape(-1)
        w = as_array(angular_velocity).astype(np.float64).reshape(-1)
        if v.size != self.dim_linear:
            raise ValueError(
                f"linear_velocity doit avoir {self.dim_linear} composantes (got {v.size})."
            )
        dim_angular = w.size

        self._vel_bias += self._bias_step(dt)
        slip = self._maybe_slip()
        S = self._scale_matrix()
        v_noisy = S @ (v * slip + self._vel_bias) + self._white_vel(dt)
        w_noisy = w + self._white_omega(dt, dim_angular)

        q = self.config.encoder_quantization_m
        if q > 0.0:
            speed = float(np.linalg.norm(v_noisy))
            self._integrated_distance_m += speed * dt
            quantized = np.round(self._integrated_distance_m / q) * q
            # Réinjecte une petite erreur de cohérence sur la norme (simplification)
            if speed > 1e-9:
                v_noisy = v_noisy * (quantized / max(self._integrated_distance_m, 1e-12))

        return v_noisy, w_noisy

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """
        Concatène [v, ω] bruités si ``truth`` = np.concatenate([v, w]).
        """
        n = self.dim_linear
        if truth.size <= n:
            raise ValueError("truth doit contenir v et ω concaténés.")
        v, w = truth[:n], truth[n:]
        vm, wm = self.apply_twist(v, w, dt, state=state, t=t)
        return np.concatenate([vm, wm])

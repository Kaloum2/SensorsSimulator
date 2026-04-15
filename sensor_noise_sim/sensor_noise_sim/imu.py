"""
Modèles de bruit pour IMU (gyroscope, accéléromètre, magnétomètre optionnel).

Les modèles suivent une structure classique : bruit blanc additif, dérive de biais
(marche aléatoire), erreurs d'échelle et de non-orthogonalité simplifiées (matrices
diagonales ou complètes selon la configuration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from sensor_noise_sim.base import NoiseModel, NoiseState, as_array, _ensure_rng


@dataclass
class IMUNoiseConfig:
    """
    Paramètres de bruit IMU (toutes les grandeurs en unités SI cohérentes).

    Gyroscope (rad/s) :
        - ``gyro_white_noise_std`` : écart-type du bruit blanc (par axe, ou scalaire).
        - ``gyro_bias_instability_std`` : écart-type du bruit de marche aléatoire du biais
          (par sqrt(s) si ``use_allan_style_bias`` est vrai, sinon par s).

    Accéléromètre (m/s²) :
        - ``accel_white_noise_std``, ``accel_bias_instability_std`` : idem gyro.

    Magnétomètre (T ou µT selon votre convention — restez cohérent) :
        - ``mag_white_noise_std``, ``mag_bias_instability_std`` : optionnels.

    Erreurs multiplicatives :
        - ``gyro_scale_ppm``, ``accel_scale_ppm`` : erreur d'échelle en parties par million
          (diagonale). Valeur scalaire ou vecteur 3.
        - ``gyro_misalignment_rad``, ``accel_misalignment_rad`` : petits angles (rad)
          pour matrice triangulaire supérieure (3 paramètres pour 3 axes).

    Notes
    -----
    Si un capteur n'est pas utilisé, laisser les std à 0. et ne pas appeler la méthode
    correspondante, ou utiliser :class:`IMUNoise` avec des zéros uniquement.
    """

    gyro_white_noise_std: float | list[float] = 0.0
    gyro_bias_instability_std: float | list[float] = 0.0
    accel_white_noise_std: float | list[float] = 0.0
    accel_bias_instability_std: float | list[float] = 0.0

    mag_white_noise_std: float | list[float] = 0.0
    mag_bias_instability_std: float | list[float] = 0.0

    gyro_bias_initial: float | list[float] | None = None
    accel_bias_initial: float | list[float] | None = None
    mag_bias_initial: float | list[float] | None = None

    gyro_scale_ppm: float | list[float] = 0.0
    accel_scale_ppm: float | list[float] = 0.0

    gyro_misalignment_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_misalignment_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    use_allan_style_bias: bool = True
    """Si vrai, l'écart-type du biais est multiplié par ``sqrt(dt)`` (marche aléatoire)."""


def _vec3(x: float | list[float], default: float = 0.0) -> np.ndarray:
    a = as_array(x if x is not None else [default, default, default])
    if a.size == 1:
        return np.full(3, float(a[0]), dtype=np.float64)
    if a.size != 3:
        raise ValueError("Attendu scalaire ou vecteur de taille 3.")
    return a.astype(np.float64)


def _small_angle_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matrice de rotation pour petits angles (ordre X, Y, Z simplifié)."""
    return np.array(
        [
            [1.0, -rz, ry],
            [rz, 1.0, -rx],
            [-ry, rx, 1.0],
        ],
        dtype=np.float64,
    )


class IMUNoise(NoiseModel):
    """
    Simulateur de bruit IMU : gyro, accéléro, magnétomètre optionnel.

    À chaque pas, les biais de gyro/accélé/mag évoluent selon une marche aléatoire.
    Les mesures sont::

        mesure = M_mis * S * (vérité + biais + bruit_blanc)

    où ``S`` est l'échelle (diag) et ``M_mis`` une petite non-orthogonalité.

    Examples
    --------
    >>> imu = IMUNoise(IMUNoiseConfig(gyro_white_noise_std=0.001, accel_white_noise_std=0.01))
    >>> w = np.array([0., 0., 0.01])  # rad/s
    >>> a = np.array([0., 0., 9.81])
    >>> wg, ag = imu.apply_gyro_accel(w, a, dt=0.01)
    """

    def __init__(
        self,
        config: IMUNoiseConfig,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(seed=seed, rng=rng)
        self.config = config
        self._reset_state()

    def _reset_state(self) -> None:
        c = self.config
        self._gyro_bias = _vec3(c.gyro_bias_initial if c.gyro_bias_initial is not None else 0.0)
        self._accel_bias = _vec3(c.accel_bias_initial if c.accel_bias_initial is not None else 0.0)
        self._mag_bias = _vec3(c.mag_bias_initial if c.mag_bias_initial is not None else 0.0)

        self._gyro_white = _vec3(c.gyro_white_noise_std)
        self._gyro_bias_rw = _vec3(c.gyro_bias_instability_std)
        self._accel_white = _vec3(c.accel_white_noise_std)
        self._accel_bias_rw = _vec3(c.accel_bias_instability_std)
        self._mag_white = _vec3(c.mag_white_noise_std)
        self._mag_bias_rw = _vec3(c.mag_bias_instability_std)

        gppm = _vec3(c.gyro_scale_ppm)
        appm = _vec3(c.accel_scale_ppm)
        self._gyro_scale = np.diag(1.0 + gppm * 1e-6)
        self._accel_scale = np.diag(1.0 + appm * 1e-6)
        self._gyro_mis = _small_angle_matrix(*c.gyro_misalignment_rad)
        self._accel_mis = _small_angle_matrix(*c.accel_misalignment_rad)

    def _bias_step(self, sigma: np.ndarray, dt: float) -> np.ndarray:
        if self.config.use_allan_style_bias:
            return self.rng.normal(0.0, 1.0, size=3) * sigma * np.sqrt(max(dt, 0.0))
        return self.rng.normal(0.0, 1.0, size=3) * sigma * dt

    def _white(self, sigma: np.ndarray, dt: float) -> np.ndarray:
        # Bruit blanc discret : souvent modélisé avec variance ~ 1/dt pour densité spectrale ;
        # ici on utilise sigma * sqrt(dt) pour cohérence avec intégration discrète courante.
        return self.rng.normal(0.0, 1.0, size=3) * sigma * np.sqrt(max(dt, 1e-12))

    def apply_gyro_accel(
        self,
        gyro_truth: np.ndarray,
        accel_truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applique le bruit gyro + accéléromètre.

        Parameters
        ----------
        gyro_truth : (3,) ndarray
            Taux de rotation vérité (rad/s).
        accel_truth : (3,) ndarray
            Accélération spécifique vérité (m/s²).
        dt : float
            Pas de temps (s).

        Returns
        -------
        gyro_meas, accel_meas : ndarray (3,)
        """
        w = as_array(gyro_truth)
        a = as_array(accel_truth)
        if w.size != 3 or a.size != 3:
            raise ValueError("gyro_truth et accel_truth doivent avoir 3 composantes.")

        self._gyro_bias += self._bias_step(self._gyro_bias_rw, dt)
        self._accel_bias += self._bias_step(self._accel_bias_rw, dt)

        wg = self._white(self._gyro_white, dt)
        wa = self._white(self._accel_white, dt)

        gyro_noisy = w + self._gyro_bias + wg
        accel_noisy = a + self._accel_bias + wa

        gyro_meas = self._gyro_mis @ self._gyro_scale @ gyro_noisy
        accel_meas = self._accel_mis @ self._accel_scale @ accel_noisy
        return gyro_meas, accel_meas

    def apply_magnetometer(
        self,
        mag_truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """Applique biais + bruit blanc au champ magnétique vérité (vecteur 3D)."""
        m = as_array(mag_truth)
        if m.size != 3:
            raise ValueError("mag_truth doit avoir 3 composantes.")
        self._mag_bias += self._bias_step(self._mag_bias_rw, dt)
        wm = self._white(self._mag_white, dt)
        return m + self._mag_bias + wm

    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """
        Non utilisé directement pour l'IMU (préférer ``apply_gyro_accel``).

        Si ``truth`` a 6 éléments [gx,gy,gz,ax,ay,az], renvoie les 6 bruités concaténés.
        """
        if truth.size == 6:
            g, a = truth[:3], truth[3:]
            wg, wa = self.apply_gyro_accel(g, a, dt, state=state, t=t)
            return np.concatenate([wg, wa])
        raise ValueError("Pour IMU, utiliser apply_gyro_accel ou pass truth de taille 6.")

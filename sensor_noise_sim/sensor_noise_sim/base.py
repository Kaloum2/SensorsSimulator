"""
Classes de base : générateur aléatoire, protocole des modèles de bruit, état temporel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.random import Generator


def _ensure_rng(seed: Optional[int], rng: Optional[Generator]) -> Generator:
    """Retourne un générateur NumPy valide (priorité à ``rng``, sinon ``seed``)."""
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


@dataclass
class NoiseState:
    """
    Conteneur d'état mutable partagé entre plusieurs modèles (biais intégrés, temps, etc.).

    Attributes
    ----------
    time_s : float
        Temps simulé courant (s), incrémenté par les pipelines si demandé.
    extra : dict
        Espace libre pour extensions (ex. identifiants de segment).
    """

    time_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class NoiseModel(ABC):
    """
    Interface pour tout effet de bruit ou d'erreur appliqué à un signal.

    Les sous-classes typiques maintiennent un état interne (biais aléatoire, compteurs)
    et utilisent :meth:`reset` pour repartir d'une séquence reproductible.
    """

    def __init__(self, seed: Optional[int] = None, rng: Optional[Generator] = None) -> None:
        self._seed = seed
        self.rng = _ensure_rng(seed, rng)

    def reset(self, seed: Optional[int] = None) -> None:
        """Réinitialise le générateur et l'état du modèle."""
        if seed is not None:
            self._seed = seed
        self.rng = _ensure_rng(self._seed, None)
        self._reset_state()

    def _reset_state(self) -> None:
        """Surcharge pour réinitialiser les variables d'état (biais, etc.)."""

    @abstractmethod
    def apply(
        self,
        truth: np.ndarray,
        dt: float,
        *,
        state: Optional[NoiseState] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """
        Transforme la valeur ``vérité`` en mesure bruitée.

        Parameters
        ----------
        truth : ndarray
            Valeur physique idéale (forme laissée à la sous-classe).
        dt : float
            Pas de temps (s). Utilisé pour intégrer les marches aléatoires, etc.
        state : NoiseState, optional
            État partagé (temps, métadonnées).
        t : float, optional
            Temps absolu (s). Si fourni, peut remplacer ``state.time_s``.

        Returns
        -------
        ndarray
            Mesure simulée.
        """


def as_array(x: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    """Convertit une entrée scalaire ou séquence en ``ndarray`` 1D de type flottant."""
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        arr = np.array([arr.item()], dtype=dtype)
    return arr.reshape(-1)

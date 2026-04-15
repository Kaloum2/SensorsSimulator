"""
Démonstration : charge la config YAML et simule quelques pas capteurs.

Depuis le dossier du projet (là où se trouve ``pyproject.toml``) :

    poetry run python examples/demo.py
"""

from pathlib import Path

import numpy as np

from sensor_noise_sim import build_sensor_pipeline_from_dict, load_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "examples" / "config_example.yaml"
    data = load_config(cfg_path)
    pipe = build_sensor_pipeline_from_dict(data)

    dt = 0.01
    gyro = np.array([0.0, 0.0, 0.05])
    accel = np.array([0.05, 0.0, 9.81])

    for k in range(5):
        g_m, a_m = pipe.step_imu(gyro, accel, dt=dt)
        p_m = pipe.step_baro(101325.0, dt=dt)
        v_m, w_m = pipe.step_odom(np.array([0.3, 0.0]), np.array([0.02]), dt=dt)
        print(
            f"step {k}: |g|={np.linalg.norm(g_m):.4f} |a|={np.linalg.norm(a_m):.4f} "
            f"P={p_m:.1f} Pa |v|={np.linalg.norm(v_m):.3f}"
        )


if __name__ == "__main__":
    main()

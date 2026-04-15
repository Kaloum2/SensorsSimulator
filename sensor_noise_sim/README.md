# sensor-noise-sim

Bibliothèque Python pour simuler les bruits et erreurs de capteurs utilisés en robotique et navigation : **IMU** (gyroscope, accéléromètre, magnétomètre optionnel), **baromètre** et **odométrie**. Elle inclut aussi des **artefacts de collecte** (chocs, vibrations, pics, pertes de mesure) configurables.

## Installation avec Poetry

Nécessite [Poetry](https://python-poetry.org/docs/#installation) installé (`pip install poetry` ou installateur officiel).

```bash
cd sensor_noise_sim
poetry install
```

Pour un environnement virtuel activé et un shell Poetry :

```bash
poetry shell
```

## Installation avec pip

```bash
cd sensor_noise_sim
pip install -r requirements.txt
pip install -e .
```

(`requirements.txt` liste les dépendances minimales ; Poetry reste la référence pour les versions.)

## Utilisation rapide

```python
from sensor_noise_sim import build_sensor_pipeline_from_dict, load_config

data = load_config("examples/config_example.yaml")  # depuis le dossier du pyproject.toml
pipe = build_sensor_pipeline_from_dict(data)

g, a = pipe.step_imu(gyro_truth, accel_truth, dt=0.01)
p = pipe.step_baro(pressure_pa, dt=0.01)
v, w = pipe.step_odom(linear_velocity, angular_velocity, dt=0.05)
```

## Structure du paquet

| Module | Rôle |
|--------|------|
| `sensor_noise_sim.imu` | Bruit gyro / accélé / mag, biais, échelle, non-orthogonalité |
| `sensor_noise_sim.barometer` | Pression, biais, bruit blanc, conversion altitude approximative |
| `sensor_noise_sim.odometry` | Vitesse, glissement, échelle, quantification encodeur |
| `sensor_noise_sim.artifacts` | Chocs, vibrations, spikes, dropout, soft/hard iron magnétique |
| `sensor_noise_sim.pipeline` | Orchestration `SensorNoisePipeline` |
| `sensor_noise_sim.config` | Chargement YAML / JSON |
| `sensor_noise_sim.factory` | Construction depuis un dict complet (capteurs + artefacts) |

## Tests

```bash
poetry run pytest
```

## Licence

MIT

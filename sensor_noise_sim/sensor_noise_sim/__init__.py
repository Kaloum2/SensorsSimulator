"""
sensor_noise_sim — Simulation de bruits et erreurs pour capteurs (IMU, baromètre, odométrie).

Utilisation rapide
------------------

Depuis un fichier YAML::

    from sensor_noise_sim import build_sensor_pipeline_from_dict, load_config

    data = load_config(\"config.yaml\")
    pipe = build_sensor_pipeline_from_dict(data)
    g, a = pipe.step_imu(gyro_truth, accel_truth, dt=0.01)

Construction manuelle::

    from sensor_noise_sim import IMUNoise, IMUNoiseConfig, SensorNoisePipeline

    imu = IMUNoise(IMUNoiseConfig(gyro_white_noise_std=0.002))
    pipe = SensorNoisePipeline(imu=imu)
"""

from sensor_noise_sim.barometer import BarometerNoise, BarometerNoiseConfig
from sensor_noise_sim.base import NoiseModel, NoiseState
from sensor_noise_sim.config import (
    load_config,
    load_config_with_overrides,
    load_json,
    load_yaml,
    merge_dicts,
)
from sensor_noise_sim.factory import (
    build_accel_artifact_layers,
    build_gyro_artifact_layers,
    build_sensor_pipeline_from_dict,
)
from sensor_noise_sim.imu import IMUNoise, IMUNoiseConfig
from sensor_noise_sim.odometry import OdometryNoise, OdometryNoiseConfig
from sensor_noise_sim.pipeline import (
    PipelineConfig,
    SensorNoisePipeline,
    build_pipeline_from_config,
)

# Artefacts (import direct pour pipelines personnalisés)
from sensor_noise_sim.artifacts import (
    CustomCallableConfig,
    CustomNoise,
    Dropout,
    DropoutConfig,
    ShockConfig,
    ShockImpulse,
    SoftIronHardIron,
    SoftIronHardIronConfig,
    SpikeConfig,
    SpikeNoise,
    VibrationConfig,
    VibrationOverlay,
)

__all__ = [
    "NoiseModel",
    "NoiseState",
    "IMUNoise",
    "IMUNoiseConfig",
    "BarometerNoise",
    "BarometerNoiseConfig",
    "OdometryNoise",
    "OdometryNoiseConfig",
    "SensorNoisePipeline",
    "PipelineConfig",
    "build_pipeline_from_config",
    "build_sensor_pipeline_from_dict",
    "build_accel_artifact_layers",
    "build_gyro_artifact_layers",
    "load_config",
    "load_yaml",
    "load_json",
    "merge_dicts",
    "load_config_with_overrides",
    "ShockConfig",
    "ShockImpulse",
    "SpikeConfig",
    "SpikeNoise",
    "Dropout",
    "DropoutConfig",
    "VibrationConfig",
    "VibrationOverlay",
    "SoftIronHardIron",
    "SoftIronHardIronConfig",
    "CustomNoise",
    "CustomCallableConfig",
]

__version__ = "0.1.0"

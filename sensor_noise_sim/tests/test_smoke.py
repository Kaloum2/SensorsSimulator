import numpy as np

from sensor_noise_sim import (
    BarometerNoise,
    BarometerNoiseConfig,
    IMUNoise,
    IMUNoiseConfig,
    OdometryNoise,
    OdometryNoiseConfig,
    build_sensor_pipeline_from_dict,
)
from sensor_noise_sim.artifacts import ShockConfig, ShockImpulse


def test_imu_shapes():
    imu = IMUNoise(
        IMUNoiseConfig(
            gyro_white_noise_std=0.01,
            accel_white_noise_std=0.05,
            gyro_bias_instability_std=1e-6,
        ),
        seed=1,
    )
    g = np.array([0.0, 0.0, 0.1])
    a = np.array([0.0, 0.0, 9.81])
    gm, am = imu.apply_gyro_accel(g, a, dt=0.01)
    assert gm.shape == (3,) and am.shape == (3,)


def test_barometer_float():
    b = BarometerNoise(BarometerNoiseConfig(pressure_white_noise_std_pa=2.0), seed=2)
    p = b.apply_pressure(101325.0, dt=0.02)
    assert isinstance(p, float)


def test_odometry_twist():
    o = OdometryNoise(
        OdometryNoiseConfig(velocity_white_noise_std=0.01, slip_probability=0.0),
        dim_linear=2,
        seed=3,
    )
    v = np.array([1.0, 0.0])
    w = np.array([0.02])
    vm, wm = o.apply_twist(v, w, dt=0.05)
    assert vm.shape == (2,) and wm.shape == (1,)


def test_yaml_style_pipeline():
    data = {
        "seed": 42,
        "imu": {
            "gyro_white_noise_std": 0.001,
            "accel_white_noise_std": 0.02,
        },
        "barometer": {"pressure_white_noise_std_pa": 5.0},
        "odometry": {"velocity_white_noise_std": 0.03},
        "artifacts": {
            "accel": {
                "shock": {"probability_per_step": 0.0, "amplitude_std_m_s2": 1.0},
                "vibration": {
                    "frequencies_hz": [30.0],
                    "amplitudes_m_s2": [0.1],
                    "axis": [0.0, 0.0, 1.0],
                },
            }
        },
    }
    pipe = build_sensor_pipeline_from_dict(data)
    g, a = pipe.step_imu(np.zeros(3), np.array([0.0, 0.0, 9.81]), dt=0.01)
    assert g.shape == (3,) and a.shape == (3,)
    p = pipe.step_baro(101325.0, dt=0.02)
    assert p > 0
    v, w = pipe.step_odom(np.array([0.5, 0.0]), np.array([0.0]), dt=0.05)
    assert v.shape == (2,)


def test_shock_deterministic_with_seed():
    s = ShockImpulse(ShockConfig(probability_per_step=1.0, amplitude_std_m_s2=2.0), seed=99)
    x = np.array([0.0, 0.0, 9.81])
    y = s.apply(x, dt=0.01)
    assert not np.allclose(y, x)

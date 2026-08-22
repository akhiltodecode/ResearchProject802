
import numpy as np
import pandas as pd

FEATURES = [
    "machine_load_pct", "spindle_speed_rpm", "feed_rate_mm_min",
    "ambient_temp_c", "vibration_mm_s", "pressure_bar",
    "runtime_hours", "product_hardness", "tool_wear_pct"
]
TARGET = "energy_kwh"

def make_dataset(n_samples: int = 120_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    load = rng.uniform(25, 100, n_samples)
    rpm = rng.uniform(800, 5000, n_samples)
    feed = rng.uniform(100, 1800, n_samples)
    ambient = rng.normal(25, 5, n_samples).clip(10, 42)
    vibration = (0.7 + 0.02 * load + 0.00018 * rpm + rng.normal(0, .35, n_samples)).clip(.2, 8)
    pressure = rng.uniform(3.5, 9.5, n_samples)
    runtime = rng.uniform(0, 12000, n_samples)
    hardness = rng.uniform(35, 75, n_samples)
    tool_wear = (runtime / 12000 * 75 + rng.normal(10, 12, n_samples)).clip(0, 100)

    # Nonlinear, noisy energy response approximating an industrial machine.
    energy = (
        4.0
        + 0.13 * load
        + 0.00072 * rpm
        + 0.0015 * feed
        + 0.055 * np.maximum(ambient - 22, 0)
        + 0.30 * vibration
        + 0.12 * pressure
        + 0.020 * hardness
        + 0.025 * tool_wear
        + 0.000012 * load * rpm
        + 0.000004 * feed * hardness
        + 2.2 * np.sin(rpm / 850.0) ** 2
        + rng.normal(0, 0.8, n_samples)
    ).clip(1, None)

    return pd.DataFrame({
        "machine_load_pct": load,
        "spindle_speed_rpm": rpm,
        "feed_rate_mm_min": feed,
        "ambient_temp_c": ambient,
        "vibration_mm_s": vibration,
        "pressure_bar": pressure,
        "runtime_hours": runtime,
        "product_hardness": hardness,
        "tool_wear_pct": tool_wear,
        TARGET: energy,
    })

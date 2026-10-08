from copy import deepcopy
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from baseline6.data import BASE_WEATHER_COLUMNS, PV_WEATHER_COLUMNS, TARGET_COLUMNS
from baseline6.model import ConditionalTimeGrad, generate_scenarios
from baseline6.runtime import load_config
from baseline6.sweep import sweep, trial_configs
from baseline6.train import make_dataset, train


@pytest.fixture
def tiny_config(tmp_path):
    torch.set_num_threads(1)
    config = load_config("baseline6/configs/heew.yaml")
    config["model"].update(num_cells=8, num_layers=1, conditioning_length=8, residual_layers=2, residual_channels=4)
    config["diffusion"].update(num_steps=2, beta_end=0.1)
    config["training"].update(epochs=1, batch_size=2, val_every=1)
    config["evaluation"].update(batch_size=2, val_scenarios=2, test_scenarios=3, sample_chunk_size=4)
    config["run"].update(device="cpu", output_root=str(tmp_path / "runs"))
    timestamps = pd.DatetimeIndex([
        ts for year in (2014, 2020, 2021, 2022)
        for ts in pd.date_range(f"{year}-01-01", periods=48, freq="h")
    ])
    cols = dict(Year=timestamps.year, Month=timestamps.month, Day=timestamps.day, Hour=timestamps.hour)
    rng = np.random.default_rng(12)
    energy = pd.DataFrame(cols)
    weather = pd.DataFrame(cols)
    for i, label in enumerate(TARGET_COLUMNS):
        energy[label] = rng.uniform(10, 50, len(timestamps)) * (i + 1)
    for label in BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS:
        weather[label] = rng.uniform(0, 1, len(timestamps))
    for kind, frame in (("energy", energy), ("weather", weather)):
        path = tmp_path / f"{kind}.csv"
        frame.to_csv(path, index=False)
        config["data"][f"{kind}_path"] = str(path)
    return config


def test_train_only_scaling_and_splits(tiny_config):
    train_data = make_dataset(tiny_config, "train")
    val_data = make_dataset(tiny_config, "val")
    test_data = make_dataset(tiny_config, "test")
    assert len(train_data) == 4 and len(val_data) == len(test_data) == 2
    assert all(d.startswith("2021") for d in val_data.dates)
    assert all(d.startswith("2022") for d in test_data.dates)
    frame = pd.read_csv(tiny_config["data"]["energy_path"])
    train_values = frame.loc[frame.Year <= 2020, TARGET_COLUMNS].to_numpy(np.float32)
    np.testing.assert_allclose(train_data.target_mean, train_values.mean(0), rtol=1e-6)
    np.testing.assert_array_equal(train_data.target_mean, test_data.target_mean)
    np.testing.assert_array_equal(train_data.target_std, val_data.target_std)


def test_six_trials_and_reject_duplicates(tiny_config):
    search = yaml.safe_load(open("baseline6/configs/search_space.yaml"))
    assert len(trial_configs(tiny_config, search)) == 6
    bad = deepcopy(search)
    bad["configurations"].pop()
    with pytest.raises(ValueError, match="six"):
        trial_configs(tiny_config, bad)
    bad = deepcopy(search)
    bad["configurations"][1]["overrides"] = bad["configurations"][0]["overrides"]
    with pytest.raises(ValueError, match="Duplicate"):
        trial_configs(tiny_config, bad)


def test_checkpoint_to_test_outputs(tiny_config, tmp_path, monkeypatch):
    checkpoint = train(tiny_config, "smoke")
    from baseline6.evaluate import main
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                                      "--outdir", str(out), "--device", "cpu"])
    main()
    payload = np.load(out / "baseline6_scenarios.npz")
    assert payload["scenarios"].shape == (2, 3, 4, 24)
    metrics = json.loads((out / "global_metrics.json").read_text())
    for label in TARGET_COLUMNS:
        for suffix in ("RMSE_Z", "MAE_Z", "Precision_Z", "Recall_Z", "CR", "IW"):
            assert np.isfinite(metrics[f"{label}_{suffix}"])
    assert metrics["sample_seed"] == 90042
    assert len(list((out / "random_timeseries_50").glob("*.png"))) == 2
    assert (out / "pearson" / "generated_global_pearson.png").exists()
    with pytest.raises(FileExistsError):
        train(tiny_config, "smoke")


def test_sweep_exports_usable_winner(tiny_config, tmp_path):
    search = {"configurations": [
        {"id": f"trial{i}", "overrides": {"model.num_cells": 16,
                                         "training.learning_rate": (i + 1) * 1e-5}}
        for i in range(6)
    ]}
    out = tmp_path / "sweep"
    best = sweep(tiny_config, search, out)
    winner = yaml.safe_load((out / "best_config.yaml").read_text())
    assert winner["training"]["learning_rate"] == best["learning_rate"]
    assert len(pd.read_csv(out / "tuning_results.csv")) == 6
    assert sweep(tiny_config, search, out, skip_completed=True) == best
    winner["run"]["seed"] = 123
    assert train(winner).exists()
    # Changed CSV contents must never reuse a completed tuning run.
    energy_path = tiny_config["data"]["energy_path"]
    frame = pd.read_csv(energy_path)
    frame.loc[0, "Electricity"] += 1
    frame.to_csv(energy_path, index=False)
    with pytest.raises(ValueError, match="different protocol"):
        sweep(tiny_config, search, out, skip_completed=True)


def test_unknown_override_rejected():
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config("baseline6/configs/heew.yaml", ["diffusion.sampling_steps=20"])

"""Existing checkpoints preserve DDPM samples, chunking, and dataset checks."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from baseline6 import evaluate, score_npz
from baseline6.model import ConditionalTimeGrad, generate_scenarios
from baseline6.runtime import seed_everything
from baseline6.train import data_manifest, make_dataset
from test_baseline6 import tiny_config


@pytest.mark.parametrize("seed,batch_size,chunk_size,max_days", [(42, 2, 4, None), (3407, 1, 2, 1)])
def test_ddpm_samples_and_rescore(tiny_config, tmp_path, monkeypatch, seed, batch_size, chunk_size, max_days):
    config = deepcopy(tiny_config)
    config["run"]["seed"] = seed
    config["evaluation"].update(batch_size=batch_size, sample_chunk_size=chunk_size)
    dataset = make_dataset(config, "test")
    seed_everything(seed)
    model = ConditionalTimeGrad(config, dataset.condition_channels).eval()
    payload = {"model": model.state_dict(), "config": config, "epoch": 3,
               "method": "conditional-timegrad-baseline6", "data_manifest": data_manifest(config, dataset)}
    checkpoint = tmp_path / "best.pt"
    torch.save(payload, checkpoint)
    before = checkpoint.read_bytes()
    rng = torch.Generator(device="cpu").manual_seed(seed + 90_000)
    expected, processed = [], 0
    with torch.no_grad():
        for conditions, year, target, dates in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            if max_days is not None and processed >= max_days:
                break
            take = len(dates) if max_days is None else min(len(dates), max_days - processed)
            z = generate_scenarios(model, conditions[:take], year[:take], 3, rng, chunk_size)
            x = dataset.denormalize(z)
            x[:, :, 3].clamp_(min=0)
            expected.append(x.numpy())
            processed += take
    out = tmp_path / "evaluation"
    args = ["evaluate", "--checkpoint", str(checkpoint), "--outdir", str(out), "--no-plots", "--device", "cpu"]
    if max_days is not None:
        args.extend(["--max-days", str(max_days)])
    monkeypatch.setattr(sys, "argv", args)
    evaluate.main()
    assert checkpoint.read_bytes() == before
    with np.load(out / "baseline6_scenarios.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["scenarios"], np.concatenate(expected))
        np.testing.assert_array_equal(saved["target_std"], dataset.target_std)
        assert saved["conditions"].shape == (processed, 18, 24)
        assert saved["sample_seed"].item() == seed + 90_000
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["training_timesteps"] == 2 and result["sampling_steps"] == 2
    assert result["denoiser_calls_per_trajectory"] == 48
    assert result["conditioning_protocol"] == "cold_start_generated_within_day_lag1"
    assert result["sampling_chunk_size"] == chunk_size and result["sampling_batch_size"] == batch_size
    assert result["partial_test"] == (max_days == 1)
    assert result["VS_pair_count"] == 3456 and result["seed"] == seed
    rescored = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(out / "baseline6_scenarios.npz"),
                                     "--outdir", str(rescored), "--no-plots"])
    score_npz.main()
    offline = json.loads((rescored / "global_metrics.json").read_text())
    for key in ("ES", "VS", "macro_nCRPS", "PV_R2", "PV_IS", "PV_RMSE_Z", "sample_seed"):
        assert offline[key] == result[key]
    # Keep the original hash/normalization guard before model sampling.
    bad_payload = {**payload, "data_manifest": {}}
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save(bad_payload, bad_checkpoint)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(bad_checkpoint),
                                     "--outdir", str(tmp_path / "bad_out"), "--device", "cpu"])
    with pytest.raises(ValueError, match="Data or normalization differs"):
        evaluate.main()


def test_rescore_missing_stats_uses_checkpoint_manifest(tiny_config, tmp_path, monkeypatch):
    dataset = make_dataset(tiny_config, "test")
    targets = dataset.denormalize(dataset.targets).numpy()
    source = tmp_path / "legacy.npz"
    np.savez(source, scenarios=np.repeat(targets[:, None], 3, axis=1), targets=targets,
             dates=dataset.dates, seed=42)
    out = tmp_path / "rescored"
    args = ["score_npz", "--npz", str(source), "--outdir", str(out), "--no-plots"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        score_npz.main()
    checkpoint = tmp_path / "stats.pt"
    torch.save({"data_manifest": data_manifest(tiny_config, dataset), "epoch": 7}, checkpoint)
    monkeypatch.setattr(sys, "argv", args + ["--checkpoint", str(checkpoint)])
    score_npz.main()
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["ES"] == pytest.approx(0, abs=1e-12)
    assert result["VS"] == pytest.approx(0, abs=1e-12)
    assert "checkpoint data_manifest" in result["normalization_source"]
    bad = tmp_path / "bad_stats.pt"
    torch.save({}, bad)
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(source), "--outdir", str(tmp_path / "bad_rescore"), "--checkpoint", str(bad)])
    with pytest.raises(ValueError, match="lacks training normalization"):
        score_npz.main()


def test_rescore_checkpoint_settings_and_normalization_guard(tiny_config, tmp_path, monkeypatch):
    dataset = make_dataset(tiny_config, "test")
    targets = dataset.denormalize(dataset.targets).numpy()
    source = tmp_path / "saved.npz"
    np.savez(source, scenarios=np.repeat(targets[:, None], 3, axis=1), targets=targets,
             dates=dataset.dates, seed=123, target_mean=dataset.target_mean,
             target_std=dataset.target_std)
    checkpoint = tmp_path / "best.pt"
    payload = {"data_manifest": data_manifest(tiny_config, dataset),
               "config": {"evaluation": {"precision_recall_k": 1, "max_precision_samples": 4}}}
    torch.save(payload, checkpoint)
    out = tmp_path / "report"
    args = ["score_npz", "--npz", str(source), "--checkpoint", str(checkpoint), "--no-plots"]
    monkeypatch.setattr(sys, "argv", args + ["--outdir", str(out)])
    score_npz.main()
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["precision_recall_k"] == 1
    assert result["max_precision_samples"] == 4
    override = tmp_path / "override"
    monkeypatch.setattr(sys, "argv", args + ["--outdir", str(override),
                                           "--precision-recall-k", "2", "--max-precision-samples", "5"])
    score_npz.main()
    result = json.loads((override / "global_metrics.json").read_text())
    assert result["precision_recall_k"] == 2 and result["max_precision_samples"] == 5
    payload["data_manifest"]["normalization"]["target_std"][0] += 1
    torch.save(payload, checkpoint)
    monkeypatch.setattr(sys, "argv", args + ["--outdir", str(tmp_path / "mismatch")])
    with pytest.raises(ValueError, match="normalization differs"):
        score_npz.main()

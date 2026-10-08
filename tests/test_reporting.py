import csv
import json
import sys

import numpy as np
import pytest
import torch

from baseline6.data import TARGET_COLUMNS, HEEWConditionDataset
from baseline6.metrics import summarize
from baseline6.reporting import CHANNEL_SUFFIXES, write_report
from baseline6 import score_npz


def fixture_arrays():
    rng = np.random.default_rng(20)
    target = rng.normal(size=(3, 4, 24))
    samples = target[:, None] + rng.normal(size=(3, 5, 4, 24))
    return samples, target, np.arange(4.), np.arange(1., 5.), [f"2022-01-0{i}" for i in range(1, 4)]


def test_report_schema_legacy_alias_and_plots(tmp_path):
    samples, target, mean, std, dates = fixture_arrays()
    result = write_report(samples, target, mean, std, dates, TARGET_COLUMNS, tmp_path)
    expected = {"global_metrics.csv", "global_metrics_wide.csv", "global_metrics.json",
                "metrics_summary.json", "channel_metrics.csv", "daily_scores.csv",
                "metric_definitions.json", "global_pearson.png", "pearson", "random_timeseries_50"}
    assert expected == {p.name for p in tmp_path.iterdir()}
    assert len(list((tmp_path / "pearson").glob("*.png"))) == 3
    assert len(list((tmp_path / "random_timeseries_50").glob("*.png"))) == 3
    assert result["VS_pair_count"] == 3456
    assert result["VS"] == result["VS_Z"]
    assert result["ES"] == result["ES_Z"]
    with (tmp_path / "channel_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and set(rows[0]) == {"channel", *CHANNEL_SUFFIXES}
    with (tmp_path / "global_metrics.csv").open() as handle:
        assert next(csv.reader(handle)) == ["metric", "value"]
    legacy = summarize(samples, target, TARGET_COLUMNS, mean, std)
    for key, value in legacy.items():
        assert result[key] == pytest.approx(value)
    with pytest.raises(FileExistsError):
        write_report(samples, target, mean, std, dates, TARGET_COLUMNS, tmp_path)


def test_constant_targets_json_null_and_no_plots(tmp_path):
    result = write_report(np.ones((2, 3, 4, 24)), np.ones((2, 4, 24)),
                          np.zeros(4), np.ones(4), ["2022-01-01", "2022-01-02"],
                          TARGET_COLUMNS, tmp_path, no_plots=True)
    assert result["PV_R2"] is None
    assert result["random_timeseries_count"] == 0
    json.loads((tmp_path / "global_metrics.json").read_text(),
               parse_constant=lambda x: pytest.fail(f"Invalid JSON constant {x}"))
    assert not (tmp_path / "pearson").exists()


@pytest.mark.parametrize("invalid", ["nan", "zero_std", "dates"])
def test_invalid_inputs_fail_without_outputs(tmp_path, invalid):
    samples, target, mean, std, dates = fixture_arrays()
    if invalid == "nan":
        samples[0, 0, 0, 0] = np.nan
    elif invalid == "zero_std":
        std[0] = 0
    else:
        dates.pop()
    with pytest.raises(ValueError):
        write_report(samples, target, mean, std, dates, TARGET_COLUMNS, tmp_path)
    assert not list(tmp_path.iterdir())


def test_npz_rescore_without_checkpoint(tmp_path, monkeypatch):
    samples, target, mean, std, dates = fixture_arrays()
    source = tmp_path / "old.npz"
    np.savez_compressed(source, scenarios=samples, targets=target, dates=dates,
                        target_mean=mean, target_std=std, seed=123,
                        channel_names=TARGET_COLUMNS, sample_seed=90123)
    original = source.read_bytes()
    out = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(source), "--outdir", str(out), "--no-plots"])
    score_npz.main()
    scores = json.loads((out / "global_metrics.json").read_text())
    assert scores["seed"] == 123 and scores["sample_seed"] == 90123
    assert scores["rescored_without_sampling"]
    assert source.read_bytes() == original
    direct = summarize(samples, target, TARGET_COLUMNS, mean, std, seed=123)
    for key, value in direct.items():
        assert scores[key] == pytest.approx(value)


def test_legacy_sidecar_seed_and_explicit_override(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(2, 3, 4, 24)), rng.normal(size=(2, 4, 24))
    source = tmp_path / "baseline6_scenarios.npz"
    np.savez(source, scenarios=x, targets=y, dates=["2022-01-01", "2022-01-02"],
             target_mean=np.zeros(4), target_std=np.ones(4))
    out = tmp_path / "rescored"
    command = ["score_npz", "--npz", str(source), "--outdir", str(out), "--no-plots"]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit):
        score_npz.main()
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(source), "--outdir", str(tmp_path / "explicit"),
                                     "--seed", "123", "--no-plots"])
    score_npz.main()
    explicit = json.loads((tmp_path / "explicit" / "global_metrics.json").read_text())
    assert explicit["seed"] == 123 and explicit["sample_seed"] is None
    monkeypatch.setattr(sys, "argv", command)
    sidecar = tmp_path / "global_metrics.json"
    sidecar.write_text(json.dumps({"seed": 3407, "sampler": "DDPM_full_step_autoregressive",
                                  "num_test_days": 2, "scenarios_per_day": 3}))
    score_npz.main()
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["seed"] == 3407 and result["sample_seed"] == 93407
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(source), "--outdir", str(tmp_path / "override"),
                                     "--seed", "42", "--no-plots"])
    score_npz.main()
    result = json.loads((tmp_path / "override" / "global_metrics.json").read_text())
    assert result["seed"] == 42 and result["sample_seed"] == 93407
    assert result["original_evaluation_seed"] == 3407


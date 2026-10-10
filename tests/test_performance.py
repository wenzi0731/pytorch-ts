from copy import deepcopy
import json

import pytest
import torch

from baseline6.benchmark import benchmark
from baseline6.model import ConditionalTimeGrad, generate_scenarios
from baseline6.performance import Progress
from baseline6.train import train
from baseline6.runtime import load_checkpoint
from test_baseline6 import tiny_config


@pytest.mark.parametrize("steps", [2, 7, 100])
@pytest.mark.parametrize("batch", [1, 3])
def test_cached_ddpm_exact_values_rng_and_projection_counts(tiny_config, steps, batch):
    tiny_config["diffusion"]["num_steps"] = steps
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    diff = model.diffusion
    torch.nn.init.normal_(diff.denoise_fn.output_projection.weight, std=0.02)
    cond = torch.randn(batch, 1, 8)
    calls = []
    hook = diff.denoise_fn.residual_layers[0].conditioner_projection.register_forward_hook(
        lambda *args: calls.append(1))
    diff.cache_conditioner, diff.debug_checks = False, True
    reference_rng = torch.Generator().manual_seed(42)
    expected = diff.sample_with_generator(cond, reference_rng)
    assert len(calls) == steps
    calls.clear()
    diff.cache_conditioner, diff.debug_checks = True, False
    actual_rng = torch.Generator().manual_seed(42)
    actual = diff.sample_with_generator(cond, actual_rng)
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(reference_rng.get_state(), actual_rng.get_state())


def test_full_day_progress_and_fresh_cache(tiny_config):
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    torch.nn.init.normal_(model.diffusion.denoise_fn.output_projection.weight, std=0.02)
    c, year = torch.randn(2, 18, 24), torch.randn(2, 1, 24)
    def sample(progress=None):
        return generate_scenarios(model, c, year, 3, torch.Generator().manual_seed(42), 4, progress)
    model.diffusion.cache_conditioner = False
    reference = sample()
    model.diffusion.cache_conditioner = True
    updates = []
    optimized = sample(lambda *args: updates.append(args))
    torch.testing.assert_close(reference, optimized, rtol=0, atol=0)
    assert updates[-1] == (6, 6)
    assert any(len(row) == 3 and row[-1] == 24 for row in updates)
    # Caches are local, so changed weights cannot reuse stale projections.
    with torch.no_grad():
        model.diffusion.denoise_fn.cond_upsampler.linear1.weight.add_(0.3)
    after_update = sample()
    model.diffusion.cache_conditioner = False
    torch.testing.assert_close(after_update, sample(), rtol=0, atol=0)


@pytest.mark.parametrize("debug", [False, True])
def test_nonfinite_samples_still_fail(tiny_config, debug):
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    model.diffusion.debug_checks = debug
    with torch.no_grad():
        model.diffusion.denoise_fn.output_projection.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        model.diffusion.sample_with_generator(torch.ones(2, 1, 8), torch.Generator().manual_seed(42))


@pytest.mark.parametrize("decay", [0.0, 0.9])
def test_training_timing_metadata_and_ema(tiny_config, decay, capsys):
    tiny_config["training"]["ema_decay"] = decay
    checkpoint_path = train(tiny_config, "timed")
    saved = load_checkpoint(checkpoint_path)
    record = json.loads((checkpoint_path.parent / "history.jsonl").read_text().splitlines()[0])
    assert record["training_seconds"] > 0 and record["validation_seconds"] > 0
    assert record["epoch_seconds"] >= record["training_seconds"] + record["validation_seconds"]
    if decay == 0:
        for key, value in saved["model"].items():
            torch.testing.assert_close(value, saved["raw_model"][key], rtol=0, atol=0)
    else:
        assert any(not torch.equal(value, saved["raw_model"][key]) for key, value in saved["model"].items())
    output = capsys.readouterr().out
    assert '"event": "run_start"' in output and '"event": "validation"' in output
    assert '"actual_model_device": "cpu"' in output


def test_benchmark_preserves_settings_and_reports_parity(tiny_config):
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    model.diffusion.debug_checks = True
    state_before = deepcopy(model.state_dict())
    result = benchmark(model, torch.zeros(1, 18, 24), torch.zeros(1, 1, 24),
                       scenarios=1, chunk_size=1, repeats=1)
    assert result["rng_state_equal"] and result["bitwise_equal"]
    assert result["sampler_speedup"] > 0
    assert model.diffusion.debug_checks is True
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)


def test_progress_can_be_disabled_without_rng_change(monkeypatch, capsys):
    monkeypatch.setenv("TIMEGRAD_PROGRESS", "0")
    before = torch.random.get_rng_state().clone()
    Progress("test").emit(force=True, phase="start")
    assert capsys.readouterr().out == ""
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runner required")
def test_cuda_deepcopy_flatten_and_sampler_parity(tiny_config):
    import warnings
    model = deepcopy(ConditionalTimeGrad(tiny_config, 18).cuda()).eval()
    torch.nn.init.normal_(model.diffusion.denoise_fn.output_projection.weight, std=0.02)
    c, year = torch.randn(2, 18, 24, device="cuda"), torch.zeros(2, 1, 24, device="cuda")
    with warnings.catch_warnings(record=True) as caught:
        model.diffusion.cache_conditioner = False
        expected = model.sample(c, year, torch.Generator(device="cuda").manual_seed(42))
        model.diffusion.cache_conditioner = True
        actual = model.sample(c, year, torch.Generator(device="cuda").manual_seed(42))
    assert not any("single contiguous chunk" in str(w.message) for w in caught)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

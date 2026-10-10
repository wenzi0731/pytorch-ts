from copy import deepcopy
import inspect

import pytest
import torch

from baseline6.model import ConditionalTimeGrad, generate_scenarios
from baseline6.upstream import GaussianDiffusion
from test_baseline6 import tiny_config


def tensors():
    return torch.randn(2, 4, 24), torch.randn(2, 18, 24), torch.randn(2, 1, 24)


def test_original_noise_loss_and_ddpm_sample_parity(tiny_config):
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    diffusion = model.diffusion
    original = GaussianDiffusion(deepcopy(diffusion.denoise_fn), input_size=4,
                                 diff_steps=2, beta_end=0.1, beta_schedule="linear")
    original.load_state_dict(diffusion.state_dict())
    x, cond, noise = torch.randn(3, 1, 4), torch.randn(3, 1, 8), torch.randn(3, 1, 4)
    t = torch.tensor([0, 1, 1])
    torch.testing.assert_close(diffusion.p_losses(x, cond, t, noise),
                               original.p_losses(x, cond, t, noise), rtol=0, atol=0)
    torch.manual_seed(345)
    expected = original.sample(cond=cond)
    global_before = torch.random.get_rng_state().clone()
    actual = diffusion.sample_with_generator(cond, torch.Generator().manual_seed(345))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(global_before, torch.random.get_rng_state())


def test_no_current_or_future_target_in_rnn_condition(tiny_config):
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    y, c, year = tensors()
    changed = y.clone()
    changed[:, :, 7:] += 100
    first = model.teacher_forced_condition(y, c, year)
    second = model.teacher_forced_condition(changed, c, year)
    torch.testing.assert_close(first[:, :8], second[:, :8], rtol=0, atol=0)
    assert not torch.allclose(first[:, 8:], second[:, 8:])
    # Vectorized training RNN equals sequential teacher-forced recurrence.
    covariates = model.covariates(c, year)
    state, sequential = None, []
    for h in range(24):
        previous = torch.zeros(2, 1, 4) if h == 0 else y[:, :, h-1:h].transpose(1, 2)
        out, state = model.rnn(model.rnn_input(previous, covariates[:, h:h+1]), state)
        sequential.append(model.proj_dist_args(out))
    torch.testing.assert_close(first, torch.cat(sequential, dim=1))
    assert list(inspect.signature(model.sample).parameters) == ["conditions", "year", "generator", "progress"]


def test_condition_gradients_joint_channels_and_sampling(tiny_config):
    model = ConditionalTimeGrad(tiny_config, 18)
    # Upstream zero-initializes output weights; activate them to test gradient paths.
    torch.nn.init.normal_(model.diffusion.denoise_fn.output_projection.weight, std=0.05)
    y, c, year = tensors()
    c.requires_grad_()
    year.requires_grad_()
    model(y, c, year).backward()
    assert c.grad.abs().sum() > 0 and year.grad.abs().sum() > 0
    assert model.rnn.weight_ih_l0.grad.abs().sum() > 0
    assert model.embed.weight.grad.abs().sum() > 0
    model.eval()
    def sample(conditions):
        return generate_scenarios(model, conditions, year, 3, torch.Generator().manual_seed(42), 4)
    before = torch.random.get_rng_state().clone()
    first = sample(c)
    assert first.shape == (2, 3, 4, 24) and torch.isfinite(first).all()
    torch.testing.assert_close(first, sample(c), rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.allclose(first[:, 0], first[:, 1])
    assert not torch.allclose(first, sample(c + 2))
    # One denoiser processes a four-dimensional vector, not four fitted models.
    x = torch.randn(2, 1, 4, requires_grad=True)
    out = model.diffusion.denoise_fn(x, torch.zeros(2, dtype=torch.long), torch.randn(2, 1, 8))
    out[:, :, 0].sum().backward()
    assert x.grad[:, :, 1:].abs().sum() > 0


@pytest.mark.parametrize("cell_type", ["LSTM", "GRU"])
def test_sample_uses_own_previous_generation_only(tiny_config, monkeypatch, cell_type):
    tiny_config["model"]["cell_type"] = cell_type
    model = ConditionalTimeGrad(tiny_config, 18).eval()
    observed = []
    hook = model.rnn.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
    def fake_sample(cond, generator):
        return torch.full((len(cond), 1, 4), float(len(observed)))
    monkeypatch.setattr(model.diffusion, "sample_with_generator", fake_sample)
    _, c, year = tensors()
    result = model.sample(c, year, torch.Generator().manual_seed(42))
    hook.remove()
    assert len(observed) == 24
    for h, inputs in enumerate(observed):
        torch.testing.assert_close(inputs[:, :, :4], torch.full((2, 1, 4), float(h)))
        torch.testing.assert_close(result[:, :, h], torch.full((2, 4), float(h+1)))


@pytest.mark.parametrize("steps", [0, 501])
def test_reject_embedding_overflow(tiny_config, steps):
    tiny_config["diffusion"]["num_steps"] = steps
    with pytest.raises(ValueError, match="500"):
        ConditionalTimeGrad(tiny_config, 18)

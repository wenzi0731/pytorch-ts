"""Cold-start condition-only adaptation of TimeGrad (joint within-hour diffusion).

No observed energy history is supplied at inference. Lag-1 generated targets
within the same day preserve TimeGrad's autoregressive factorization.
"""
from __future__ import annotations

import torch
from torch import nn

from baseline6.upstream import EpsilonTheta, GaussianDiffusion


class IsolatedGaussianDiffusion(GaussianDiffusion):
    """Same upstream DDPM mean/variance, but with an explicit sampling RNG."""

    @torch.no_grad()
    def sample_with_generator(self, cond, generator):
        shape = (*cond.shape[:-1], self.input_size)
        current = torch.randn(shape, device=cond.device, dtype=cond.dtype, generator=generator)
        for step in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((len(cond),), step, device=cond.device, dtype=torch.long)
            mean, _, log_variance = self.p_mean_variance(current, cond, t, clip_denoised=False)
            # Draw at step zero too, matching upstream's random draw count.
            noise = torch.randn(shape, device=cond.device, dtype=cond.dtype, generator=generator)
            current = mean + (0.5 * log_variance).exp() * noise * (step > 0)
            if not torch.isfinite(current).all():
                raise FloatingPointError(f"Nonfinite TimeGrad DDPM state at diffusion step {step}")
        return current


class ConditionalTimeGrad(nn.Module):
    def __init__(self, config, condition_channels):
        super().__init__()
        model, diffusion = config["model"], config["diffusion"]
        self.condition_channels = int(condition_channels)
        self.num_timesteps = int(diffusion["num_steps"])
        if not 1 <= self.num_timesteps <= 500:
            raise ValueError("Unchanged upstream diffusion embedding requires 1 <= num_steps <= 500")
        if diffusion["schedule"] not in ("linear", "quad"):
            raise ValueError("The predeclared protocol supports linear or quad schedules")
        if not 1e-4 <= float(diffusion["beta_end"]) < 1:
            raise ValueError("Require 1e-4 <= beta_end < 1")
        if model["cell_type"] not in ("LSTM", "GRU"):
            raise ValueError("cell_type must be LSTM or GRU")
        for key in ("num_cells", "num_layers", "conditioning_length", "residual_layers",
                    "residual_channels", "dilation_cycle_length"):
            if int(model[key]) < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= float(model["dropout"]) < 1:
            raise ValueError("dropout must be in [0,1)")
        # Same ordering as upstream: target lags, dimension embeddings, covariates.
        # Weather/calendar plus the existing PV year condition are covariates.
        self.embed = nn.Embedding(4, 1)
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[model["cell_type"]]
        self.rnn = rnn_cls(
            input_size=4 + 4 + self.condition_channels + 1,
            hidden_size=int(model["num_cells"]),
            num_layers=int(model["num_layers"]),
            dropout=float(model["dropout"]) if int(model["num_layers"]) > 1 else 0.0,
            batch_first=True,
        )
        self.proj_dist_args = nn.Linear(int(model["num_cells"]), int(model["conditioning_length"]))
        denoiser = EpsilonTheta(
            target_dim=4, cond_length=int(model["conditioning_length"]),
            residual_layers=int(model["residual_layers"]),
            residual_channels=int(model["residual_channels"]),
            dilation_cycle_length=int(model["dilation_cycle_length"]),
        )
        self.diffusion = IsolatedGaussianDiffusion(
            denoiser, input_size=4, diff_steps=self.num_timesteps, loss_type="l2",
            beta_end=float(diffusion["beta_end"]), beta_schedule=diffusion["schedule"],
        )
        # Dataset train-only Z-score replaces upstream context-based MeanScaler.
        # No per-test-day target scale is computed.
        if not all(torch.isfinite(buffer).all() for buffer in self.diffusion.buffers()):
            raise ValueError("Nonfinite diffusion schedule buffers")

    def covariates(self, conditions, year):
        if conditions.ndim != 3 or conditions.shape[1:] != (self.condition_channels, 24):
            raise ValueError("conditions must have shape [day, condition_channels, 24]")
        if year.shape != (len(conditions), 1, 24):
            raise ValueError("year must have shape [day,1,24]")
        if not torch.isfinite(conditions).all() or not torch.isfinite(year).all():
            raise ValueError("Nonfinite conditions")
        return torch.cat((conditions, year), dim=1).transpose(1, 2)

    def rnn_input(self, previous, covariates):
        embedding = self.embed(torch.arange(4, device=previous.device)).reshape(1, 1, 4)
        embedding = embedding.expand(len(previous), previous.shape[1], -1)
        return torch.cat((previous, embedding, covariates), dim=-1)

    def teacher_forced_condition(self, target, conditions, year):
        """Only y[:hour] enters h_hour, never y_hour or later targets."""
        covariates = self.covariates(conditions, year)
        if target.shape != (len(conditions), 4, 24) or not torch.isfinite(target).all():
            raise ValueError("Require finite targets [day,4,24]")
        trajectory = target.transpose(1, 2)
        previous = torch.cat((torch.zeros_like(trajectory[:, :1]), trajectory[:, :-1]), dim=1)
        output, _ = self.rnn(self.rnn_input(previous, covariates))
        return self.proj_dist_args(output)

    def forward(self, target, conditions, year):
        cond = self.teacher_forced_condition(target, conditions, year)
        return self.diffusion.log_prob(target.transpose(1, 2), cond)

    @torch.no_grad()
    def sample(self, conditions, year, generator):
        if self.training:
            raise RuntimeError("Call model.eval() before sampling")
        covariates = self.covariates(conditions, year)
        previous = torch.zeros((len(conditions), 1, 4), device=conditions.device, dtype=conditions.dtype)
        state, samples = None, []
        for hour in range(24):
            output, state = self.rnn(self.rnn_input(previous, covariates[:, hour:hour+1]), state)
            cond = self.proj_dist_args(output)
            previous = self.diffusion.sample_with_generator(cond, generator)
            samples.append(previous)
        return torch.cat(samples, dim=1).transpose(1, 2)


@torch.no_grad()
def generate_scenarios(model, conditions, year, scenarios, generator, chunk_size=32):
    if scenarios < 1 or chunk_size < 1 or len(conditions) < 1:
        raise ValueError("scenarios, chunk_size and batch size must be positive")
    batches = []
    for start in range(0, len(conditions) * scenarios, chunk_size):
        indices = torch.arange(start, min(start + chunk_size, len(conditions) * scenarios),
                               device=conditions.device) // scenarios
        batches.append(model.sample(conditions[indices], year[indices], generator))
    return torch.cat(batches).reshape(len(conditions), scenarios, 4, 24)

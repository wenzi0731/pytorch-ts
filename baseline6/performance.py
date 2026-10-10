"""Execution-only controls: do not change the saved experimental protocol."""
import json
import os
import time

import torch
import torch.nn.functional as F
import math


def env_bool(name, default):
    value = os.environ.get(name, str(int(default))).lower()
    if value not in ("0", "1", "true", "false"):
        raise ValueError(f"{name} must be 0 or 1")
    return value in ("1", "true")


def configure_threads():
    value = os.environ.get("TIMEGRAD_CPU_THREADS")
    if value is not None:
        count = int(value)
        if count < 1:
            raise ValueError("TIMEGRAD_CPU_THREADS must be positive")
        torch.set_num_threads(count)


def execution_info(model):
    parameter = next(model.parameters())
    return {
        "actual_model_device": str(parameter.device),
        "gpu_name": torch.cuda.get_device_name(parameter.device) if parameter.is_cuda else None,
        "torch_threads": torch.get_num_threads(),
        "debug_sampling_checks": model.diffusion.debug_checks,
        "cache_conditioner": model.diffusion.cache_conditioner,
        "sampling_implementation": "cached_condition_v1",
    }


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Progress:
    """Wall-clock-throttled logging; never inspect tensors or consume RNG."""
    def __init__(self, event, **context):
        self.event, self.context = event, context
        self.started = time.perf_counter()
        self.last = self.started
        self.enabled = env_bool("TIMEGRAD_PROGRESS", True)

    def emit(self, force=False, **fields):
        now = time.perf_counter()
        if self.enabled and (force or now - self.last >= 30):
            print(json.dumps({"event": self.event, **self.context, **fields,
                              "elapsed_seconds": round(now - self.started, 3)}), flush=True)
            self.last = now


def condition_cache(denoiser, cond):
    """Constant within one hourly DDPM chain; discarded before the next hour."""
    up = denoiser.cond_upsampler(cond)
    return [layer.conditioner_projection(up) for layer in denoiser.residual_layers]


def cached_denoise(denoiser, inputs, steps, projected_conditions):
    """Same upstream operations/weights/order; skip only invariant projections.

    Sampling-only, with no persistent buffers, detached weights or state_dict
    changes. Training always calls the original EpsilonTheta.forward.
    """
    x = F.leaky_relu(denoiser.input_projection(inputs), 0.4)
    embedding = denoiser.diffusion_embedding(steps)
    skips = []
    for layer, condition in zip(denoiser.residual_layers, projected_conditions):
        diffusion_step = layer.diffusion_projection(embedding).unsqueeze(-1)
        y = layer.dilated_conv(x + diffusion_step) + condition
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = F.leaky_relu(layer.output_projection(y), 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = (x + residual) / math.sqrt(2.0)
        skips.append(skip)
    x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(denoiser.residual_layers))
    x = F.leaky_relu(denoiser.skip_projection(x), 0.4)
    return denoiser.output_projection(x)

"""Compare reference and optimized sampling on the same synthetic conditions.

Read-only: optionally loads an existing checkpoint; no CSV, training or output
files. This diagnoses sampler speed, not model accuracy or the full pipeline.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time

import torch

from baseline6.model import ConditionalTimeGrad, generate_scenarios
from baseline6.performance import execution_info, synchronize
from baseline6.runtime import choose_device, load_checkpoint, load_config, seed_everything


@torch.no_grad()
def benchmark(model, conditions, year, scenarios=2, chunk_size=128, repeats=3, seed=42):
    if min(scenarios, chunk_size, repeats) < 1:
        raise ValueError("Counts must be positive")
    model.eval()
    device = conditions.device
    original = (model.diffusion.cache_conditioner, model.diffusion.debug_checks)
    times = {"reference": [], "optimized": []}
    max_error, rng_equal, exact = 0.0, True, True
    within_tolerance = True
    try:
        # Warm up both modes. Alternate order of measured modes to reduce bias.
        for repeat in range(-1, repeats):
            outputs, states = {}, {}
            order = ("reference", "optimized") if repeat % 2 else ("optimized", "reference")
            for mode in order:
                model.diffusion.cache_conditioner = mode == "optimized"
                model.diffusion.debug_checks = mode == "reference"
                rng = torch.Generator(device=device).manual_seed(seed + 90000)
                synchronize(device)
                started = time.perf_counter()
                output = generate_scenarios(model, conditions, year, scenarios, rng, chunk_size)
                synchronize(device)
                seconds = time.perf_counter() - started
                if repeat >= 0:
                    times[mode].append(seconds)
                outputs[mode], states[mode] = output, rng.get_state()
                print(json.dumps({"event": "benchmark", "repeat": repeat, "mode": mode,
                                  "seconds": seconds, "warmup": repeat < 0}), flush=True)
            reference, optimized = outputs["reference"], outputs["optimized"]
            max_error = max(max_error, (reference - optimized).abs().max().item())
            exact = exact and torch.equal(reference, optimized)
            within_tolerance = within_tolerance and torch.allclose(reference, optimized, rtol=1e-5, atol=1e-6)
            rng_equal = rng_equal and torch.equal(states["reference"], states["optimized"])
    finally:
        model.diffusion.cache_conditioner, model.diffusion.debug_checks = original
    medians = {key: statistics.median(value) for key, value in times.items()}
    result = {**execution_info(model), "batch_size": len(conditions), "scenarios": scenarios,
              "chunk_size": chunk_size, "T": model.num_timesteps, "repeats": repeats,
              "seconds_median": medians,
              "sampler_speedup": medians["reference"] / medians["optimized"],
              "max_abs_difference": max_error, "bitwise_equal": exact,
              "within_tolerance": bool(within_tolerance), "rng_state_equal": rng_equal,
              "comparison": "reference DDPM + per-step checks vs cached DDPM + hourly checks; both use flattened RNN"}
    print(json.dumps(result, indent=2), flush=True)
    if not rng_equal or not within_tolerance:
        raise RuntimeError("Sampling parity failed; do not use accelerated results without investigation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default="baseline6/configs/heew.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.scenarios, args.chunk_size, args.repeats, args.threads) < 1:
        parser.error("Counts must be positive")
    torch.set_num_threads(args.threads)
    seed_everything(42)
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if checkpoint and checkpoint.get("method") != "conditional-timegrad-baseline6":
        raise ValueError("Expected a baseline6 checkpoint")
    config = checkpoint["config"] if checkpoint else load_config(args.config)
    channels = int(checkpoint["condition_channels"]) if checkpoint else (
        18 if config["data"]["weather_feature_set"] == "pv10" else 14)
    device = choose_device(args.device)
    model = ConditionalTimeGrad(config, channels)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Nonzero output weights exercise all condition paths before training.
        torch.nn.init.normal_(model.diffusion.denoise_fn.output_projection.weight, std=0.01)
    model.to(device).eval()
    rng = torch.Generator().manual_seed(2024)
    conditions = torch.randn(args.batch_size, channels, 24, generator=rng).to(device)
    year = torch.zeros(args.batch_size, 1, 24, device=device)
    benchmark(model, conditions, year, args.scenarios, args.chunk_size, args.repeats)


if __name__ == "__main__":
    main()

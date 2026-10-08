"""Load unchanged, dependency-free upstream modules without importing GluonTS."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "7860c9693d55b5c086867477cc33c89485ed0167"


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


epsilon = load_source("_baseline6_upstream_epsilon", "pts/model/time_grad/epsilon_theta.py")
gaussian = load_source("_baseline6_upstream_gaussian", "pts/modules/gaussian_diffusion.py")
EpsilonTheta = epsilon.EpsilonTheta
GaussianDiffusion = gaussian.GaussianDiffusion

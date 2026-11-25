from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_BACKENDS = ("fast", "portable")


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true",
                     help="run correctness cases with every seed and pattern length")


@pytest.fixture(scope="session")
def full(request):
    return request.config.getoption("--full")


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    import build
    build.build()
    import residual_kv
    return residual_kv.best_device()


@pytest.fixture(params=ALL_BACKENDS)
def backend(request, device):
    import residual_kv
    if request.param not in residual_kv.available_backends(device):
        name = torch.cuda.get_device_name(device)
        pytest.skip(f"{request.param} backend not supported on {name}")
    return request.param

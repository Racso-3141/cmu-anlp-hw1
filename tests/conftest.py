import pytest
import torch

from src.paths import PRETRAINED_LLAMA

# Reference artifacts live alongside the tests, not in the repo-wide data/ dir,
# so that data/ can be .gitignored for large downloads without losing them.
from pathlib import Path

TEST_DATA = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def pretrained_llama():
    """The 42M TinyStories Llama. Skips (rather than fails) when not downloaded,
    so unit tests that don't need it still run on a fresh clone."""
    if not PRETRAINED_LLAMA.exists():
        pytest.skip(f"{PRETRAINED_LLAMA} not present — run `bash setup.sh`")
    from src.llama import load_pretrained

    model = load_pretrained(str(PRETRAINED_LLAMA))
    model.eval()
    return model


@pytest.fixture(scope="session")
def sanity_reference():
    return torch.load(TEST_DATA / "sanity_check.data", weights_only=False)

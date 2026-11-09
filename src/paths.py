"""Repository-relative paths.

Resolved against the repo root rather than the current working directory, so
every script behaves the same no matter where you run it from.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "configs"

TOKENIZER_MODEL = DATA_DIR / "tokenizer.model"
PRETRAINED_LLAMA = DATA_DIR / "stories42M.pt"
TINYSTORIES_TOKENS = DATA_DIR / "tinystories-tokens.npz"


def require(path: Path, hint: str = "") -> Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing.\n"
            + (f"  {hint}\n" if hint else "")
            + "  Run `bash setup.sh` to download the required data files."
        )
    return path

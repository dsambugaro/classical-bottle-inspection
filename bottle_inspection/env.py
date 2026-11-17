"""Where the dataset is: the BOTTLE_DATA environment variable, or `code/.env`.

The dataset folder is the one downloaded from Kaggle (Water Bottle Defect-Level Detection),
with `images/` and `labels/` inside. Its path comes, in this order, from:

1. the `--data` argument of the scripts;
2. the BOTTLE_DATA environment variable;
3. the BOTTLE_DATA line of `code/.env` (copy `.env.example` to `.env`);
4. the default `../archive`, next to the `code/` folder.

A relative path is relative to the `code/` folder.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_DIR: Path = Path(__file__).resolve().parent.parent
"""The `code/` folder."""

DATA_ENV: str = "BOTTLE_DATA"
"""Environment variable with the path of the dataset folder."""

DEFAULT_DATA: str = "../archive"
"""Dataset folder used when BOTTLE_DATA is not set (relative to `code/`)."""

ENV_FILE: Path = CODE_DIR / ".env"
"""Optional file with KEY=VALUE lines; the real environment variables take precedence."""


def load_env_file(path: Path = ENV_FILE) -> None:
    """Copies the KEY=VALUE lines of `path` into os.environ, without overwriting a variable."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def data_path_setting() -> str:
    """The dataset path as written in BOTTLE_DATA (or the default), not resolved."""
    load_env_file()
    return os.environ.get(DATA_ENV) or DEFAULT_DATA


def default_data_dir() -> Path:
    """The dataset folder, resolved against the `code/` folder when relative."""
    path = Path(data_path_setting()).expanduser()
    return path if path.is_absolute() else (CODE_DIR / path).resolve()

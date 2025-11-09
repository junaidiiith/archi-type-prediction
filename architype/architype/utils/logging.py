"""Logging helpers and shared prompt-cache directory management."""

from __future__ import annotations

import logging
from pathlib import Path

PROMPTS_DATA_DIR = Path(".tmp/prompts")


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured for file-based auditing.

    Notes
    -----
    The logging configuration is idempotent; subsequent calls reuse the
    existing root configuration.
    """

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("architype.log"),
                # Uncomment to mirror logs to stdout during development.
                # logging.StreamHandler(),
            ],
        )

    return logging.getLogger(name)


def get_prompts_data_dir() -> str:
    """
    Ensure the on-disk cache for prompt/response pairs exists and return it.
    """

    PROMPTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(PROMPTS_DATA_DIR.resolve())

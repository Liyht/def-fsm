"""Resolution of machine-specific paths.

Everything in this repository refers to corpora, base models and output
directories through ``${NAME}`` placeholders instead of absolute paths. This
module turns those placeholders into real paths.

Values are looked up in decreasing order of priority:

1. the process environment,
2. ``configs/paths.yaml`` (copy ``configs/paths.example.yaml`` and edit it),
3. ``.env`` at the project root (copy ``.env.example``),

plus ``PROJECT_ROOT``, which is always available and points at the repository
root, so a config can say ``${PROJECT_ROOT}/prompts/fsm_assistant.txt``.

Typical use::

    from def_fsm.paths import expand, expand_config

    prompt = expand("${PROJECT_ROOT}/prompts/fsm_assistant.txt")
    cfg = expand_config(yaml.safe_load(open("configs/base.yaml")))
"""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"
EXAMPLE_PATHS_FILE = PROJECT_ROOT / "configs" / "paths.example.yaml"
ENV_FILE = PROJECT_ROOT / ".env"

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_cache = None


def load_paths(refresh=False):
    """Return the mapping used to expand ``${NAME}`` placeholders."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

    values = {}
    if PATHS_FILE.exists():
        with open(PATHS_FILE, "r") as f:
            values.update({k: str(v) for k, v in (yaml.safe_load(f) or {}).items()})

    # The environment wins over paths.yaml, and PROJECT_ROOT is not overridable.
    for key in list(values) + _known_names():
        if os.environ.get(key):
            values[key] = os.environ[key]
    values["PROJECT_ROOT"] = str(PROJECT_ROOT)

    _cache = values
    return _cache


def _known_names():
    """Names documented in paths.example.yaml, so the environment alone suffices."""
    if not EXAMPLE_PATHS_FILE.exists():
        return []
    with open(EXAMPLE_PATHS_FILE, "r") as f:
        return list((yaml.safe_load(f) or {}).keys())


def expand(value):
    """Expand every ``${NAME}`` in a string. Non-strings are returned unchanged."""
    if not isinstance(value, str) or "${" not in value:
        return value

    values = load_paths()

    def _sub(match):
        name = match.group(1)
        if name not in values:
            raise KeyError(
                f"'{name}' is not defined. Set it in configs/paths.yaml (copy "
                f"configs/paths.example.yaml) or export it as an environment variable."
            )
        return values[name]

    return _PLACEHOLDER.sub(_sub, value)


def expand_config(cfg):
    """Recursively expand placeholders in a nested dict / list / string."""
    if isinstance(cfg, dict):
        return {k: expand_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [expand_config(v) for v in cfg]
    return expand(cfg)


def require(name):
    """Return a single path value, raising a helpful error when it is missing."""
    values = load_paths()
    if name not in values or not values[name]:
        raise KeyError(
            f"'{name}' is not defined. Set it in configs/paths.yaml (copy "
            f"configs/paths.example.yaml) or export it as an environment variable."
        )
    return values[name]

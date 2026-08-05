"""Load API keys from Sinclair secrets (or Mac mirror) into os.environ.

Neutral / agent-safe: stdlib only. Use from agent env or dl env.

Sources under secrets_dir():
  openai/auth.json  → OPENAI_API_KEY
  langsmith.env     → LANGSMITH_* (canonical modern)
  tavily.env        → TAVILY_API_KEY

Legacy LANGCHAIN_* aliases are set at load for old notebooks only.
Existing environment variables win (setdefault).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

LANGSMITH_ENDPOINT_DEFAULT = "https://api.smith.langchain.com"

# Modern keys first; LANGCHAIN_* are legacy aliases only.
DOTENV_KEYS = (
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "TAVILY_API_KEY",
    # legacy (derived)
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
)


def secrets_dir() -> Path:  #AI
    """Resolve canonical Sinclair secrets or refreshed Mac mirror."""
    sinclair_secrets = Path("/s/agent_rw/secrets")
    if sinclair_secrets.is_dir():
        return sinclair_secrets

    refresh = Path.home() / "code/agent/scripts/mac_pull_agent_rw.sh"
    subprocess.run([refresh, "--if-stale"], check=True)
    if "AGENT_RW_LOCAL" in os.environ:
        local_root = Path(os.environ["AGENT_RW_LOCAL"])
    else:
        local_root = Path.home() / "agent_rw"
    result = local_root / "secrets"
    return result


def read_env(path: Path) -> dict[str, str]:  #AI
    """Read shell-style KEY=VALUE entries from one env file."""
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        values[name] = value.strip().strip('"').strip("'")
    return values


def apply_langsmith_defaults(environ: dict[str, str] | None = None) -> None:  #AI
    """Ensure LANGSMITH_ENDPOINT + legacy LANGCHAIN_* aliases."""
    env = os.environ if environ is None else environ
    env.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT_DEFAULT)
    env.setdefault("LANGSMITH_TRACING", "true")
    if "LANGSMITH_API_KEY" in env:
        env.setdefault("LANGCHAIN_API_KEY", env["LANGSMITH_API_KEY"])
    if "LANGSMITH_TRACING" in env:
        env.setdefault("LANGCHAIN_TRACING_V2", env["LANGSMITH_TRACING"])
    env.setdefault("LANGCHAIN_ENDPOINT", env["LANGSMITH_ENDPOINT"])
    if "LANGSMITH_PROJECT" in env:
        env.setdefault("LANGCHAIN_PROJECT", env["LANGSMITH_PROJECT"])


# Back-compat name used by earlier callers / re-exports
apply_langchain_aliases = apply_langsmith_defaults


def load_api_keys(langsmith_project: str | None = None) -> dict[str, str]:  #AI
    """Load OpenAI, LangSmith, and Tavily into os.environ; return loaded map.

    Canonical env names are LANGSMITH_*. Does not overwrite keys already set.
    If langsmith_project is set, assigns LANGSMITH_PROJECT (and LANGCHAIN_PROJECT alias).
    """
    base = secrets_dir()
    openai_auth = json.loads((base / "openai/auth.json").read_text())
    loaded: dict[str, str] = {}
    loaded.update(read_env(base / "langsmith.env"))
    loaded.update(read_env(base / "tavily.env"))
    loaded["OPENAI_API_KEY"] = openai_auth["OPENAI_API_KEY"]

    for name, value in loaded.items():
        os.environ.setdefault(name, value)

    apply_langsmith_defaults()
    if langsmith_project is not None:
        os.environ["LANGSMITH_PROJECT"] = langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        loaded["LANGSMITH_PROJECT"] = langsmith_project
    return loaded


def write_dotenv(path: Path, langsmith_project: str | None = None) -> Path:  #AI
    """Write loaded keys to a .env file (e.g. LangGraph studio/)."""
    load_api_keys(langsmith_project=langsmith_project)
    lines: list[str] = []
    for name in DOTENV_KEYS:
        if name in os.environ:
            lines.append(f"{name}={os.environ[name]}")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path

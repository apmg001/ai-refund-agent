"""Application configuration.

Settings are loaded from environment variables (optionally a ``.env`` file) with
sensible defaults so the system runs with zero configuration. Centralizing config in
one validated object keeps magic strings out of the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError

__all__ = ["Settings", "get_settings"]

# Project root = three levels up from this file: src/refund_agent/config.py -> repo root
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"


class Settings(BaseSettings):
    """Validated runtime configuration.

    All fields can be overridden via environment variables prefixed with
    ``REFUND_AGENT_`` (e.g. ``REFUND_AGENT_LLM_PROVIDER=heuristic``).
    """

    model_config = SettingsConfigDict(
        env_prefix="REFUND_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application -------------------------------------------------------- #
    app_name: str = "ShopSphere Refund Agent"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Data --------------------------------------------------------------- #
    data_dir: Path = _DEFAULT_DATA_DIR
    crm_filename: str = "crm_database.json"
    policy_rules_filename: str = "refund_policy_rules.json"
    policy_doc_filename: str = "refund_policy.md"

    # --- LLM ---------------------------------------------------------------- #
    # "auto"      -> use Ollama if reachable, otherwise fall back to heuristic
    # "ollama"    -> force the local open-source model (errors if unreachable)
    # "heuristic" -> force the deterministic, dependency-free planner
    llm_provider: str = "auto"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: float = 60.0

    # --- Agent -------------------------------------------------------------- #
    agent_max_iterations: int = Field(default=8, ge=1, le=25)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ConfigurationError(
                f"Invalid log_level '{value}'. Expected one of {sorted(allowed)}."
            )
        return upper

    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        allowed = {"auto", "ollama", "heuristic"}
        lower = value.lower()
        if lower not in allowed:
            raise ConfigurationError(
                f"Invalid llm_provider '{value}'. Expected one of {sorted(allowed)}."
            )
        return lower

    # --- Derived paths ------------------------------------------------------ #
    @property
    def crm_path(self) -> Path:
        """Absolute path to the CRM database file."""
        return self.data_dir / self.crm_filename

    @property
    def policy_rules_path(self) -> Path:
        """Absolute path to the machine-readable policy rules file."""
        return self.data_dir / self.policy_rules_filename

    @property
    def policy_doc_path(self) -> Path:
        """Absolute path to the human-readable policy document."""
        return self.data_dir / self.policy_doc_filename


def get_settings() -> Settings:
    """Construct and return a :class:`Settings` instance.

    Returns:
        The validated settings object.

    Raises:
        ConfigurationError: If configuration validation fails.
    """
    try:
        return Settings()
    except ConfigurationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Failed to load settings: {exc}") from exc

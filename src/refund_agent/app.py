"""Application composition root.

Wires the object graph together in one place (a tiny dependency-injection container)
so the CLI, API, and tests all construct the system the same way. Centralizing
construction keeps wiring out of the entry points and makes it trivial to substitute
a component (e.g. a different clock or LLM provider) for testing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent.agent_loop import RefundAgent
from .agent.tool_registry import ToolRegistry
from .agent.tools import build_default_tools
from .config import Settings, get_settings
from .llm.base import LLMProvider
from .llm.factory import build_llm_provider
from .logging_config import configure_logging, get_logger
from .observability.reasoning_log import ReasoningLog
from .repositories.crm_repository import CrmRepository
from .repositories.policy_repository import PolicyRepository
from .services.policy_engine import PolicyEngine
from .services.refund_service import RefundService
from .utils.clock import Clock, SystemClock

__all__ = ["Application", "build_application"]

_logger = get_logger(__name__)


@dataclass
class Application:
    """Container holding the fully-wired application components.

    Attributes:
        settings: Effective configuration.
        crm: The CRM repository.
        policy_repo: The policy repository.
        policy_engine: The deterministic policy engine.
        refund_service: The refund orchestration service.
        reasoning_log: The shared reasoning/observability log.
        llm: The selected LLM provider.
        agent: The function-calling refund agent.
    """

    settings: Settings
    crm: CrmRepository
    policy_repo: PolicyRepository
    policy_engine: PolicyEngine
    refund_service: RefundService
    reasoning_log: ReasoningLog
    llm: LLMProvider
    agent: RefundAgent


def build_application(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
    configure_logs: bool = True,
) -> Application:
    """Construct and wire the entire application object graph.

    Args:
        settings: Optional settings override. Loaded from the environment if omitted.
        clock: Optional clock override (used by tests for determinism).
        configure_logs: Whether to (idempotently) configure logging.

    Returns:
        A fully-wired :class:`Application`.

    Raises:
        DataLoadError: If the CRM or policy files cannot be loaded.
        ConfigurationError: If settings are invalid.
    """
    settings = settings or get_settings()
    if configure_logs:
        configure_logging(settings.log_level)

    clock = clock or SystemClock()
    _logger.info("Bootstrapping %s (env=%s)", settings.app_name, settings.environment)

    crm = CrmRepository(settings.crm_path, clock=clock)
    policy_repo = PolicyRepository(settings.policy_rules_path, settings.policy_doc_path)
    policy_engine = PolicyEngine(policy_repo.rules, clock=clock)
    refund_service = RefundService(crm, policy_engine, clock=clock)

    reasoning_log = ReasoningLog()
    llm = build_llm_provider(settings)

    registry = ToolRegistry(build_default_tools(crm, refund_service, policy_repo))
    agent = RefundAgent(
        llm=llm,
        tools=registry,
        reasoning_log=reasoning_log,
        max_iterations=settings.agent_max_iterations,
    )

    _logger.info("Application ready (llm=%s)", getattr(llm, "name", "unknown"))
    return Application(
        settings=settings,
        crm=crm,
        policy_repo=policy_repo,
        policy_engine=policy_engine,
        refund_service=refund_service,
        reasoning_log=reasoning_log,
        llm=llm,
        agent=agent,
    )

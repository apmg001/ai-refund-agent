"""Shared pytest fixtures.

Provides deterministically-wired components for the test suite. A
:class:`FrozenClock` is used everywhere so time-dependent rules (the return window)
behave identically regardless of when the tests run. The reference "now" is chosen to
match the relative-date design of the mock CRM data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from refund_agent.agent.agent_loop import RefundAgent
from refund_agent.agent.tool_registry import ToolRegistry
from refund_agent.agent.tools import build_default_tools
from refund_agent.llm.heuristic_provider import HeuristicProvider
from refund_agent.observability.reasoning_log import ReasoningLog
from refund_agent.repositories.crm_repository import CrmRepository
from refund_agent.repositories.policy_repository import PolicyRepository
from refund_agent.services.policy_engine import PolicyEngine
from refund_agent.services.refund_service import RefundService
from refund_agent.utils.clock import FrozenClock

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# A fixed reference instant for deterministic return-window calculations.
_FROZEN_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def clock() -> FrozenClock:
    """Return a frozen clock pinned to the reference instant."""
    return FrozenClock(_FROZEN_NOW)


@pytest.fixture()
def crm(clock: FrozenClock) -> CrmRepository:
    """Return a CRM repository loaded from the project mock data."""
    return CrmRepository(_DATA_DIR / "crm_database.json", clock=clock)


@pytest.fixture()
def policy_repo() -> PolicyRepository:
    """Return a policy repository loaded from the project mock data."""
    return PolicyRepository(
        _DATA_DIR / "refund_policy_rules.json",
        _DATA_DIR / "refund_policy.md",
    )


@pytest.fixture()
def policy_engine(policy_repo: PolicyRepository, clock: FrozenClock) -> PolicyEngine:
    """Return a policy engine wired with the frozen clock."""
    return PolicyEngine(policy_repo.rules, clock=clock)


@pytest.fixture()
def refund_service(
    crm: CrmRepository, policy_engine: PolicyEngine, clock: FrozenClock
) -> RefundService:
    """Return a refund service wired with the frozen clock."""
    return RefundService(crm, policy_engine, clock=clock)


@pytest.fixture()
def agent(
    crm: CrmRepository,
    refund_service: RefundService,
    policy_repo: PolicyRepository,
) -> RefundAgent:
    """Return a refund agent driven by the deterministic heuristic provider."""
    registry = ToolRegistry(build_default_tools(crm, refund_service, policy_repo))
    return RefundAgent(
        llm=HeuristicProvider(),
        tools=registry,
        reasoning_log=ReasoningLog(),
        max_iterations=8,
    )

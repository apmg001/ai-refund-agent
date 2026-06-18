"""Agent loop and refund-service tests.

End-to-end checks that the function-calling loop (driven by the deterministic
heuristic provider) reaches correct outcomes, records a reasoning trace, and that
the refund service enforces policy and idempotency.
"""

from __future__ import annotations

from decimal import Decimal

from refund_agent.agent.agent_loop import RefundAgent
from refund_agent.agent.tool_registry import ToolRegistry
from refund_agent.agent.tools import build_default_tools
from refund_agent.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec
from refund_agent.models.domain import RefundRequest
from refund_agent.models.enums import DecisionType, OrderStatus, RefundReason
from refund_agent.observability.reasoning_log import ReasoningLog
from refund_agent.services.refund_service import RefundService


# ---------------------------------------------------------------------- #
# Agent loop
# ---------------------------------------------------------------------- #
def test_agent_approves_and_processes(agent: RefundAgent) -> None:
    """An eligible request is approved, processed, and confirmed to the customer."""
    result = agent.run("My headphones ORD-10001 are defective, please refund.")
    assert "process_refund" in result.tool_calls
    assert "approved" in result.reply.lower()
    assert "RFND-" in result.reply


def test_agent_denies_out_of_window(agent: RefundAgent) -> None:
    """An out-of-window request is denied and not processed."""
    result = agent.run("Refund ORD-10002, I changed my mind.")
    assert "process_refund" not in result.tool_calls
    assert "unable to approve" in result.reply.lower()


def test_agent_escalates(agent: RefundAgent) -> None:
    """A high-value/fraud request is escalated to a human."""
    result = agent.run("Refund my bag ORD-10005, it's defective.")
    assert "process_refund" not in result.tool_calls
    assert "review" in result.reply.lower()


def test_agent_asks_for_order_id(agent: RefundAgent) -> None:
    """With no order id, the agent asks for one instead of calling tools."""
    result = agent.run("I want a refund please.")
    assert result.tool_calls == ()
    assert "order" in result.reply.lower()


def test_agent_reports_unknown_order(agent: RefundAgent) -> None:
    """A well-formed but non-existent order id is reported as not found."""
    result = agent.run("Refund my order ORD-99999, it's defective.")
    assert "lookup_order" in result.tool_calls
    assert "process_refund" not in result.tool_calls
    assert "couldn't find" in result.reply.lower() or "could not find" in result.reply.lower()


def test_agent_records_reasoning_trace(agent: RefundAgent) -> None:
    """The reasoning log captures the full trace for the session."""
    result = agent.run("ORD-10001 is defective, refund please.")
    events = agent._log.get_events(result.session_id)  # noqa: SLF001 - test introspection
    types = {e.event_type.value for e in events}
    assert "USER_MESSAGE" in types
    assert "TOOL_CALL" in types
    assert "DECISION" in types
    assert "AGENT_RESPONSE" in types


# ---------------------------------------------------------------------- #
# Refund service
# ---------------------------------------------------------------------- #
def test_service_executes_only_when_approved(refund_service: RefundService) -> None:
    """The service moves money for approvals but not for denials."""
    approved = refund_service.process(
        RefundRequest(order_id="ORD-10001", reason=RefundReason.DEFECTIVE)
    )
    assert approved.executed
    assert approved.receipt is not None

    denied = refund_service.process(
        RefundRequest(order_id="ORD-10002", reason=RefundReason.CHANGED_MIND)
    )
    assert not denied.executed
    assert denied.decision.decision is DecisionType.DENIED


def test_repeat_full_refund_is_denied(refund_service: RefundService) -> None:
    """A second refund attempt on a fully-refunded order is denied, not re-executed."""
    request = RefundRequest(order_id="ORD-10012", reason=RefundReason.WRONG_ITEM)
    first = refund_service.process(request)
    assert first.executed and first.receipt is not None

    second = refund_service.process(request)
    assert not second.executed
    assert second.receipt is None
    assert second.decision.decision is DecisionType.DENIED


def test_partial_then_remaining_tops_up(refund_service: RefundService) -> None:
    """A non-fault partial refund, then a later fault claim, refunds the remainder.

    Reproduces the multi-refund sequence: ORD-10001 ($129.99 electronics) is first
    refunded for a non-fault reason (15% fee withheld), then a defective claim recovers
    exactly the withheld remainder — with a *new* transaction id, never re-reporting the
    first, and never exceeding the order total.
    """
    first = refund_service.process(
        RefundRequest(order_id="ORD-10001", reason=RefundReason.CHANGED_MIND)
    )
    assert first.decision.decision is DecisionType.PARTIALLY_APPROVED
    assert first.receipt is not None
    assert first.receipt.refunded_amount == Decimal("110.49")  # 129.99 less 15% fee

    second = refund_service.process(
        RefundRequest(order_id="ORD-10001", reason=RefundReason.DEFECTIVE)
    )
    assert second.decision.decision is DecisionType.APPROVED
    assert second.receipt is not None
    assert second.receipt.refunded_amount == Decimal("19.50")  # the withheld remainder
    assert second.receipt.transaction_id != first.receipt.transaction_id

    order = refund_service._crm.get_order("ORD-10001")  # noqa: SLF001 - test introspection
    assert order.status is OrderStatus.REFUNDED
    assert order.amount_already_refunded == Decimal("129.99")  # never exceeds the total


def test_service_updates_order_status(refund_service: RefundService) -> None:
    """A full refund marks the order REFUNDED in the CRM overlay."""
    refund_service.process(RefundRequest(order_id="ORD-10001", reason=RefundReason.DEFECTIVE))
    order = refund_service._crm.get_order("ORD-10001")  # noqa: SLF001 - test introspection
    assert order.status is OrderStatus.REFUNDED
    assert order.amount_already_refunded == Decimal("129.99")


# ---------------------------------------------------------------------- #
# Reconciliation guardrail (financial-fact integrity)
# ---------------------------------------------------------------------- #
class _ScriptedProvider(LLMProvider):
    """A fake provider that plays back a fixed list of responses in order.

    It lets us reproduce a misbehaving model — e.g. one that checks eligibility and
    then fabricates a success message without ever calling ``process_refund``.
    """

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self._i = 0

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _make_agent(crm, refund_service, policy_repo, provider) -> RefundAgent:
    """Wire a RefundAgent around a specific provider for guardrail tests."""
    registry = ToolRegistry(build_default_tools(crm, refund_service, policy_repo))
    return RefundAgent(
        llm=provider,
        tools=registry,
        reasoning_log=ReasoningLog(),
        max_iterations=8,
    )


def test_guardrail_executes_refund_model_skipped(crm, refund_service, policy_repo) -> None:
    """The reported bug: model approves, skips process_refund, claims success.

    The guardrail must execute the refund for real and produce a confirmation that
    carries the genuine transaction id — and the order must actually be refunded.
    """
    provider = _ScriptedProvider([
        LLMResponse(tool_calls=(ToolCall("lookup_order", {"order_id": "ORD-10001"}),)),
        LLMResponse(tool_calls=(ToolCall(
            "check_refund_eligibility", {"order_id": "ORD-10001", "reason": "DEFECTIVE"}
        ),)),
        # Model fabricates success WITHOUT calling process_refund:
        LLMResponse(content="I've processed a full refund of $129.99 to your card."),
    ])
    agent = _make_agent(crm, refund_service, policy_repo, provider)

    result = agent.run("ORD-10001 is broken, refund please.")

    assert "process_refund" in result.tool_calls          # guardrail forced execution
    assert "RFND-" in result.reply                          # real transaction id present
    # And the money actually moved:
    order = crm.get_order("ORD-10001")
    assert order.status is OrderStatus.REFUNDED


def test_guardrail_blocks_fabricated_success_on_denial(crm, refund_service, policy_repo) -> None:
    """Model claims success on a DENIED order; guardrail overrides with the truth."""
    provider = _ScriptedProvider([
        LLMResponse(tool_calls=(ToolCall("lookup_order", {"order_id": "ORD-10004"}),)),
        LLMResponse(tool_calls=(ToolCall(
            "check_refund_eligibility", {"order_id": "ORD-10004", "reason": "NO_LONGER_NEEDED"}
        ),)),
        LLMResponse(content="Your refund of $59.99 has been approved and processed!"),
    ])
    agent = _make_agent(crm, refund_service, policy_repo, provider)

    result = agent.run("Refund my game code ORD-10004.")

    assert "process_refund" not in result.tool_calls
    assert "unable to approve" in result.reply.lower()
    assert "RFND-" not in result.reply
    assert crm.get_order("ORD-10004").status is not OrderStatus.REFUNDED


def test_guardrail_suppresses_claim_with_no_decision(crm, refund_service, policy_repo) -> None:
    """Model claims a refund with no tool calls at all; guardrail suppresses it."""
    provider = _ScriptedProvider([
        LLMResponse(content="Sure! Your refund of $100 has been processed."),
    ])
    agent = _make_agent(crm, refund_service, policy_repo, provider)

    result = agent.run("Give me a refund.")

    assert result.tool_calls == ()
    assert "has been processed" not in result.reply.lower()
    assert "order id" in result.reply.lower()


def test_guardrail_does_not_leak_previous_turn_outcome(crm, refund_service, policy_repo) -> None:
    """A refund in turn 1 must not be re-reported when turn 2 only asks about another order.

    Reproduces the multi-turn session bug: after refunding ORD-10001, a follow-up that
    merely looks up ORD-10014 must not echo the stale ORD-10001 confirmation.
    """
    provider = _ScriptedProvider([
        # Turn 1 — refund ORD-10001 (model fabricates success; guardrail executes it).
        LLMResponse(tool_calls=(ToolCall("lookup_order", {"order_id": "ORD-10001"}),)),
        LLMResponse(tool_calls=(ToolCall(
            "check_refund_eligibility", {"order_id": "ORD-10001", "reason": "DEFECTIVE"}
        ),)),
        LLMResponse(content="I've processed your refund."),
        # Turn 2 — just ask about ORD-10014.
        LLMResponse(tool_calls=(ToolCall("lookup_order", {"order_id": "ORD-10014"}),)),
        LLMResponse(content="Here are the details for ORD-10014: a monitor priced at $200."),
    ])
    agent = _make_agent(crm, refund_service, policy_repo, provider)
    session_id = agent.new_session()

    first = agent.run("Refund ORD-10001, it's defective.", session_id=session_id)
    assert "RFND-" in first.reply  # turn 1 really did refund

    second = agent.run("Tell me about ORD-10014.", session_id=session_id)
    assert "RFND-" not in second.reply          # no stale transaction id leaked
    assert second.reply != first.reply
    assert "process_refund" not in second.tool_calls
    assert "10014" in second.reply
"""Agent tools.

Each tool is a thin, well-typed adapter that exposes a capability of the underlying
domain services to the LLM in a function-calling-friendly shape. Tools are the *only*
way the agent can affect the world, which keeps the surface area small and auditable.

Design notes:
    * Every tool declares its arguments with a pydantic model. The model both
      validates incoming arguments (rejecting anything malformed) and generates the
      JSON schema advertised to the LLM, so the contract is defined exactly once.
    * Tools return plain JSON-serializable ``dict`` objects. They never raise to the
      caller for *expected* domain conditions (e.g. "order not found"); instead they
      return a structured ``{"error": ...}`` payload so the agent can react. Truly
      unexpected failures are wrapped in :class:`ToolExecutionError`.
    * Tools contain no business policy of their own — they delegate to
      :class:`RefundService` / :class:`PolicyEngine` so the deterministic engine stays
      the single source of truth.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from ..exceptions import (
    CustomerNotFoundError,
    InvalidToolArgumentsError,
    OrderNotFoundError,
    RefundAgentError,
    ToolExecutionError,
)
from ..llm.base import ToolSpec
from ..logging_config import get_logger
from ..models.domain import RefundRequest
from ..models.enums import RefundReason
from ..models.reason_classifier import classify_reason
from ..repositories.crm_repository import CrmRepository
from ..repositories.policy_repository import PolicyRepository
from ..services.refund_service import RefundService

__all__ = [
    "Tool",
    "LookupCustomerTool",
    "LookupOrderTool",
    "CheckRefundEligibilityTool",
    "ProcessRefundTool",
    "GetRefundPolicyTool",
    "build_default_tools",
]

_logger = get_logger(__name__)


# ====================================================================== #
# Argument schemas
# ====================================================================== #
class _LookupCustomerArgs(BaseModel):
    """Arguments for :class:`LookupCustomerTool`."""

    customer_id: Optional[str] = None
    email: Optional[str] = None


class _OrderArgs(BaseModel):
    """Arguments for tools that operate on a single order."""

    order_id: str


class _RefundArgs(BaseModel):
    """Arguments for eligibility and processing tools."""

    order_id: str
    reason: Optional[str] = None
    requested_amount: Optional[float] = None


# Alternative argument names a model might emit, mapped to the canonical name. These
# recover values that would otherwise be silently dropped by schema validation.
_REASON_ALIASES = {
    "reason_for_refund": "reason",
    "refund_reason": "reason",
    "reason_text": "reason",
    "reason_code": "reason",
    "reasonForRefund": "reason",
    "refundReason": "reason",
    "reason_description": "reason",
}
_ORDER_ALIASES = {"orderId": "order_id", "order": "order_id", "order_number": "order_id"}
_AMOUNT_ALIASES = {
    "amount": "requested_amount",
    "requestedAmount": "requested_amount",
    "refund_amount": "requested_amount",
}
_REFUND_ARG_ALIASES = {**_REASON_ALIASES, **_ORDER_ALIASES, **_AMOUNT_ALIASES}


class _PolicyArgs(BaseModel):
    """Arguments for :class:`GetRefundPolicyTool`."""

    topic: str


# ====================================================================== #
# Tool base class
# ====================================================================== #
class Tool(abc.ABC):
    """Base class for an agent tool.

    Subclasses declare ``name``, ``description`` and an ``args_model`` (a pydantic
    model). The base class handles argument validation and JSON-schema generation so
    concrete tools only implement :meth:`_run` with already-validated arguments.
    """

    name: str = "tool"
    description: str = ""
    args_model: type[BaseModel] = BaseModel
    # Maps non-canonical argument names a model might emit to the canonical name the
    # schema declares. Recovers values a model would otherwise place under a wrong key
    # (e.g. ``reason_for_refund`` instead of ``reason``) instead of silently dropping
    # them. Populated by subclasses where it matters.
    arg_aliases: dict[str, str] = {}

    def spec(self) -> ToolSpec:
        """Return the :class:`ToolSpec` advertised to the LLM.

        Returns:
            A tool specification whose JSON schema is derived from ``args_model``.
        """
        schema = self.args_model.model_json_schema()
        # Some providers reject schema metadata keys; keep only what they need.
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)

    def _normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Remap known alias keys to canonical names before validation.

        An alias is only applied when the canonical key is not already present, so an
        explicit canonical value always wins. Unknown keys are left untouched (pydantic
        ignores them).

        Args:
            arguments: The raw arguments supplied by the model.

        Returns:
            A new dict with recognized aliases remapped to canonical names.
        """
        if not self.arg_aliases:
            return arguments
        normalized = dict(arguments)
        for alias, canonical in self.arg_aliases.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized.pop(alias)
        return normalized

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate ``arguments`` and run the tool.

        Args:
            arguments: Raw keyword arguments supplied by the LLM.

        Returns:
            A JSON-serializable result dictionary. Expected domain failures are
            returned as ``{"error": ...}`` rather than raised.

        Raises:
            InvalidToolArgumentsError: If ``arguments`` fail schema validation.
            ToolExecutionError: If the tool fails unexpectedly.
        """
        normalized = self._normalize_arguments(arguments)
        try:
            validated = self.args_model.model_validate(normalized)
        except ValidationError as exc:
            raise InvalidToolArgumentsError(
                f"Invalid arguments for tool '{self.name}': {exc.errors()}"
            ) from exc

        try:
            return self._run(validated)
        except RefundAgentError:
            # Domain errors that subclasses choose to raise propagate untouched.
            raise
        except Exception as exc:  # noqa: BLE001 - convert anything unexpected
            raise ToolExecutionError(
                f"Tool '{self.name}' failed to execute: {exc}"
            ) from exc

    @abc.abstractmethod
    def _run(self, args: Any) -> dict[str, Any]:
        """Execute the tool with validated ``args``. Implemented by subclasses."""
        raise NotImplementedError


# ====================================================================== #
# Concrete tools
# ====================================================================== #
class LookupCustomerTool(Tool):
    """Look up a customer by id or email."""

    name = "lookup_customer"
    description = (
        "Look up a customer profile by customer_id or email. Returns the customer's "
        "tier, account status, and a list of their order ids."
    )
    args_model = _LookupCustomerArgs

    def __init__(self, crm: CrmRepository) -> None:
        """Initialize the tool.

        Args:
            crm: The CRM repository used for lookups.
        """
        self._crm = crm

    def _run(self, args: _LookupCustomerArgs) -> dict[str, Any]:
        if not args.customer_id and not args.email:
            return {"error": "Provide either customer_id or email."}
        try:
            customer = (
                self._crm.get_customer(args.customer_id)
                if args.customer_id
                else self._crm.find_customer_by_email(str(args.email))
            )
        except (CustomerNotFoundError,) as exc:
            return {"error": str(exc)}

        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "tier": customer.tier.value,
            "account_status": customer.account_status.value,
            "fraud_flag": customer.fraud_flag,
            "order_ids": [o.order_id for o in customer.orders],
        }


class LookupOrderTool(Tool):
    """Fetch an order and its owning customer's context."""

    name = "lookup_order"
    description = (
        "Look up an order by order_id. Returns product, category, amount, status, "
        "delivery information, and the owning customer's account standing."
    )
    args_model = _OrderArgs

    def __init__(self, crm: CrmRepository) -> None:
        """Initialize the tool.

        Args:
            crm: The CRM repository used for lookups.
        """
        self._crm = crm

    def _run(self, args: _OrderArgs) -> dict[str, Any]:
        try:
            order = self._crm.get_order(args.order_id)
            customer = self._crm.get_customer_for_order(args.order_id)
        except (OrderNotFoundError, CustomerNotFoundError) as exc:
            return {
                "error": str(exc),
                "order_id": args.order_id,
                "found": False,
                "hint": (
                    "Order ids look like 'ORD-' followed by five digits "
                    "(e.g. ORD-10001). Ask the customer to confirm their order number."
                ),
            }

        return {
            "order_id": order.order_id,
            "product_name": order.product_name,
            "category": order.category.value,
            "amount": str(order.amount),
            "currency": "USD",
            "status": order.status.value,
            "order_date": order.order_date.isoformat(),
            "delivery_date": order.delivery_date.isoformat() if order.delivery_date else None,
            "is_final_sale": order.is_final_sale,
            "amount_already_refunded": str(order.amount_already_refunded),
            "remaining_refundable": str(order.remaining_refundable),
            "customer": {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "tier": customer.tier.value,
                "account_status": customer.account_status.value,
                "fraud_flag": customer.fraud_flag,
            },
        }


class CheckRefundEligibilityTool(Tool):
    """Run the deterministic policy engine for an order without moving money."""

    name = "check_refund_eligibility"
    description = (
        "Evaluate whether an order qualifies for a refund under company policy. "
        "Provide order_id and a reason (one of: DEFECTIVE, DAMAGED, WRONG_ITEM, "
        "NOT_AS_DESCRIBED, ARRIVED_LATE, NO_LONGER_NEEDED, CHANGED_MIND, OTHER). "
        "Returns the decision (APPROVED, PARTIALLY_APPROVED, DENIED, ESCALATED), the "
        "approved amount, any restocking fee, and the policy rationale. This does NOT "
        "process the refund."
    )
    args_model = _RefundArgs
    arg_aliases = _REFUND_ARG_ALIASES

    def __init__(self, refund_service: RefundService) -> None:
        """Initialize the tool.

        Args:
            refund_service: The service wrapping the policy engine.
        """
        self._service = refund_service

    def _run(self, args: _RefundArgs) -> dict[str, Any]:
        request = _build_refund_request(args)
        try:
            decision = self._service.evaluate(request)
        except (OrderNotFoundError, CustomerNotFoundError) as exc:
            return {"error": str(exc)}

        return _decision_to_dict(decision)


class ProcessRefundTool(Tool):
    """Execute a refund — guarded by the policy engine (defense in depth)."""

    name = "process_refund"
    description = (
        "Process (execute) a refund for an order. The policy engine is re-run and the "
        "refund is only executed if it is APPROVED or PARTIALLY_APPROVED; otherwise no "
        "money moves. Returns a transaction_id and the refunded amount when executed."
    )
    args_model = _RefundArgs
    arg_aliases = _REFUND_ARG_ALIASES

    def __init__(self, refund_service: RefundService) -> None:
        """Initialize the tool.

        Args:
            refund_service: The service that evaluates and executes refunds.
        """
        self._service = refund_service

    def _run(self, args: _RefundArgs) -> dict[str, Any]:
        request = _build_refund_request(args)
        try:
            outcome = self._service.process(request)
        except (OrderNotFoundError, CustomerNotFoundError) as exc:
            return {"error": str(exc)}

        result = _decision_to_dict(outcome.decision)
        result["executed"] = outcome.executed
        if outcome.receipt is not None:
            result.update(
                {
                    "transaction_id": outcome.receipt.transaction_id,
                    "refunded_amount": str(outcome.receipt.refunded_amount),
                    "processed_at": outcome.receipt.processed_at.isoformat(),
                }
            )
        return result


class GetRefundPolicyTool(Tool):
    """Retrieve the relevant section of the human-readable refund policy."""

    name = "get_refund_policy"
    description = (
        "Retrieve the section of the written refund policy relevant to a topic "
        "(e.g. 'return window', 'restocking fee', 'final sale', 'digital'). Use this "
        "to cite or explain policy to the customer."
    )
    args_model = _PolicyArgs

    def __init__(self, policy_repo: PolicyRepository) -> None:
        """Initialize the tool.

        Args:
            policy_repo: The repository holding the policy document.
        """
        self._policy = policy_repo

    def _run(self, args: _PolicyArgs) -> dict[str, Any]:
        return {"topic": args.topic, "policy": self._policy.search_document(args.topic)}


# ====================================================================== #
# Helpers
# ====================================================================== #
def _build_refund_request(args: _RefundArgs) -> RefundRequest:
    """Construct a :class:`RefundRequest` from validated tool arguments.

    The free-text/label reason is resolved through the deterministic classifier so a
    sloppy or invalid label (e.g. ``NOT_WORKING``) maps to the correct reason
    (``DEFECTIVE``) rather than silently becoming a fee-bearing ``OTHER``. Only a
    genuinely unmappable reason falls back to ``OTHER``.

    Args:
        args: Validated refund tool arguments.

    Returns:
        A normalized :class:`RefundRequest`.
    """
    if args.reason is None:
        _logger.info(
            "No refund reason supplied for %s; treating as OTHER (non-fault: no fee "
            "waiver or extended window).",
            args.order_id,
        )
        reason = RefundReason.OTHER
    else:
        reason = classify_reason(args.reason)
        if reason is None:
            _logger.warning("Could not classify refund reason %r; treating as OTHER", args.reason)
            reason = RefundReason.OTHER

    return RefundRequest(
        order_id=args.order_id,
        reason=reason,
        requested_amount=args.requested_amount,
    )


def _decision_to_dict(decision: Any) -> dict[str, Any]:
    """Serialize a :class:`PolicyDecision` to a JSON-friendly dict for the LLM."""
    return {
        "order_id": decision.order_id,
        "decision": decision.decision.value,
        "approved_amount": str(decision.approved_amount),
        "restocking_fee": str(decision.restocking_fee),
        "currency": decision.currency,
        "rationale": decision.rationale,
        "rule_results": [
            {"rule_id": r.rule_id, "severity": r.severity.value, "message": r.message}
            for r in decision.rule_results
        ],
    }


def build_default_tools(
    crm: CrmRepository,
    refund_service: RefundService,
    policy_repo: PolicyRepository,
) -> list[Tool]:
    """Construct the standard set of agent tools.

    Args:
        crm: The CRM repository.
        refund_service: The refund service.
        policy_repo: The policy repository.

    Returns:
        A list of ready-to-register :class:`Tool` instances.
    """
    return [
        LookupCustomerTool(crm),
        LookupOrderTool(crm),
        CheckRefundEligibilityTool(refund_service),
        ProcessRefundTool(refund_service),
        GetRefundPolicyTool(policy_repo),
    ]
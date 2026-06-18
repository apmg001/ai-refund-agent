"""CRM repository.

Loads the mock CRM JSON into validated domain models and exposes O(1) lookups by
customer id, order id, and email. The repository is the only component that knows
how the data is physically stored; swapping the JSON file for a real database would
mean changing this class alone (separation of concerns).

Order dates in the mock data are stored as relative offsets (``placed_days_ago`` /
``delivered_days_ago``) and resolved to absolute UTC timestamps at load time using
the injected :class:`~refund_agent.utils.clock.Clock`. This keeps demo behaviour
deterministic regardless of the calendar date the system is run.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from ..exceptions import CustomerNotFoundError, DataLoadError, OrderNotFoundError
from ..logging_config import get_logger
from ..models.domain import Customer, Order
from ..models.enums import OrderStatus, ProductCategory
from ..utils.clock import Clock, SystemClock

__all__ = ["CrmRepository"]

_logger = get_logger(__name__)


class CrmRepository:
    """In-memory CRM backed by a JSON file.

    The full dataset is loaded once at construction and indexed for fast lookups.
    Instances are read-only with respect to the source file; refund processing
    mutates an in-memory overlay (see :class:`~refund_agent.services.refund_service`).
    """

    def __init__(self, crm_path: Path, clock: Optional[Clock] = None) -> None:
        """Load and index the CRM database.

        Args:
            crm_path: Path to the CRM JSON file.
            clock: Clock used to resolve relative order dates. Defaults to
                :class:`SystemClock`.

        Raises:
            DataLoadError: If the file is missing, unreadable, or malformed.
        """
        self._clock: Clock = clock or SystemClock()
        self._customers_by_id: dict[str, Customer] = {}
        self._customer_id_by_order: dict[str, str] = {}
        self._customer_id_by_email: dict[str, str] = {}
        self._load(crm_path)
        _logger.info(
            "CRM loaded: %d customers, %d orders",
            len(self._customers_by_id),
            len(self._customer_id_by_order),
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self, crm_path: Path) -> None:
        """Read, parse, and index the CRM file."""
        try:
            raw = crm_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DataLoadError(f"Unable to read CRM file '{crm_path}': {exc}") from exc

        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"CRM file '{crm_path}' is not valid JSON: {exc}") from exc

        customers = payload.get("customers")
        if not isinstance(customers, list):
            raise DataLoadError("CRM file must contain a top-level 'customers' list.")

        for entry in customers:
            customer = self._build_customer(entry)
            self._index_customer(customer)

    def _build_customer(self, entry: dict[str, Any]) -> Customer:
        """Convert a raw CRM record into a validated :class:`Customer`."""
        try:
            orders = tuple(self._build_order(o) for o in entry.get("orders", []))
            return Customer(
                customer_id=entry["customer_id"],
                name=entry["name"],
                email=entry["email"],
                tier=entry.get("tier", "STANDARD"),
                account_status=entry.get("account_status", "ACTIVE"),
                fraud_flag=bool(entry.get("fraud_flag", False)),
                lifetime_refunds=int(entry.get("lifetime_refunds", 0)),
                orders=orders,
            )
        except KeyError as exc:
            raise DataLoadError(f"Customer record missing required field: {exc}") from exc
        except Exception as exc:  # pydantic ValidationError, etc.
            raise DataLoadError(f"Invalid customer record: {exc}") from exc

    def _build_order(self, entry: dict[str, Any]) -> Order:
        """Convert a raw order record into a validated :class:`Order`."""
        now = self._clock.now()
        placed_days_ago = entry.get("placed_days_ago")
        delivered_days_ago = entry.get("delivered_days_ago")

        order_date = now - timedelta(days=int(placed_days_ago)) if placed_days_ago is not None else now
        delivery_date = (
            now - timedelta(days=int(delivered_days_ago))
            if delivered_days_ago is not None
            else None
        )

        try:
            return Order(
                order_id=entry["order_id"],
                product_name=entry["product_name"],
                category=ProductCategory(entry.get("category", "OTHER")),
                amount=entry["amount"],
                status=OrderStatus(entry["status"]),
                order_date=order_date,
                delivery_date=delivery_date,
                is_final_sale=bool(entry.get("is_final_sale", False)),
                amount_already_refunded=entry.get("amount_already_refunded", 0.0),
            )
        except KeyError as exc:
            raise DataLoadError(f"Order record missing required field: {exc}") from exc
        except ValueError as exc:
            raise DataLoadError(f"Order record has an invalid enum value: {exc}") from exc

    def _index_customer(self, customer: Customer) -> None:
        """Add a customer to all lookup indexes."""
        if customer.customer_id in self._customers_by_id:
            raise DataLoadError(f"Duplicate customer_id '{customer.customer_id}' in CRM.")
        self._customers_by_id[customer.customer_id] = customer
        self._customer_id_by_email[customer.email.lower()] = customer.customer_id
        for order in customer.orders:
            if order.order_id in self._customer_id_by_order:
                raise DataLoadError(f"Duplicate order_id '{order.order_id}' in CRM.")
            self._customer_id_by_order[order.order_id] = customer.customer_id

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_customer(self, customer_id: str) -> Customer:
        """Return the customer with ``customer_id``.

        Args:
            customer_id: The customer identifier.

        Returns:
            The matching :class:`Customer`.

        Raises:
            CustomerNotFoundError: If no such customer exists.
        """
        customer = self._customers_by_id.get(customer_id)
        if customer is None:
            raise CustomerNotFoundError(f"No customer found with id '{customer_id}'.")
        return customer

    def find_customer_by_email(self, email: str) -> Customer:
        """Return the customer with the given email (case-insensitive).

        Raises:
            CustomerNotFoundError: If no such customer exists.
        """
        customer_id = self._customer_id_by_email.get(email.lower())
        if customer_id is None:
            raise CustomerNotFoundError(f"No customer found with email '{email}'.")
        return self._customers_by_id[customer_id]

    def get_order(self, order_id: str) -> Order:
        """Return the order with ``order_id``.

        Raises:
            OrderNotFoundError: If no such order exists.
        """
        customer_id = self._customer_id_by_order.get(order_id)
        if customer_id is None:
            raise OrderNotFoundError(f"No order found with id '{order_id}'.")
        order = self._customers_by_id[customer_id].find_order(order_id)
        if order is None:  # pragma: no cover - index invariant
            raise OrderNotFoundError(f"No order found with id '{order_id}'.")
        return order

    def get_customer_for_order(self, order_id: str) -> Customer:
        """Return the customer who owns ``order_id``.

        Raises:
            OrderNotFoundError: If no such order exists.
        """
        customer_id = self._customer_id_by_order.get(order_id)
        if customer_id is None:
            raise OrderNotFoundError(f"No order found with id '{order_id}'.")
        return self._customers_by_id[customer_id]

    def list_customers(self) -> list[Customer]:
        """Return all customers (read-only snapshot)."""
        return list(self._customers_by_id.values())

    def replace_order(self, order: Order) -> None:
        """Replace an existing order in the in-memory store.

        Used by the refund service to reflect processed refunds. Because
        :class:`Customer` and :class:`Order` are immutable, this rebuilds the owning
        customer with the updated order.

        Args:
            order: The new order state (must share an existing ``order_id``).

        Raises:
            OrderNotFoundError: If the order is not already present.
        """
        customer = self.get_customer_for_order(order.order_id)
        new_orders = tuple(order if o.order_id == order.order_id else o for o in customer.orders)
        updated = customer.model_copy(update={"orders": new_orders})
        self._customers_by_id[customer.customer_id] = updated

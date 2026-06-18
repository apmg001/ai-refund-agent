"""CRM repository tests.

Validates loading, indexing, lookup error handling, relative-date resolution, and
the immutable order-replacement path used by refund execution.
"""

from __future__ import annotations

import pytest

from refund_agent.exceptions import CustomerNotFoundError, OrderNotFoundError
from refund_agent.models.enums import OrderStatus
from refund_agent.repositories.crm_repository import CrmRepository


def test_loads_all_customers_and_orders(crm: CrmRepository) -> None:
    """The mock dataset loads exactly 15 customers and 15 orders."""
    customers = crm.list_customers()
    assert len(customers) == 15
    assert sum(len(c.orders) for c in customers) == 15


def test_lookup_by_id_and_email(crm: CrmRepository) -> None:
    """Customers are retrievable by id and by email."""
    by_id = crm.get_customer("CUST-001")
    by_email = crm.find_customer_by_email(by_id.email)
    assert by_id.customer_id == by_email.customer_id


def test_unknown_customer_raises(crm: CrmRepository) -> None:
    """Looking up a missing customer raises a specific error."""
    with pytest.raises(CustomerNotFoundError):
        crm.get_customer("CUST-999")


def test_unknown_order_raises(crm: CrmRepository) -> None:
    """Looking up a missing order raises a specific error."""
    with pytest.raises(OrderNotFoundError):
        crm.get_order("ORD-99999")


def test_relative_delivery_dates_resolved(crm: CrmRepository) -> None:
    """Relative delivery offsets are resolved to absolute timestamps on load."""
    delivered = crm.get_order("ORD-10001")
    assert delivered.delivery_date is not None

    not_delivered = crm.get_order("ORD-10010")  # SHIPPED, never delivered
    assert not_delivered.delivery_date is None


def test_replace_order_updates_in_memory(crm: CrmRepository) -> None:
    """Replacing an order swaps it in the in-memory overlay."""
    order = crm.get_order("ORD-10001")
    updated = order.model_copy(update={"status": OrderStatus.REFUNDED})
    crm.replace_order(updated)
    assert crm.get_order("ORD-10001").status is OrderStatus.REFUNDED

# ShopSphere Refund Policy (v2026.1)

This policy governs all customer-initiated refund requests. It is the single source of
truth for the AI Customer Support Agent. Every decision must be explainable by one or
more clauses below.

## 1. Eligibility Window

- **Standard window:** Refunds must be requested within **30 days** of the delivery date.
- **Extended window (seller-fault):** Requests citing a seller-fault reason
  (defective, damaged, wrong item, not as described, arrived late) are accepted within
  **90 days** of delivery.

## 2. Order Status

- Refunds are only processed for orders in **DELIVERED** or **PARTIALLY_REFUNDED** status.
- Orders that are still **PENDING / PROCESSING / SHIPPED** cannot be refunded; the customer
  should request an order **cancellation** instead.
- **CANCELLED** orders and orders already **fully REFUNDED** are not eligible.

## 3. Non-Refundable Items

- **Digital goods** (software keys, downloads) and **gift cards** are non-refundable under
  all circumstances.
- **Hygiene-sensitive items** (beauty, grocery) are non-refundable once delivered **unless**
  the request is seller-fault (e.g., the item arrived defective or damaged).
- **Final-sale / clearance** items are non-refundable **unless** the request is seller-fault.

## 4. Restocking Fee

- Opened **electronics** returned for a non-seller-fault reason (e.g., changed mind, no
  longer needed) are subject to a **15% restocking fee**. The refund is approved for the
  remaining balance after the fee.

## 5. Manual Review (Escalation)

A request is **escalated to a human agent** (never auto-approved or auto-denied) when:

- The refundable amount exceeds **$500.00**, or
- The customer account is **flagged for potential fraud / abuse**, or
- The customer account is **SUSPENDED**.

## 6. Refund Amount

- The approved amount never exceeds the remaining refundable balance
  (`order total − amount already refunded`).
- For partial-refund scenarios, only the un-refunded balance is considered.

## 7. Account Standing

- **CLOSED** accounts are not eligible for refunds.
- **SUSPENDED** accounts are escalated for manual review.

## 8. Decision Outcomes

| Outcome              | Meaning                                                        |
|----------------------|---------------------------------------------------------------|
| `APPROVED`           | Full eligible amount refunded.                                |
| `PARTIALLY_APPROVED` | Refund approved minus an applicable restocking fee.           |
| `DENIED`             | Request violates a hard policy rule.                          |
| `ESCALATED`          | Requires human review before a refund can be issued.          |

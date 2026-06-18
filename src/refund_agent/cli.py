"""Command-line interface.

The primary way to exercise the agent from a terminal. It provides:

    * An interactive REPL where you chat with the agent as a customer.
    * Slash-commands to inspect the mock data and the live reasoning trace
      (``/customers``, ``/orders``, ``/logs``, ``/policy``, ``/new``, ``/help``,
      ``/quit``).
    * A ``--message`` flag for one-shot, non-interactive use (handy for scripting).
    * A ``--demo`` flag that runs a scripted set of requests exercising every refund
      decision path, printing the agent's reasoning and reply for each.

Reasoning events are streamed live: the CLI subscribes an observer to the shared
:class:`ReasoningLog`, so each tool call and decision is printed as it happens.

Run it with::

    python -m refund_agent.cli            # interactive
    python -m refund_agent.cli --demo     # scripted walkthrough
    python -m refund_agent.cli --message "Refund ORD-10001, it's defective"
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .app import Application, build_application
from .config import get_settings
from .exceptions import RefundAgentError
from .logging_config import get_logger
from .observability.reasoning_log import EventType, ReasoningEvent

__all__ = ["main"]

_logger = get_logger(__name__)


# ====================================================================== #
# Optional rich console (degrades gracefully to plain printing)
# ====================================================================== #
try:  # pragma: no cover - presentation only
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console: Optional["Console"] = Console()
except ImportError:  # pragma: no cover - rich is optional
    _console = None


def _print(message: str = "", *, style: str = "") -> None:
    """Print a line via rich if available, else the stdlib."""
    if _console is not None:
        _console.print(message, style=style or None)
    else:
        # Strip rich markup tokens for the plain fallback.
        print(message)


# Event-type -> display colour for the live reasoning stream.
_EVENT_STYLES = {
    EventType.USER_MESSAGE: "bold white",
    EventType.LLM_REQUEST: "dim cyan",
    EventType.LLM_RESPONSE: "cyan",
    EventType.TOOL_CALL: "yellow",
    EventType.TOOL_RESULT: "green",
    EventType.TOOL_ERROR: "bold red",
    EventType.DECISION: "bold magenta",
    EventType.AGENT_RESPONSE: "bold blue",
    EventType.ERROR: "bold red",
}


def _make_reasoning_observer(verbose: bool):
    """Return an observer that prints reasoning events as they are recorded."""

    def _observer(event: ReasoningEvent) -> None:
        style = _EVENT_STYLES.get(event.event_type, "white")
        prefix = f"   · [{event.event_type.value}]"
        detail = ""
        if event.event_type == EventType.TOOL_CALL:
            detail = f" {event.detail.get('tool')}({event.detail.get('arguments')})"
        elif event.event_type == EventType.DECISION:
            detail = f" -> {event.detail.get('decision')}"
        elif verbose and event.detail:
            detail = f" {event.detail}"
        _print(f"{prefix} {event.title}{detail}", style=style)

    return _observer


# ====================================================================== #
# Command handlers
# ====================================================================== #
def _cmd_help() -> None:
    """Print the list of available slash-commands."""
    _print(
        "\nCommands:\n"
        "  /help              Show this help\n"
        "  /customers         List all mock customers\n"
        "  /orders <CUST-ID>  List a customer's orders\n"
        "  /logs              Show the reasoning trace for the current session\n"
        "  /policy [topic]    Show the refund policy (optionally a section)\n"
        "  /new               Start a fresh conversation/session\n"
        "  /quit              Exit\n"
        "\nOr just type a message as if you were a customer "
        "(mention an order id like ORD-10001).\n",
        style="dim",
    )


def _cmd_customers(app: Application) -> None:
    """List all customers in the mock CRM."""
    customers = app.crm.list_customers()
    if _console is not None:
        table = Table(title="Mock Customers", show_lines=False)
        for col in ("Customer", "Name", "Tier", "Account", "Orders"):
            table.add_column(col)
        for c in customers:
            table.add_row(
                c.customer_id,
                c.name,
                c.tier.value,
                c.account_status.value,
                ", ".join(o.order_id for o in c.orders),
            )
        _console.print(table)
    else:
        for c in customers:
            order_ids = ", ".join(o.order_id for o in c.orders)
            print(f"{c.customer_id}  {c.name:<8}  {c.tier.value:<8}  {order_ids}")


def _cmd_orders(app: Application, customer_id: str) -> None:
    """List a single customer's orders."""
    try:
        customer = app.crm.get_customer(customer_id)
    except RefundAgentError as exc:
        _print(str(exc), style="red")
        return
    for o in customer.orders:
        _print(
            f"{o.order_id}  {o.product_name:<22} {o.category.value:<11} "
            f"${o.amount}  {o.status.value}  finalSale={o.is_final_sale}"
        )


def _cmd_logs(app: Application, session_id: Optional[str]) -> None:
    """Print the full reasoning trace for the current session."""
    if not session_id:
        _print("No active session yet. Send a message first.", style="dim")
        return
    events = app.reasoning_log.get_events(session_id)
    if not events:
        _print("No reasoning events recorded for this session.", style="dim")
        return
    for event in events:
        style = _EVENT_STYLES.get(event.event_type, "white")
        _print(f"[{event.sequence:>2}] {event.event_type.value:<14} {event.title}", style=style)


def _cmd_policy(app: Application, topic: str) -> None:
    """Print the (optionally topic-scoped) refund policy text."""
    if topic:
        _print(app.policy_repo.search_document(topic))
    else:
        _print(app.policy_repo.document)


# ====================================================================== #
# REPL
# ====================================================================== #
def _run_repl(app: Application, verbose: bool) -> int:
    """Run the interactive read-eval-print loop.

    Args:
        app: The wired application.
        verbose: Whether to print full event detail in the live stream.

    Returns:
        Process exit code.
    """
    app.reasoning_log.subscribe(_make_reasoning_observer(verbose))
    session_id = app.agent.new_session()

    banner = (
        f"{app.settings.app_name}\n"
        f"LLM provider: {getattr(app.llm, 'name', 'unknown')}   "
        f"Session: {session_id}\n"
        "Type /help for commands, /quit to exit."
    )
    if _console is not None:
        _console.print(Panel(banner, style="bold blue"))
    else:
        print(banner)

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\nGoodbye!", style="dim")
            return 0

        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            command, arg = parts[0], (parts[1] if len(parts) > 1 else "")
            if command in ("/quit", "/exit"):
                _print("Goodbye!", style="dim")
                return 0
            if command == "/help":
                _cmd_help()
            elif command == "/customers":
                _cmd_customers(app)
            elif command == "/orders":
                if not arg:
                    _print("Usage: /orders <CUST-ID>", style="red")
                else:
                    _cmd_orders(app, arg.strip())
            elif command == "/logs":
                _cmd_logs(app, session_id)
            elif command == "/policy":
                _cmd_policy(app, arg.strip())
            elif command == "/new":
                session_id = app.agent.new_session()
                _print(f"Started new session: {session_id}", style="dim")
            else:
                _print(f"Unknown command '{command}'. Type /help.", style="red")
            continue

        _handle_turn(app, line, session_id)

    return 0


def _handle_turn(app: Application, message: str, session_id: str) -> None:
    """Run one agent turn and print the reply, guarding against failures."""
    try:
        result = app.agent.run(message, session_id=session_id)
    except RefundAgentError as exc:  # pragma: no cover - defensive
        _logger.exception("Agent turn failed")
        _print(f"agent> (error) {exc}", style="bold red")
        return
    _print(f"\nagent> {result.reply}", style="bold blue")


# ====================================================================== #
# Demo mode
# ====================================================================== #
_DEMO_SCRIPT: tuple[tuple[str, str], ...] = (
    ("APPROVED (full)", "Hi, my headphones ORD-10001 arrived defective. I'd like a refund."),
    ("DENIED (window)", "I want to return ORD-10002, I changed my mind about the shoes."),
    ("PARTIALLY_APPROVED (restocking fee)", "Please refund ORD-10003, I changed my mind on the smartwatch."),
    ("DENIED (digital)", "Refund my game code ORD-10004 please, no longer needed."),
    ("ESCALATED (fraud + high value)", "I need a refund for ORD-10005, the bag is defective."),
    ("APPROVED (extended window)", "ORD-10006 coffee maker is defective, I want my money back."),
    ("DENIED (final sale)", "Refund the clearance tee ORD-10007, changed my mind."),
    ("DENIED (gift card)", "I'd like to refund gift card ORD-10008."),
    ("ESCALATED (high value)", "My laptop ORD-10009 is defective, please refund."),
    ("DENIED (not delivered)", "Refund ORD-10010, I changed my mind on the blender."),
    ("DENIED (already refunded)", "Please refund ORD-10011 again."),
    ("APPROVED (wrong item)", "I received the wrong item for ORD-10012, refund please."),
    ("DENIED (hygiene)", "Refund the perfume ORD-10013, I changed my mind."),
    ("ESCALATED (suspended account)", "My monitor ORD-10014 is defective, refund please."),
    ("APPROVED (partial remaining)", "ORD-10015 cookware is defective, refund the rest please."),
)


def _run_demo(app: Application, verbose: bool) -> int:
    """Run the scripted demo across every decision path.

    Args:
        app: The wired application.
        verbose: Whether to print full event detail in the live stream.

    Returns:
        Process exit code.
    """
    app.reasoning_log.subscribe(_make_reasoning_observer(verbose))
    _print(f"Running scripted demo with the '{getattr(app.llm, 'name', '?')}' provider.\n", style="bold")

    for label, message in _DEMO_SCRIPT:
        if _console is not None:
            _console.rule(f"[bold]{label}")
        else:
            print(f"\n===== {label} =====")
        _print(f"you> {message}", style="bold white")
        session_id = app.agent.new_session()
        result = app.agent.run(message, session_id=session_id)
        _print(f"agent> {result.reply}", style="bold blue")
    return 0


# ====================================================================== #
# Entry point
# ====================================================================== #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="refund-agent",
        description="Terminal interface for the AI refund support agent.",
    )
    parser.add_argument("--message", "-m", help="Run a single message non-interactively and exit.")
    parser.add_argument("--demo", action="store_true", help="Run a scripted walkthrough of all decision paths.")
    parser.add_argument("--provider", choices=("auto", "ollama", "heuristic"), help="Override the LLM provider.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full reasoning-event detail.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    args = _parse_args(argv)

    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})

    try:
        app = build_application(settings)
    except RefundAgentError as exc:
        _print(f"Failed to start: {exc}", style="bold red")
        return 1

    if args.demo:
        return _run_demo(app, args.verbose)

    if args.message:
        app.reasoning_log.subscribe(_make_reasoning_observer(args.verbose))
        session_id = app.agent.new_session()
        result = app.agent.run(args.message, session_id=session_id)
        _print(f"\nagent> {result.reply}", style="bold blue")
        return 0

    return _run_repl(app, args.verbose)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

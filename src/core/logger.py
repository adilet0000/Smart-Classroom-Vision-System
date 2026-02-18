from __future__ import annotations

from rich.console import Console

console = Console()


def log_info(msg: str) -> None:
   console.print(f"[bold cyan]INFO[/bold cyan] {msg}")


def log_warn(msg: str) -> None:
   console.print(f"[bold yellow]WARN[/bold yellow] {msg}")


def log_error(msg: str) -> None:
   console.print(f"[bold red]ERROR[/bold red] {msg}")

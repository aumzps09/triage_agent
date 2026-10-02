"""Console runner: triages the 3 sample tickets and prints each TriageResult."""

from __future__ import annotations

import json

from .agent import run_ticket
from .sample_tickets import SAMPLE_TICKETS


def main() -> None:
    for ticket in SAMPLE_TICKETS:
        result = run_ticket(ticket)
        print(f"=== {ticket['id']} ({ticket['customer_id']}) ===")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()

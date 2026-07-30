"""Check that every recorded LLM request has one terminal event."""

from __future__ import annotations

import argparse

from observability.request_recorder import validate_ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger")
    args = parser.parse_args()

    summary = validate_ledger(args.ledger)
    print(
        f"started={summary.started} completed={summary.completed} "
        f"failed={summary.failed} "
        f"incomplete={len(summary.incomplete_request_ids)}"
    )
    if not summary.is_complete:
        for request_id in summary.incomplete_request_ids:
            print(f"incomplete request: {request_id}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

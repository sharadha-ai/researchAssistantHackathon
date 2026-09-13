"""
Non-interactive entry point for scheduled runs (GitHub Actions or local
cron). hackathon_agent.py's __main__ is an interactive REPL (uses input()),
which doesn't work in CI — this is the equivalent for unattended runs.

Run directly:
    python run_weekly.py
"""

import os

from hackathon_agent import ask, notify_telegram, render_report, save_digest

DEFAULT_QUESTION = (
    "What's new this week in Agentic AI hiring trends and in tooling/"
    "frameworks worth knowing about?"
)


def main() -> None:
    question = os.environ.get("DIGEST_QUESTION", DEFAULT_QUESTION)
    digest, cost = ask(question)
    report = render_report(question, digest, cost)

    md_path, json_path = save_digest(question, digest, cost)
    print(report)
    print(f"\nSaved: {md_path}, {json_path}")

    if os.environ.get("SEND_TELEGRAM", "false").lower() == "true":
        notify_telegram(report)


if __name__ == "__main__":
    main()
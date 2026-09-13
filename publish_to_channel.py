"""
Manually-triggered publish step. Reads the most recent saved digest and
posts its LinkedIn draft to the student-facing Telegram channel — unless a
fine-tuned override is provided at trigger time, in which case that exact
text is published instead.

This script never runs on a schedule. The only way anything reaches the
channel is a human explicitly triggering this workflow — that trigger *is*
the approval.
"""

import glob
import json
import os

import requests


def get_latest_digest() -> dict:
    files = sorted(glob.glob("digests/*.json"))
    if not files:
        raise FileNotFoundError(
            "No digest files found in digests/ — run the weekly digest workflow first."
        )
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def publish_to_channel(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    channel_id = os.environ["TELEGRAM_CHANNEL_ID"]
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": channel_id, "text": text[:4000]},  # Telegram's message cap
        timeout=15,
    )
    resp.raise_for_status()
    print("Published to channel.")


def main() -> None:
    override = os.environ.get("MESSAGE_OVERRIDE", "").strip()
    if override:
        text = override
        print("Publishing fine-tuned override text provided at trigger time.")
    else:
        digest_data = get_latest_digest()
        text = digest_data["digest"]["linkedin_post_draft"]
        print(f"Publishing linkedin_post_draft from digest dated {digest_data['timestamp_utc']}")

    publish_to_channel(text)


if __name__ == "__main__":
    main()
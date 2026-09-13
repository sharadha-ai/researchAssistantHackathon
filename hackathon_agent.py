"""
Agentic AI Market/Research Analyst — Hackathon build (Agents for Humans, AWS)
Same research core (job-market + tooling-trend sub-agents, orchestrator) as
the earlier phases, rebuilt with Strands-native polish:

  - Bedrock or Groq model provider (MODEL_PROVIDER env var; Groq is default
    for the local-app submission path, Bedrock kept available but optional)
  - Real rate-limit handling via ModelRetryStrategy
  - Optional Bedrock Guardrail wiring
  - Bedrock prompt caching (system prompt is long + static — ideal for it)
  - A minimal, env-configurable $ costing model
  - Structured output (Pydantic) for the "study vs. share" digest, produced
    by a separate tool-less formatter agent (see ask() for why)
  - A Steering policy (strands.vended_plugins.steering) enforcing a
    per-tool call budget with real feedback to the model
  - Native Strands observability (StrandsTelemetry, console exporter)

All new knobs are environment variables — see .env.example. Copy it to
.env and fill in your model ID / region / keys before running.

Run:
    cp .env.example .env   # then edit .env
    pip install -r requirements_hackathon.txt
    python hackathon_agent.py
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime, timedelta, timezone

import feedparser
import requests
from ddgs import DDGS
from pydantic import BaseModel, Field
from strands import Agent, ModelRetryStrategy, tool
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry
from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler

from memory_store import filter_new

# --------------------------------------------------------------------------
# Native Strands observability. Console exporter for now — swap to
# .setup_otlp_exporter() pointed at an OTLP collector (e.g. AWS X-Ray) once
# this is running somewhere persistent.
# --------------------------------------------------------------------------
StrandsTelemetry().setup_console_exporter()

DEFAULT_FEEDS = {
    "aws_ml_blog": "https://aws.amazon.com/blogs/machine-learning/feed/",
    "openai_news": "https://openai.com/news/rss.xml",
    "huggingface_blog": "https://huggingface.co/blog/feed.xml",
    "deepmind_blog": "https://deepmind.google/blog/rss.xml",
    "arxiv_cs_ai": "https://rss.arxiv.org/rss/cs.AI",
}

SUB_AGENT_TOOL_CALL_BUDGET = int(os.environ.get("SUB_AGENT_TOOL_CALL_BUDGET", "2"))
DEBUG_GROUNDING = os.environ.get("DEBUG_GROUNDING", "0") == "1"

PRICE_PER_1K_INPUT = float(os.environ.get("PRICE_PER_1K_INPUT_TOKENS", "0"))
PRICE_PER_1K_OUTPUT = float(os.environ.get("PRICE_PER_1K_OUTPUT_TOKENS", "0"))
PRICE_PER_1K_CACHE_READ = float(os.environ.get("PRICE_PER_1K_CACHE_READ_TOKENS", "0"))
PRICE_PER_1K_CACHE_WRITE = float(os.environ.get("PRICE_PER_1K_CACHE_WRITE_TOKENS", "0"))


# --------------------------------------------------------------------------
# Model factory — Bedrock or Groq, with retry, guardrail, caching, and
# service-tier all driven by env vars so they can be changed without
# touching code.
# --------------------------------------------------------------------------

def make_model():
    provider = os.environ.get("MODEL_PROVIDER", "groq").lower()

    if provider == "bedrock":
        kwargs = dict(
            model_id=os.environ["BEDROCK_MODEL_ID"],
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            service_tier=os.environ.get("BEDROCK_SERVICE_TIER", "default"),
        )
        if os.environ.get("BEDROCK_CACHE_PROMPT", "true").lower() == "true":
            kwargs["cache_prompt"] = "default"
        guardrail_id = os.environ.get("BEDROCK_GUARDRAIL_ID", "").strip()
        if guardrail_id:
            kwargs["guardrail_id"] = guardrail_id
            kwargs["guardrail_version"] = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
            kwargs["guardrail_latest_message"] = True
        return BedrockModel(**kwargs)

    # Groq via Strands' OpenAI-compatible model class — the tested, working
    # path with no AWS access requirements.
    return OpenAIModel(
        client_args={
            "api_key": os.environ["GROQ_API_KEY"],
            "base_url": "https://api.groq.com/openai/v1",
        },
        model_id=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
        params={"temperature": 0.5},
    )


def make_retry_strategy() -> ModelRetryStrategy:
    # Real throttling protection for shared on-demand capacity — NOT the
    # same code path as tool-execution errors (those are handled by our own
    # prompt instructions / steering policy below, not this).
    return ModelRetryStrategy(
        max_attempts=int(os.environ.get("MODEL_RETRY_MAX_ATTEMPTS", "4")),
        initial_delay=int(os.environ.get("MODEL_RETRY_INITIAL_DELAY", "2")),
        max_delay=int(os.environ.get("MODEL_RETRY_MAX_DELAY", "30")),
    )


# --------------------------------------------------------------------------
# Minimal costing model. Neither Bedrock nor Groq returns a $ figure
# natively — this reads result.metrics.accumulated_usage and applies
# env-configured per-1K prices.
# --------------------------------------------------------------------------

def estimate_cost(result) -> dict:
    usage = result.metrics.accumulated_usage
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)

    cost = (
        (input_tokens / 1000) * PRICE_PER_1K_INPUT
        + (output_tokens / 1000) * PRICE_PER_1K_OUTPUT
        + (cache_read / 1000) * PRICE_PER_1K_CACHE_READ
        + (cache_write / 1000) * PRICE_PER_1K_CACHE_WRITE
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "estimated_cost_usd": round(cost, 6),
    }


# --------------------------------------------------------------------------
# Steering policy — enforces a per-tool call budget with real feedback to
# the model, instead of a hard turn-count cutoff.
# --------------------------------------------------------------------------

class ToolCallBudgetPolicy(SteeringHandler):
    def __init__(self, budget: int = SUB_AGENT_TOOL_CALL_BUDGET):
        super().__init__()
        self.budget = budget
        self._calls_by_tool: dict[str, int] = {}

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        name = tool_use.get("name", "")
        count = self._calls_by_tool.get(name, 0)
        if count >= self.budget:
            return Guide(
                reason=(
                    f"You've already called '{name}' {count} time(s) this turn, "
                    f"which is the budget for this question. Do not call it "
                    f"again — answer using what it already returned, and say "
                    f"plainly if that isn't enough to fully answer."
                )
            )
        self._calls_by_tool[name] = count + 1
        return Proceed(reason=f"{name} call {count + 1}/{self.budget}")

    async def steer_after_model(self, *, agent, message, stop_reason, **kwargs):
        # Extension point for an automated grounding check — see
        # README_hackathon.md.
        return Proceed(reason="no post-response check configured yet")


# --------------------------------------------------------------------------
# Low-level tools.
# --------------------------------------------------------------------------

def _format_items(items: list[dict]) -> str:
    if not items:
        return "No new items since last check (already surfaced previously)."
    return "\n\n".join(f"- {i['title']}\n  {i.get('detail', '')}\n  {i['url']}" for i in items)


def _maybe_debug_print(source: str, formatted: str) -> None:
    if DEBUG_GROUNDING:
        import sys
        print(f"\n[DEBUG_GROUNDING] raw output from {source}:\n{formatted}\n", file=sys.stderr)


@tool
def search_job_trends(query: str) -> str:
    """
    Search the web for recent job postings, tools, or trend news related to
    Agentic AI. Results already surfaced in a previous run are filtered out.

    Args:
        query: What to search for, e.g. "Agentic AI engineer job openings 2026"

    Returns:
        New (not previously seen) results as title / snippet / url.
    """
    results = DDGS().text(query, max_results=8)
    items = [{"title": r["title"], "detail": r["body"], "url": r["href"]} for r in results]
    formatted = _format_items(filter_new("search_job_trends", items))
    _maybe_debug_print("search_job_trends", formatted)
    return formatted


@tool
def github_trending(query: str, days: int = 7) -> str:
    """
    Find GitHub repositories matching a topic that were created recently and
    are already gaining stars. Repos already surfaced in a previous run are
    filtered out.

    Args:
        query: Topic to search for, e.g. "agentic ai" or "llm agent framework"
        days: How many days back to look for newly created repos (default 7)

    Returns:
        New (not previously seen) repos as name / stars / description / url.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://api.github.com/search/repositories",
        params={"q": f"{query} created:>{since}", "sort": "stars", "order": "desc", "per_page": 8},
        headers={"Accept": "application/vnd.github+json"},
        timeout=15,
    )
    resp.raise_for_status()
    repos = resp.json().get("items", [])
    items = [
        {
            "title": f"{r['full_name']} ({r['stargazers_count']}⭐)",
            "detail": r.get("description") or "No description",
            "url": r["html_url"],
        }
        for r in repos
    ]
    formatted = _format_items(filter_new("github_trending", items))
    _maybe_debug_print("github_trending", formatted)
    return formatted


@tool
def read_rss_feed(feed_name_or_url: str, max_items: int = 5) -> str:
    """
    Fetch the latest entries from an RSS/Atom feed. Entries already surfaced
    in a previous run are filtered out.

    Args:
        feed_name_or_url: Either a known feed name (aws_ml_blog, openai_news,
            huggingface_blog, deepmind_blog, arxiv_cs_ai) or a full feed URL.
        max_items: Max number of entries to consider before filtering (default 5)

    Returns:
        New (not previously seen) entries as title / date / link.
    """
    feed_url = DEFAULT_FEEDS.get(feed_name_or_url, feed_name_or_url)
    feed = feedparser.parse(feed_url)
    if feed.bozo and not feed.entries:
        return f"Could not parse feed: {feed_url}"
    entries = feed.entries[:max_items]
    items = [
        {
            "title": e.get("title", "Untitled"),
            "detail": e.get("published", e.get("updated", "unknown date")),
            "url": e.get("link", ""),
        }
        for e in entries
    ]
    formatted = _format_items(filter_new(f"rss:{feed_name_or_url}", items))
    _maybe_debug_print(f"read_rss_feed[{feed_name_or_url}]", formatted)
    return formatted


# --------------------------------------------------------------------------
# Sub-agents + orchestrator + formatter, built once and reused.
# --------------------------------------------------------------------------

_job_market_agent: Agent | None = None
_tooling_agent: Agent | None = None
_orchestrator: Agent | None = None
_formatter: Agent | None = None


def _get_job_market_agent() -> Agent:
    global _job_market_agent
    if _job_market_agent is None:
        _job_market_agent = Agent(
            model=make_model(),
            tools=[search_job_trends],
            retry_strategy=make_retry_strategy(),
            plugins=[ToolCallBudgetPolicy()],
            system_prompt=(
                "You are a Job Market Analyst sub-agent focused only on hiring "
                "trends for Agentic AI roles. Use search_job_trends to find real "
                "postings and hiring signals. Some results may already have been "
                "reported in a previous run and won't appear again — if the tool "
                "returns 'No new items', say so plainly rather than inventing "
                "findings. Only name a specific tool, product, or technology if "
                "it is explicitly present in the search results — otherwise "
                "describe the skill category in general terms. Report concrete "
                "findings only."
            ),
        )
    return _job_market_agent


def _get_tooling_agent() -> Agent:
    global _tooling_agent
    if _tooling_agent is None:
        _tooling_agent = Agent(
            model=make_model(),
            tools=[github_trending, read_rss_feed],
            retry_strategy=make_retry_strategy(),
            plugins=[ToolCallBudgetPolicy()],
            system_prompt=(
                "You are a Tooling & Framework Trend-Watcher sub-agent focused "
                "only on Agentic AI frameworks, libraries, and research. Use "
                "github_trending and read_rss_feed to find what's newly gaining "
                "traction. Some results may already have been reported in a "
                "previous run and won't appear again — if a tool returns 'No new "
                "items', say so plainly rather than inventing findings. Only "
                "name a specific tool, product, or technology if it is "
                "explicitly present in the results — otherwise describe the "
                "category in general terms."
            ),
        )
    return _tooling_agent


@tool
def job_market_analyst(question: str) -> str:
    """
    Delegate a question specifically about Agentic AI hiring trends, job
    postings, or in-demand skills to a specialized job-market analyst sub-agent.

    Args:
        question: The specific job-market question to investigate.

    Returns:
        The sub-agent's findings as text.
    """
    sub_agent = _get_job_market_agent()
    sub_agent.messages = []
    return str(sub_agent(question))


@tool
def tooling_trend_watcher(question: str) -> str:
    """
    Delegate a question specifically about trending Agentic AI tools,
    frameworks, libraries, or research to a specialized trend-watcher sub-agent.

    Args:
        question: The specific tooling/framework question to investigate.

    Returns:
        The sub-agent's findings as text.
    """
    sub_agent = _get_tooling_agent()
    sub_agent.messages = []
    return str(sub_agent(question))


# --------------------------------------------------------------------------
# Structured output — the "study vs. share" digest as a real typed object.
# --------------------------------------------------------------------------

class Finding(BaseModel):
    summary: str = Field(description="One-line summary of the finding")
    signal_strength: str = Field(
        description="'strong' if corroborated across multiple sources/sub-agents, "
        "'early' if from a single source"
    )
    source_note: str = Field(description="Which sub-agent(s)/tools this came from")


class TrainingModuleIdea(BaseModel):
    topic: str = Field(description="A concrete lesson/module topic derived from a finding")
    learning_objective: str = Field(
        description="One sentence: what a learner should be able to do after this module"
    )
    key_points: list[str] = Field(description="3-5 bullet points the module should cover")
    based_on: str = Field(description="Which finding(s) this idea is derived from")


class MarketDigest(BaseModel):
    what_to_study: list[Finding] = Field(
        description="Skills/tools the trainer should personally deepen"
    )
    what_to_share_with_learners: list[Finding] = Field(
        description="Findings solid enough (corroborated) to teach as-is"
    )
    linkedin_post_draft: str = Field(
        description="A ready-to-edit LinkedIn post (120-200 words) built around the "
        "single strongest ('strong' signal_strength) finding this run. Written in an "
        "approachable, credible voice for an audience of people considering or "
        "working in AI/Agentic AI careers. No hashtag spam, no generic hype — it "
        "should read as genuinely informed, citing what kind of evidence backs it "
        "(e.g. 'multiple postings this week ask for...') without inventing specifics "
        "not present in the findings."
    )
    training_module_ideas: list[TrainingModuleIdea] = Field(
        description="1-3 concrete training module ideas derived directly from the "
        "findings above — each one should be specific enough that the trainer could "
        "start outlining a lesson from it immediately, not a restatement of the "
        "finding itself."
    )


def _get_orchestrator() -> Agent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Agent(
            model=make_model(),
            tools=[job_market_analyst, tooling_trend_watcher],
            retry_strategy=make_retry_strategy(),
            system_prompt=(
                "You are the Market Analyst orchestrator for an Agentic AI "
                "trainer. You have two specialist sub-agents available as "
                "tools:\n"
                "- job_market_analyst: hiring trends, in-demand skills, job postings\n"
                "- tooling_trend_watcher: trending frameworks, libraries, research\n\n"
                "Delegate to whichever sub-agent(s) are relevant — use both only "
                "if the question genuinely spans hiring and tooling; for a "
                "question clearly about one domain, call only the matching "
                "sub-agent. Do not answer research questions yourself without "
                "delegating.\n\n"
                "Do not add specifics (tool names, products, frameworks) beyond "
                "what the sub-agent findings actually contain. Classify each "
                "finding's signal_strength as 'strong' only if corroborated "
                "across multiple sub-agents/sources, otherwise 'early'.\n\n"
                "After classifying findings, also produce: (1) a linkedin_post_draft "
                "— a ready-to-post draft built from the single strongest finding, "
                "not a summary of everything, and (2) training_module_ideas — 1-3 "
                "concrete lesson ideas a trainer could start outlining immediately. "
                "Both must be grounded only in the findings you actually gathered — "
                "do not invent specifics to make the content more compelling."
            ),
        )
    return _orchestrator


def _get_formatter() -> Agent:
    global _formatter
    if _formatter is None:
        _formatter = Agent(
            model=make_model(),
            retry_strategy=make_retry_strategy(),
            system_prompt=(
                "You format grounded research findings into the requested "
                "MarketDigest schema. Use only the findings supplied in the "
                "prompt. Do not invent tools, products, frameworks, job titles, "
                "or corroboration. Treat findings from one sub-agent or source "
                "as 'early'; use 'strong' only when the supplied findings show "
                "corroboration across multiple sources or sub-agents."
            ),
        )
    return _formatter


def ask(question: str) -> tuple[MarketDigest, dict]:
    """Runs one research question through the orchestrator, returns the
    structured digest plus a cost/usage summary.

    Two separate agents, not one — structured_output() sends tool_choice=
    "none" under the hood (it's meant to format an answer from what's
    already known, not to kick off new tool calls). A single agent that has
    tools attached AND is asked for structured_output on a blank
    conversation will try to delegate (per its system prompt) while the API
    request simultaneously forbids tool use — a contradiction some backends
    reject outright. Splitting into "research agent (tools allowed)" then
    "tool-less formatter agent (structurally cannot call a tool)" resolves
    it structurally rather than by hoping the model behaves.
    """
    orchestrator = _get_orchestrator()
    orchestrator.messages = []

    # Step 1: research — tools allowed, delegation happens here.
    research_result = orchestrator(question)

    # Step 2: separate, tool-less formatter — cannot attempt a tool call.
    formatter = _get_formatter()
    digest = formatter.structured_output(
        MarketDigest,
        f"Create the final digest for this question using only the research "
        f"findings below.\n\nQuestion: {question}\n\nResearch findings:\n{research_result}",
    )

    # Cost reflects the research call only — structured_output() doesn't
    # return a metrics-bearing result to read from.
    return digest, estimate_cost(research_result)


# --------------------------------------------------------------------------
# "Beautified" console report — for demo/showcase purposes.
# --------------------------------------------------------------------------

def render_report(question: str, digest: MarketDigest, cost: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"  MARKET ANALYST DIGEST")
    lines.append("=" * 70)
    lines.append(f"Q: {question}\n")

    lines.append("📚 WHAT TO STUDY")
    lines.append("-" * 70)
    for f in digest.what_to_study:
        tag = "🟢 STRONG" if f.signal_strength.lower() == "strong" else "🟡 EARLY"
        lines.append(f"  {tag}  {f.summary}")
        lines.append(f"           ({f.source_note})")
    if not digest.what_to_study:
        lines.append("  (none)")

    lines.append("\n🎓 WHAT TO SHARE WITH LEARNERS")
    lines.append("-" * 70)
    for f in digest.what_to_share_with_learners:
        tag = "🟢 STRONG" if f.signal_strength.lower() == "strong" else "🟡 EARLY"
        lines.append(f"  {tag}  {f.summary}")
        lines.append(f"           ({f.source_note})")
    if not digest.what_to_share_with_learners:
        lines.append("  (none)")

    lines.append("\n📝 LINKEDIN POST DRAFT (ready to edit)")
    lines.append("-" * 70)
    lines.append(digest.linkedin_post_draft or "(none generated)")

    lines.append("\n🧩 TRAINING MODULE IDEAS")
    lines.append("-" * 70)
    for idea in digest.training_module_ideas:
        lines.append(f"  ▸ {idea.topic}")
        lines.append(f"    Objective: {idea.learning_objective}")
        for kp in idea.key_points:
            lines.append(f"      - {kp}")
        lines.append(f"    (based on: {idea.based_on})")
    if not digest.training_module_ideas:
        lines.append("  (none)")

    lines.append("\n" + "-" * 70)
    lines.append(
        f"💵 est. cost: ${cost['estimated_cost_usd']:.6f}  |  "
        f"in: {cost['input_tokens']}  out: {cost['output_tokens']}  "
        f"cache_read: {cost['cache_read_tokens']}  cache_write: {cost['cache_write_tokens']}"
    )
    lines.append("=" * 70)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# #3 — Persisted history.
# --------------------------------------------------------------------------

DIGESTS_DIR = Path(os.environ.get("DIGESTS_DIR", "digests"))


def save_digest(question: str, digest: MarketDigest, cost: dict) -> tuple[Path, Path]:
    DIGESTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    md_path = DIGESTS_DIR / f"{stamp}.md"
    json_path = DIGESTS_DIR / f"{stamp}.json"

    md_path.write_text(render_report(question, digest, cost), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "timestamp_utc": stamp,
                "question": question,
                "digest": digest.model_dump(),
                "cost": cost,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


# --------------------------------------------------------------------------
# #2 — Push-only Telegram notification.
# --------------------------------------------------------------------------

def notify_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=15,
        )
    except requests.RequestException as e:
        print(f"[telegram] notification failed (run continues): {e}")


if __name__ == "__main__":
    print("Market Analyst orchestrator (hackathon build). Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        digest, cost = ask(user_input)
        report = render_report(user_input, digest, cost)
        print("\n" + report + "\n")
        md_path, _json_path = save_digest(user_input, digest, cost)
        print(f"(saved: {md_path})")
        if os.environ.get("SEND_TELEGRAM", "false").lower() == "true":
            notify_telegram(report)
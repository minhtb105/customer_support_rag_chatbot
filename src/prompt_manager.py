"""
LangSmith Prompt Hub integration: prompt versioning with local fallback.

Strategy (Hub-first + fallback local):
- Prompts are pushed once to LangSmith Prompt Hub via push_all_prompts().
- At runtime get_system_prompt() pulls the latest committed version from the
  Hub (with a small TTL cache). If the Hub/network/API-key is unavailable it
  falls back to the local constants in prompt_templates.py so the app never
  breaks offline.
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

try:
    from config import (
        LANGSMITH_PROJECT,
        PROMPT_HUB_CACHE_TTL_SECONDS,
        PROMPT_HUB_REPO_PREFIX,
    )
except ImportError:  # pragma: no cover - allows running as "src." package too
    from src.config import (
        LANGSMITH_PROJECT,
        PROMPT_HUB_CACHE_TTL_SECONDS,
        PROMPT_HUB_REPO_PREFIX,
    )

try:
    from observability.tracing import get_client, short_hash
except ImportError:
    from src.observability.tracing import get_client, short_hash

try:
    from prompt_templates import (
        BALANCED_SYSTEM_PROMPT,
        DIABETES_STRICT_PROMPT,
        EVALUATION_PROMPT,
        FRIENDLY_SYSTEM_PROMPT,
        SOAP_PROMPT,
        STRICT_SYSTEM_PROMPT,
        WHO_RAG_AUDIT_PROMPT,
    )
except ImportError:  # pragma: no cover
    from src.prompt_templates import (
        BALANCED_SYSTEM_PROMPT,
        DIABETES_STRICT_PROMPT,
        EVALUATION_PROMPT,
        FRIENDLY_SYSTEM_PROMPT,
        SOAP_PROMPT,
        STRICT_SYSTEM_PROMPT,
        WHO_RAG_AUDIT_PROMPT,
    )

logger = logging.getLogger(__name__)

# Registry: runtime key -> Hub repo name + local fallback text.
PROMPT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "strict": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-strict",
        "description": "Cautious medical assistant - concise factual answers",
        "local": STRICT_SYSTEM_PROMPT,
    },
    "friendly": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-friendly",
        "description": "Friendly medical assistant - simple reassuring language",
        "local": FRIENDLY_SYSTEM_PROMPT,
    },
    "balanced": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-balanced",
        "description": "Balanced medical assistant - structured reasoning answers",
        "local": BALANCED_SYSTEM_PROMPT,
    },
    "diabetes": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-diabetes-strict",
        "description": "Diabetes specialist — WHO/ADA/BYT guideline RAG with citations",
        "local": DIABETES_STRICT_PROMPT,
    },
    "soap": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-soap",
        "description": "SOAP pre-visit summary generator for diabetes follow-up",
        "local": SOAP_PROMPT,
    },
    "who_rag": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-who-rag-audit",
        "description": "WHO-RAG Infrastructure audit trail layer",
        "local": WHO_RAG_AUDIT_PROMPT,
    },
    "evaluation": {
        "repo": f"{PROMPT_HUB_REPO_PREFIX}-evaluation",
        "description": "LLM-as-judge rubric for RAG evaluation (faithfulness/precision/recall/fluency)",
        "local": EVALUATION_PROMPT,
    },
}

# tone -> {"text", "fetched_at", "version"}
_hub_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def push_all_prompts(commit_message: str = "sync from prompt_templates.py") -> Dict[str, bool]:
    """
    One-time (idempotent) upload of every registered prompt to LangSmith Hub.
    Each call creates a new commit on "<repo>:latest".

    Returns mapping tone -> success flag.
    """
    client = get_client()
    results: Dict[str, bool] = {}
    if client is None:
        logger.error("Cannot push prompts: LangSmith client unavailable.")
        return {tone: False for tone in PROMPT_REGISTRY}

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as exc:
        logger.error("langchain-core unavailable, cannot push prompts: %s", exc)
        return {tone: False for tone in PROMPT_REGISTRY}

    for tone, spec in PROMPT_REGISTRY.items():
        try:
            prompt_object = ChatPromptTemplate.from_messages([
                ("system", spec["local"]),
            ])
            client.push_prompt(
                spec["repo"],
                object=prompt_object,
                description=spec["description"],
                readme=(
                    "System prompt used by the customer support RAG chatbot "
                    f"(tone: '{tone}'). Managed by prompt_manager.py; edited "
                    "versions are picked up at runtime automatically."
                ),
                tags=[LANGSMITH_PROJECT, tone],
                commit_description=commit_message,
            )
            results[tone] = True
            logger.info("Pushed prompt '%s' (%s).", spec["repo"], commit_message)
            with _cache_lock:
                _hub_cache.pop(tone, None)  # force re-pull of the new commit
        except Exception as exc:
            results[tone] = False
            logger.warning("Failed to push prompt '%s': %s", spec["repo"], exc)

    return results


def _extract_template_text(pulled_prompt: Any) -> Optional[str]:
    """Pull the raw template string out of a pulled ChatPromptTemplate."""
    try:
        messages = getattr(pulled_prompt, "messages", None)
        if not messages:
            return None
        first = messages[0]
        inner = getattr(first, "prompt", None)  # MessagePromptTemplate wrapper
        text = getattr(inner, "template", None) or getattr(first, "content", None)
        return text if isinstance(text, str) else None
    except Exception:
        return None


def _pull_from_hub(tone: str) -> Optional[str]:
    """Fetch the latest prompt text from LangSmith Hub. None on any failure."""
    client = get_client()
    if client is None:
        return None
    repo = PROMPT_REGISTRY[tone]["repo"]
    try:
        pulled = client.pull_prompt(f"{repo}:latest")
    except Exception as exc:
        logger.debug("Hub pull failed for '%s': %s", repo, exc)
        return None

    text = _extract_template_text(pulled)
    if text:
        with _cache_lock:
            _hub_cache[tone] = {
                "text": text,
                "fetched_at": time.time(),
                "version": short_hash(text),
            }
        logger.info("Pulled prompt '%s' (version %s) from Hub.", repo, short_hash(text))
    return text


def _cached_hub_text(tone: str, allow_stale: bool = False) -> Optional[str]:
    """Return cached Hub text when fresh (or stale if allowed)."""
    with _cache_lock:
        entry = _hub_cache.get(tone)
        if entry is None:
            return None
        fresh = (time.time() - entry["fetched_at"]) < PROMPT_HUB_CACHE_TTL_SECONDS
        if fresh or allow_stale:
            return entry["text"]
    return None


def get_system_prompt(tone: str = "balanced", refresh: bool = False) -> str:
    """
    Resolve the active system prompt for a tone key.

    Order: fresh Hub cache -> Hub pull -> stale cache -> local constant.
    Never raises; always returns usable prompt text.
    """
    spec = PROMPT_REGISTRY.get(tone)
    if spec is None:
        logger.warning("Unknown prompt tone '%s', falling back to 'balanced'.", tone)
        return get_system_prompt("balanced", refresh=refresh)

    if not refresh:
        cached = _cached_hub_text(tone)
        if cached is not None:
            return cached

    pulled = _pull_from_hub(tone)
    if pulled is not None:
        return pulled

    stale = _cached_hub_text(tone, allow_stale=True)
    if stale is not None:
        logger.warning("Using stale Hub version for '%s' (Hub unreachable).", tone)
        return stale

    logger.info("Using local fallback prompt for tone '%s'.", tone)
    return spec["local"]


def get_prompt_version(tone: str) -> str:
    """Short hash of the currently active prompt text for a tone key."""
    text = get_system_prompt(tone)
    return short_hash(text)


def get_evaluation_prompt() -> str:
    """Active evaluation rubric prompt (Hub-first)."""
    return get_system_prompt("evaluation")


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    results = push_all_prompts()
    print(json.dumps({"pushed": results}, indent=2))

    for tone in ("strict", "friendly", "balanced", "evaluation"):
        text = get_system_prompt(tone, refresh=True)
        print(f"[{tone}] version={get_prompt_version(tone)} chars={len(text)}")

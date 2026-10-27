"""Input guardrails: a local injection filter plus OpenAI's free moderation endpoint."""
import logging
import re

from openai import OpenAI

# Common prompt-injection phrasings. Cheap and local; the <answer> tags in interview.py are the second layer.
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier)",
    r"(give|award|assign) me (a )?(10|ten|full|perfect)",
    r"you are now\b",
    r"(reveal|show|print) (your |the )?(system )?(prompt|instructions)",
    r"</?answer>",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def is_flagged(text: str) -> bool:
    """True if the moderation endpoint flags the text. Fails open so an outage doesn't block users."""
    try:
        result = OpenAI().moderations.create(model="omni-moderation-latest", input=text)
        return result.results[0].flagged
    except Exception:
        logging.exception("Moderation check failed")
        return False


def check_input(text: str) -> str | None:
    """Return a user-facing reason if the text should be rejected, else None."""
    if looks_like_injection(text):
        return "Your answer looks like it contains instructions for the AI. Please answer the question directly."
    if is_flagged(text):
        return "Your answer was flagged as inappropriate. Please rephrase it."
    return None

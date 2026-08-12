"""Prompt-injection screening.

**What this module is, and is not.** The pattern list below is a speed bump for
lazy attempts, not a security boundary. Anyone who wants to get past a regex
will, and the honest thing to do is design as if they already have. The actual
boundary is in ``app/tools/claims.py``: every tool validates its own arguments
with Pydantic, so a fully jailbroken model still cannot post a negative amount,
set a claim's status, or invent a policy number. Defence here is about reducing
noise and making the failure mode visible — not about being unbypassable.

Three surfaces are covered:

1. **User input** (:func:`screen_user_message`) — the obvious one.
2. **Retrieved document content** (:func:`scan_retrieved_text`) — the one that
   actually matters. If someone gets a line into a policy document, that text
   reaches the model with all the authority of a trusted source. We do *not*
   redact it: silently rewriting policy language would be worse than the
   attack. We flag it, log it, and rely on the prompt keeping retrieved text
   fenced off as data.
3. **Model output** (:func:`response_leaks_canary`) — a last-line check that
   the system prompt has not been echoed back.

Patterns are deliberately few and tight. Twenty loose patterns produce false
positives on ordinary insurance questions — "ignore the previous claim I
mentioned" is a sentence a real policyholder writes — and a guard that blocks
real customers is worse than no guard.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Distinctive string planted in the system prompt. If it ever appears in a
#: response, the model has been talked into reciting its instructions.
SYSTEM_PROMPT_CANARY = "OMNICARE-SYS-7F3A"

#: Zero-width and bidirectional control characters, used to break up keywords
#: ("i\u200bgnore all previous instructions") so a naive regex misses them.
_INVISIBLE = dict.fromkeys(
    (0x00AD, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF)
    + tuple(range(0x202A, 0x202F))
    + tuple(range(0x2066, 0x206A))
)

_WHITESPACE = re.compile(r"\s+")

#: (rule name, pattern). Each requires an instruction-like object, not just a
#: verb, which is what keeps ordinary questions out of the net.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.!?]{0,40}?"
            r"\b(?:previous|prior|above|earlier|initial|original|all)\b"
            r"[^.!?]{0,20}?\b(?:instruction|instructions|prompt|prompts|"
            r"rule|rules|directive|directives|guideline|guidelines)\b"
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(?:show|reveal|print|repeat|output|display|recite|echo|"
            r"tell\s+me|give\s+me|what\s+(?:was|were|is|are))\b[^.!?]{0,30}?"
            # Three orderings seen in the wild. Each still requires an
            # instruction-like object *plus* a qualifier, so "show me the rules
            # for jewelry coverage" stays allowed.
            r"(?:"
            r"\b(?:system|initial|original|hidden|secret)\b[^.!?]{0,15}?"
            r"\b(?:prompt|prompts|instruction|instructions|message|messages|rules)\b"
            r"|\byour\b[^.!?]{0,15}?"
            r"\b(?:prompt|prompts|instruction|instructions|directive|directives)\b"
            r"|\b(?:prompt|instruction|instructions|directive|directives|rules)\b"
            r"[^.!?]{0,25}?\b(?:you\s+(?:were\s+given|received|have\s+been\s+given)"
            r"|verbatim|word\s+for\s+word)\b"
            r")"
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on[,\s]+you|act\s+as\s+(?:if\s+"
            r"you\s+(?:are|were)|an?\s+)|pretend\s+(?:you\s+are|to\s+be)|"
            r"roleplay\s+as)\b"
        ),
    ),
    (
        "jailbreak_mode",
        re.compile(
            r"\b(?:developer\s+mode|dan\s+mode|god\s+mode|jailbreak|"
            r"unfiltered\s+mode|without\s+(?:any\s+)?(?:restrictions|filters|"
            r"limitations)|no\s+(?:restrictions|filters|guardrails))\b"
        ),
    ),
    (
        "fake_conversation_turn",
        re.compile(
            r"(?:<\|(?:im_start|im_end|system|endoftext)\|>|"
            r"\[/?(?:system|inst)\]|^\s*###\s*(?:system|instruction)|"
            r"^\s*(?:system|assistant)\s*:)",
            re.MULTILINE,
        ),
    ),
    (
        "new_instruction_injection",
        re.compile(
            r"\b(?:new|updated|revised|additional|real|actual)\s+"
            r"(?:instructions?|system\s+prompt|rules?|directives?)\s*[:\-]"
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(?:approve|authorise|authorize|mark|set)\b[^.!?]{0,30}?"
            r"\b(?:all|every|any)\b[^.!?]{0,20}?\bclaims?\b|"
            r"\bclaim\s+status\s+to\s+(?:approved|paid)\b"
        ),
    ),
)


@dataclass(frozen=True)
class GuardVerdict:
    """Outcome of a screening pass."""

    allowed: bool
    rule: str = ""
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = GuardVerdict(allowed=True)


def _fold(text: str, *, join: bool) -> str:
    """NFKC-fold, drop invisible characters, collapse whitespace, lowercase."""
    replacement = "" if join else " "
    folded = "".join(
        replacement if ord(ch) in _INVISIBLE else ch
        for ch in unicodedata.normalize("NFKC", text)
    )
    return _WHITESPACE.sub(" ", folded).strip().lower()


def normalise(text: str) -> str:
    """Fold away the tricks used to smuggle keywords past a regex.

    NFKC folds full-width and styled look-alikes onto ASCII; invisible
    characters are dropped; runs of whitespace collapse. Applied only for
    *matching* — the model always sees the user's original text.
    """
    return _fold(text, join=True)


def _probes(text: str) -> tuple[str, str]:
    """Both readings of an invisible-character split.

    ``ignore\\u00adall previous instructions`` is ambiguous: dropping the soft
    hyphen glues "ignoreall" together and hides the keyword, while replacing it
    with a space reveals it. Which one the attacker intended does not matter —
    we test both and block if either matches.
    """
    return _fold(text, join=True), _fold(text, join=False)


def _matched_rule(text: str) -> str | None:
    joined, split = _probes(text)
    for name, pattern in _RULES:
        if pattern.search(joined) or (split != joined and pattern.search(split)):
            return name
    return None


def screen_user_message(message: str, *, max_chars: int = 2000) -> GuardVerdict:
    """Screen an inbound user message.

    Length is checked first: a very long message is the cheapest way to push
    the system prompt out of an attention window, and it is unambiguous.
    """
    if not message or not message.strip():
        return GuardVerdict(False, "empty", "The message is empty.")

    if len(message) > max_chars:
        return GuardVerdict(
            False,
            "message_too_long",
            f"The message is {len(message)} characters; the limit is {max_chars}.",
        )

    if (rule := _matched_rule(message)) is not None:
        logger.warning("Blocked user message on rule %s", rule)
        return GuardVerdict(
            False, rule, f"The message matched the {rule} screening rule."
        )

    return ALLOWED


def scan_retrieved_text(text: str) -> str | None:
    """Return the rule name if retrieved document content looks like an
    instruction, else ``None``.

    Nothing is redacted. A policy clause legitimately contains imperative
    language, and quietly deleting part of a policy would be a worse failure
    than the injection. The caller logs the hit and keeps the passage fenced.
    """
    return _matched_rule(text)


def response_leaks_canary(response: str) -> bool:
    """True if the model echoed the marker planted in the system prompt."""
    return SYSTEM_PROMPT_CANARY.lower() in normalise(response)

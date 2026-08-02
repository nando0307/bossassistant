"""Defences for untrusted text that arrives through retrieval.

The enterprise-specific attack on a RAG assistant is **indirect prompt
injection**: nobody types the attack, it is already sitting inside a document
the retriever is supposed to find. Someone edits a policy page, a vendor
uploads a PDF, an intranet article gets scraped — and a sentence in that
document addresses the model instead of the reader.

The defence is layered, because each layer alone is defeatable:

1. **Spotlighting.** Retrieved text is fenced in an explicit delimiter and the
   system prompt states that everything inside is data quoted from a document,
   never instructions. This is what makes "ignore previous instructions" read as
   a string rather than a command.
2. **Delimiter integrity.** Layer 1 is worthless if a document can close the
   fence itself, so any occurrence of the delimiter inside retrieved text is
   neutralised before the prompt is assembled.
3. **Detection, not silent deletion.** Instruction-shaped spans are flagged and
   reported rather than quietly rewritten. Deleting text from a policy document
   changes what the assistant says a policy contains, which is its own failure;
   an operator seeing "3 suspicious spans in FIN-014" learns something, while a
   silently scrubbed corpus teaches nobody anything.

There is deliberately **no tool-calling on the answer path**, so retrieved text
cannot trigger an action even if it persuades the model. `test_security.py`
asserts that property rather than trusting it to stay true.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: The fence retrieved text is wrapped in. Chosen to be visually obvious in
#: traces and unlikely to appear in a policy document by accident.
CONTEXT_OPEN = "<untrusted_policy_context>"
CONTEXT_CLOSE = "</untrusted_policy_context>"

#: Spans that look like they are addressing the model rather than the reader.
#: Matched case-insensitively. This list is a detector, not a sanitiser — see
#: the module docstring on why matches are flagged rather than deleted.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction-override", re.compile(r"\b(ignore|disregard|forget)\b[^.]{0,40}\b(previous|prior|above|earlier|all)\b[^.]{0,30}\b(instruction|prompt|rule|direction)", re.I)),
    ("role-reassignment", re.compile(r"\byou are (now|no longer)\b|\bact as\b|\bpretend to be\b|\bfrom now on,? you\b", re.I)),
    ("system-impersonation", re.compile(r"^\s*(system|assistant|user)\s*:", re.I | re.M)),
    ("fence-breakout", re.compile(r"</?untrusted_policy_context>|<\|.*?\|>", re.I)),
    ("exfiltration", re.compile(r"\b(send|post|email|forward|upload)\b[^.]{0,40}\b(to|at)\b[^.]{0,20}(https?://|www\.|@)", re.I)),
    ("secret-solicitation", re.compile(r"\b(reveal|print|repeat|output|show)\b[^.]{0,30}\b(system prompt|your instructions|api[_ ]?key|password|token)\b", re.I)),
    ("authority-claim", re.compile(r"\b(this (is|was) (an? )?(official|authorised|authorized|admin))\b|\boverrides? (the )?(policy|previous|security)\b", re.I)),
)


@dataclass(frozen=True)
class InjectionFinding:
    source: str
    pattern: str
    excerpt: str


def scan_for_injection(text: str, source: str = "?") -> list[InjectionFinding]:
    """Report instruction-shaped spans in a retrieved chunk."""
    findings: list[InjectionFinding] = []
    for name, pattern in _INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            start = max(match.start() - 20, 0)
            excerpt = " ".join(text[start : match.end() + 40].split())
            findings.append(InjectionFinding(source=source, pattern=name, excerpt=excerpt))
    return findings


def neutralize_delimiters(text: str) -> str:
    """Stop retrieved text from closing the fence it is wrapped in.

    Without this, spotlighting is theatre: a document containing the closing
    delimiter followed by its own instructions would place those instructions
    outside the quoted region, where the model reads them as coming from the
    system rather than from a document.

    Angle brackets are replaced rather than stripped so the tampering stays
    visible in the prompt and in traces instead of vanishing.
    """
    neutralized = text.replace("<", "‹").replace(">", "›")
    # Also defuse chat-template control tokens, which some models honour even
    # mid-message and which never legitimately appear in a policy document.
    return re.sub(r"‹\|(.*?)\|›", r"[control-token: \1]", neutralized)


def wrap_untrusted(body: str) -> str:
    """Fence retrieved text so the model can tell data from instructions."""
    return f"{CONTEXT_OPEN}\n{body}\n{CONTEXT_CLOSE}"

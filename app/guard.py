"""Prompt-injection guard (innovation extra, cf. FR-1006).

Pattern-based screening of untrusted text (tool arguments, retrieved
content, LLM output). This is a heuristic tripwire, not a proof of safety:
hits are flagged in the audit log and surfaced on node outputs so operators
can see them; blocking is configurable per call site.
"""
import re

_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous) prompt",
    r"you are now (in )?(developer|dan|jailbreak)",
    r"reveal (the )?(system prompt|api key|credentials)",
    r"exfiltrate|send (all )?(data|secrets) to",
    r"override (the )?(safety|security) (rules|policy)",
]

_COMPILED = [re.compile(p, re.I) for p in _PATTERNS]


def scan(text) -> list[str]:
    """Return the list of matched pattern descriptions (empty = clean)."""
    if not isinstance(text, str):
        return []
    return [p.pattern for p in _COMPILED if p.search(text)]


def scan_value(value) -> list[str]:
    hits: list[str] = []
    if isinstance(value, str):
        hits.extend(scan(value))
    elif isinstance(value, dict):
        for v in value.values():
            hits.extend(scan_value(v))
    elif isinstance(value, list):
        for v in value:
            hits.extend(scan_value(v))
    return hits

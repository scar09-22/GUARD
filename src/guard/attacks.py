"""Text perturbations that stress the conformal certificate (E4) and detection power.

The certificate's only assumption is exchangeability of human calibration and human TEST
text. These attacks break it in two deployment-realistic ways:

- `fw_perturb`: random delete / duplicate / swap of FUNCTION WORDS. Function-word statistics
  are exactly what likelihood-based detectors lean on, so this is a worst-case lexical shift
  for the scorer while barely changing content.
- `synonym_perturb`: WordNet synonym substitution on content words — the proxy for INNOCENT
  tool-assisted human editing (grammar/paraphrase tools). Applied to human test text it is a
  certificate attack (E4, the headline experiment); applied to AI text it is an evasion
  attack (power degradation only — validity is untouched because the humans are unchanged).

All perturbations are deterministic given (text, rate, seed): each text derives its own RNG
stream from the pair (seed, text), so frame-level attacks are reproducible row by row and
independent of row order. Outputs are whitespace-normalised, matching data.py's convention.

Self-contained: only the standard library plus NLTK's WordNet corpus (fetched on first use
if absent).
"""
from __future__ import annotations

import random
import re

import pandas as pd

__all__ = [
    "FUNCTION_WORDS",
    "fw_perturb",
    "synonym_perturb",
    "cleanup_perturb",
    "perturb_frame",
]

# Curated closed-class vocabulary (~80 words): articles, prepositions, conjunctions,
# auxiliaries/copulas, pronouns, and a few high-frequency adverbial particles. Frozen so the
# attack surface is identical across experiments and releases.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        # articles / determiners
        "the", "a", "an", "this", "that", "these", "those", "some", "any", "each",
        "every", "no", "all", "both",
        # prepositions
        "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "over", "under", "off", "near",
        # conjunctions / subordinators
        "and", "or", "but", "nor", "so", "yet", "if", "because", "while", "although",
        "than", "as", "when", "where", "since",
        # auxiliaries / copulas / modals
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
        "have", "has", "had", "will", "would", "shall", "should", "can", "could",
        "may", "might", "must",
        # pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "there",
        # high-frequency particles / negation
        "not", "very", "too", "then",
    }
)

# A token is a leading-punctuation / word-core / trailing-punctuation triple; the core may
# contain internal apostrophes or hyphens ("don't", "well-lit").
_TOKEN_RE = re.compile(r"^([^\w]*)([\w][\w'\-]*)?([^\w]*)$", re.UNICODE)

_FW_LIST: tuple[str, ...] = tuple(sorted(FUNCTION_WORDS))  # stable order for rng.choice


def _rng_for(text: str, rate: float, seed: int) -> random.Random:
    """Per-text RNG: deterministic across runs and independent of row order in a frame.

    random.Random seeded with a str hashes it with SHA-512 internally (CPython seed
    version 2), so this does NOT depend on PYTHONHASHSEED.
    """
    return random.Random(f"{seed}|{rate}|{text}")


def _split_token(token: str) -> tuple[str, str, str]:
    """Split a whitespace token into (leading punctuation, word core, trailing punctuation)."""
    m = _TOKEN_RE.match(token)
    if m is None or m.group(2) is None:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def _match_case(replacement: str, original: str) -> str:
    """Carry the original word's simple casing onto the replacement."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def fw_perturb(text: str, rate: float, seed: int) -> str:
    """Perturb function-word tokens: with prob = rate, delete / duplicate / swap each one.

    - delete: drop the token entirely;
    - duplicate: emit it twice ("the the");
    - swap: replace it with a different uniformly chosen function word (casing preserved).
    The three operations are chosen uniformly per hit. Content words are never touched.
    Deterministic given (text, rate, seed).
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    rng = _rng_for(text, rate, seed)
    out: list[str] = []
    for token in str(text).split():
        pre, core, post = _split_token(token)
        if core.lower() in FUNCTION_WORDS and rng.random() < rate:
            op = rng.choice(("delete", "duplicate", "swap"))
            if op == "delete":
                continue
            if op == "duplicate":
                out.append(token)
                out.append(token)
                continue
            # swap with a different function word
            candidates = [w for w in _FW_LIST if w != core.lower()]
            new_core = _match_case(rng.choice(candidates), core)
            out.append(pre + new_core + post)
        else:
            out.append(token)
    return " ".join(out)


# --------------------------- WordNet-backed synonym attack --------------------------- #

_WORDNET = None  # lazy singleton
_SYN_CACHE: dict[str, tuple[str, ...]] = {}


def _get_wordnet():
    """Import NLTK WordNet lazily, downloading the corpus on first use if needed."""
    global _WORDNET
    if _WORDNET is not None:
        return _WORDNET
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError as exc:  # pragma: no cover - environment misconfiguration
        raise RuntimeError(
            "synonym_perturb requires nltk; install it in the active environment"
        ) from exc
    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        wordnet.ensure_loaded()
    _WORDNET = wordnet
    return _WORDNET


def _synonyms(word: str) -> tuple[str, ...]:
    """Sorted single-word WordNet lemmas for `word` (lowercase), excluding the word itself."""
    lw = word.lower()
    cached = _SYN_CACHE.get(lw)
    if cached is not None:
        return cached
    wn = _get_wordnet()
    syns: set[str] = set()
    for synset in wn.synsets(lw):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name != lw and "_" not in name and name.isalpha():
                syns.add(name)
    result = tuple(sorted(syns))  # sorted => deterministic rng.choice across runs
    _SYN_CACHE[lw] = result
    return result


def synonym_perturb(text: str, rate: float, seed: int) -> str:
    """Replace non-function alphabetic words with a WordNet synonym with prob = rate.

    Proxy for innocent tool-assisted human editing: content words are paraphrased, function
    words and punctuation are left intact. Words with no single-word synonym are skipped
    (the probability draw is still consumed, keeping streams aligned across rates).
    Deterministic given (text, rate, seed).
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    rng = _rng_for(text, rate, seed)
    out: list[str] = []
    for token in str(text).split():
        pre, core, post = _split_token(token)
        if (
            core.isalpha()
            and core.lower() not in FUNCTION_WORDS
            and rng.random() < rate
        ):
            options = _synonyms(core)
            if options:
                out.append(pre + _match_case(rng.choice(options), core) + post)
                continue
        out.append(token)
    return " ".join(out)


_SPELL = None  # lazy singleton
_SPELL_CACHE: dict[str, str] = {}


def _get_spell():
    global _SPELL
    if _SPELL is None:
        from spellchecker import SpellChecker
        _SPELL = SpellChecker(distance=2)  # distance-2 catches e.g. atmosfere -> atmosphere
    return _SPELL


def cleanup_perturb(text: str, rate: float, seed: int) -> str:
    """Fluency-INCREASING editor: the realistic 'innocent tool-assisted writing' proxy.

    Unlike fw_perturb / synonym_perturb (which add lexical noise and make text LESS fluent,
    moving humans AWAY from the AI-like score region), this models what grammar/spell tools
    actually do first: correct misspellings and normalise spacing/casing/punctuation, which
    RAISES likelihood under a language model and therefore pushes human text TOWARD the
    flagging region. `rate` is the probability that each detected misspelling is corrected
    (rate=1.0 -> full cleanup); whitespace normalisation always applies for rate > 0.
    """
    if rate <= 0:
        return text
    rng = _rng_for(text, rate, seed)
    spell = _get_spell()
    out = []
    for token in str(text).split():
        pre, core, post = _split_token(token)
        if core and core.isalpha() and len(core) > 3 and rng.random() < rate:
            low = core.lower()
            if low not in _SPELL_CACHE:
                corr = spell.correction(low) if low in spell.unknown([low]) else low
                _SPELL_CACHE[low] = corr if corr else low
            fixed = _SPELL_CACHE[low]
            if fixed != low:
                core = _match_case(fixed, core)
        out.append(f"{pre}{core}{post}")
    cleaned = " ".join(out)
    # Normalise repeated punctuation and stray spacing (what editors do silently).
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([.,!?;:]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def perturb_frame(df: pd.DataFrame, kind: str, rate: float, seed: int) -> pd.DataFrame:
    """Return a copy of `df` with the text column attacked; all other columns untouched.

    kind: "fw" (function-word perturbation), "synonym" (WordNet substitution), or
    "cleanup" (fluency-increasing spell/punctuation normalisation).
    Each row's perturbation is keyed to its own text, so results do not depend on row order.
    """
    fns = {"fw": fw_perturb, "synonym": synonym_perturb, "cleanup": cleanup_perturb}
    if kind not in fns:
        raise ValueError(f"kind must be one of {sorted(fns)}, got {kind!r}")
    fn = fns[kind]
    out = df.copy()
    out["text"] = [fn(t, rate, seed) for t in out["text"]]
    return out

"""Rule-based verifier for GSM8K-style answers.

The single source of truth for "is this answer correct". Used by:
  - the data-generation filter (partner's Mario rewrites must pass this),
  - eval.py,
  - later, the RLVR reward.

Convention: the final answer is the number after the last "####" marker, exactly as in
GSM8K. Anything without a "####" marker is scored as wrong (strict) so that the model is
also being trained/evaluated on producing a parseable answer.
"""

import re

_MARKER_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str, lenient: bool = False) -> float | None:
    """Return the numeric final answer from a completion, or None if not found.

    strict (default): number after the last "####".
    lenient: fall back to the last number anywhere in the text. Use only for diagnostics,
             never for the headline metric.
    """
    matches = _MARKER_RE.findall(text)
    if matches:
        return _to_float(matches[-1])
    if lenient:
        nums = _NUMBER_RE.findall(text)
        if nums:
            return _to_float(nums[-1])
    return None


def gold_answer(gsm8k_answer_field: str) -> float:
    """Parse the gold answer from a GSM8K 'answer' field (solution text ending in '#### N')."""
    val = extract_answer(gsm8k_answer_field)
    if val is None:
        raise ValueError(f"No '####' answer in gold field: {gsm8k_answer_field[:80]!r}")
    return val


def is_correct(completion: str, gold: float | str, lenient: bool = False) -> bool:
    """True if the completion's final answer equals the gold answer."""
    if isinstance(gold, str):
        gold = gold_answer(gold) if "####" in gold else _to_float(gold)
        if gold is None:
            raise ValueError("Unparseable gold answer")
    pred = extract_answer(completion, lenient=lenient)
    if pred is None:
        return False
    return abs(pred - gold) < 1e-6


if __name__ == "__main__":
    # Quick self-test.
    cases = [
        ("She has 3 + 4 = 7 apples.\n#### 7", "7", True),
        ("#### $1,250", "1250", True),
        ("#### 1250.", "1250", True),
        ("Wahoo! The answer is 18.\n#### 18", "#### 18", True),
        ("#### 17.5", "17.5", True),
        ("#### -3", "-3", True),
        ("#### 7", "8", False),
        ("The answer is 7.", "7", False),          # no marker -> wrong in strict mode
        ("#### 7\nActually #### 8", "8", True),     # last marker wins
    ]
    for text, gold, want in cases:
        got = is_correct(text, gold)
        assert got == want, (text, gold, got, want)
    assert is_correct("The answer is 7.", "7", lenient=True)
    print("verifier self-test passed")

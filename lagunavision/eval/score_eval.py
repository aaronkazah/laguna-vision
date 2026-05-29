from __future__ import annotations

from lagunavision.types import EvalManifestItem, ScoreBreakdown


CAUSE_TERMS = (
    "because",
    "caused",
    "cause",
    "missing",
    "not found",
    "undefined",
    "failed",
    "error",
    "cannot",
    "mismatch",
)


def score_answer(item: EvalManifestItem, answer: str) -> ScoreBreakdown:
    normalized = answer.casefold()
    read_key_text = any(term.casefold() in normalized for term in item.must_include)
    gave_fix = any(term.casefold() in normalized for term in item.accepted_fix_terms)
    violated_negative = any(term.casefold() in normalized for term in item.must_not_include)
    if item.rubric == "vqa":
        identified_cause = read_key_text
        gave_fix = read_key_text
    elif item.rubric == "description":
        identified_cause = gave_fix
        gave_fix = read_key_text and identified_cause
    else:
        identified_cause = read_key_text and any(term in normalized for term in CAUSE_TERMS)
    return ScoreBreakdown(
        item_id=item.id,
        read_key_text=read_key_text,
        identified_cause=identified_cause,
        gave_fix=gave_fix,
        violated_negative=violated_negative,
    )

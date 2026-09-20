from dataclasses import dataclass
import re

from agent.nodes.criteria import clear_criteria, normalize_criteria


@dataclass(frozen=True)
class CriteriaCommand:
    action: str
    fields: tuple[str, ...] = ()
    remainder: str = ""

    def apply(self, criteria):
        return (
            normalize_criteria({})
            if self.action == "reset"
            else clear_criteria(criteria, self.fields)
        )


FIELD_PATTERNS = {
    "date": r"(?:\d{4}['’]?s|\d{2}['’]?s|\d{4}|release dates?|dates?|years?|decades?|time periods?)",
    "duration": r"(?:runtime|duration|length)",
    "rating": r"(?:minimum rating|rating)",
    "genres": r"genres?",
    "actors": r"actors?",
    "directors": r"directors?",
    "streaming": r"(?:streaming|platforms?|services?)",
    "languages": r"languages?",
    "themes": r"(?:themes?|moods?)",
}


def parse_criteria_command(request: str) -> CriteriaCommand | None:
    text = request.strip().replace("’", "'")
    prefix = r"(?:(?:please|no|okay|ok)[,!]?\s+)*"
    boundary = r"(?=$|[,;.!?]|\s+(?:and|but|instead)\b)"
    for field, target in FIELD_PATTERNS.items():
        pattern = rf"{prefix}(?:forget(?: about)?|remove|clear|drop|ignore|disregard|never ?mind)\s+(?:(?:the|my|that)\s+)?{target}(?:\s+(?:filter|constraint|restriction))?{boundary}"
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return CriteriaCommand("remove", (field,), _remainder(text[match.end() :]))
    match = re.match(
        rf"{prefix}(?:any year(?: is fine)?|(?:the )?(?:year|date) (?:doesn't|does not) matter){boundary}",
        text,
        re.IGNORECASE,
    )
    if match:
        return CriteriaCommand("remove", ("date",), _remainder(text[match.end() :]))
    reset = r"(?:start over|start again|restart|reset|never ?mind|clear all|(?:forget(?: about)?|reset|clear)\s+(?:(?:all|my|the|previous|search)\s+)*(?:it|this|that|everything|filters|preferences|criteria|brief|search))"
    match = re.match(rf"{prefix}{reset}{boundary}", text, re.IGNORECASE)
    if match:
        remainder = _remainder(text[match.end() :])
        if re.fullmatch(
            r"(?:let's|let us|i want to|can we) (?:find|search for|look for) (?:another movie|other movies|a different movie)",
            remainder,
            re.IGNORECASE,
        ):
            remainder = ""
        return CriteriaCommand("reset", remainder=remainder)
    return None


def _remainder(text):
    text = text.strip(" ,;.!?")
    return re.sub(r"^(?:and|but|instead)\s+", "", text, flags=re.IGNORECASE)

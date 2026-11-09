import re
from collections import Counter
from dataclasses import dataclass

from app.documents.models import DocumentBlock
from app.reviews.models import BoundaryReason, ContractClause


_DIGIT = r"[0-9๐-๙]"
_NUMBER = rf"{_DIGIT}+"
_DOTTED_NUMBER = rf"{_NUMBER}(?:\s*\.\s*{_NUMBER})+"
_THAI_CLAUSE_PATTERN = re.compile(
    rf"^\s*ข้อ\s*(?P<number>{_NUMBER}(?:\s*\.\s*{_NUMBER})*)"
    rf"\s*(?:[.)]\s*)?(?P<title>.*)$"
)
_DOTTED_PATTERN = re.compile(
    rf"^\s*(?P<number>{_DOTTED_NUMBER})\s*(?:[.)]\s*)?(?P<title>.*)$"
)
_SINGLE_PATTERN = re.compile(
    rf"^\s*(?P<number>{_NUMBER})[.)]\s*(?P<title>.*)$"
)
_BARE_HEADING_PATTERN = re.compile(
    rf"^\s*(?P<number>{_NUMBER})\s+(?P<title>\S.*)$"
)
_HEADING_LEVEL_PATTERN = re.compile(r"Heading\s+(?P<level>\d+)", re.IGNORECASE)
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


class ClauseSegmentationError(ValueError):
    """Raised when clause boundaries cannot preserve every source block."""


@dataclass(frozen=True)
class _NumberedClause:
    number: str
    title: str


def _numbered_clause(
    text: str,
    *,
    allow_bare_heading_number: bool = False,
) -> _NumberedClause | None:
    first_line = text.splitlines()[0].strip()
    patterns = [
        _THAI_CLAUSE_PATTERN,
        _DOTTED_PATTERN,
        _SINGLE_PATTERN,
    ]
    if allow_bare_heading_number:
        patterns.append(_BARE_HEADING_PATTERN)
    for pattern in patterns:
        match = pattern.match(first_line)
        if match is None:
            continue
        number = re.sub(r"\s+", "", match.group("number")).translate(
            _THAI_DIGITS
        )
        title = match.group("title").strip() or first_line
        return _NumberedClause(number=number, title=title)
    return None


def _is_boundary(block: DocumentBlock) -> bool:
    return block.type == "heading" or _numbered_clause(block.text) is not None


def _group_blocks(blocks: list[DocumentBlock]) -> list[list[DocumentBlock]]:
    if not blocks:
        return []

    if not any(_is_boundary(block) for block in blocks):
        return [[block] for block in blocks]

    groups: list[list[DocumentBlock]] = []
    current: list[DocumentBlock] = []
    for block in blocks:
        if _is_boundary(block) and current:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)

    # A structural heading immediately followed by a numbered provision is
    # context for that provision, not a standalone review query.
    merged: list[list[DocumentBlock]] = []
    pending_headings: list[DocumentBlock] = []
    for group in groups:
        if all(block.type == "heading" for block in group):
            pending_headings.extend(group)
            continue
        merged.append([*pending_headings, *group])
        pending_headings = []
    if pending_headings:
        if merged:
            merged[-1].extend(pending_headings)
        else:
            merged.append(pending_headings)
    return merged


def _update_heading_path(
    heading_path: list[str],
    block: DocumentBlock,
) -> list[str]:
    match = _HEADING_LEVEL_PATTERN.fullmatch(block.style or "")
    if match is None:
        return [block.text]
    level = int(match.group("level"))
    return [*heading_path[: level - 1], block.text]


def segment_contract(blocks: list[DocumentBlock]) -> list[ContractClause]:
    """Split ordered parser blocks into deterministic, hierarchical clauses."""

    non_empty_blocks = [block for block in blocks if block.text.strip()]
    if not non_empty_blocks:
        raise ClauseSegmentationError("The document contains no reviewable text")

    input_indices = [block.index for block in non_empty_blocks]
    if len(input_indices) != len(set(input_indices)):
        raise ClauseSegmentationError("Document block indices must be unique")

    groups = _group_blocks(non_empty_blocks)
    has_explicit_boundary = any(_is_boundary(block) for block in non_empty_blocks)
    clauses: list[ContractClause] = []
    clauses_by_number: dict[str, ContractClause] = {}
    heading_path: list[str] = []

    for order, group in enumerate(groups):
        headings = [block for block in group if block.type == "heading"]
        for heading in headings:
            heading_path = _update_heading_path(heading_path, heading)

        numbered = next(
            (
                parsed
                for block in group
                if (
                    parsed := _numbered_clause(
                        block.text,
                        allow_bare_heading_number=block.type == "heading",
                    )
                )
                is not None
            ),
            None,
        )
        clause_id = f"clause-{order + 1:04d}"
        parent_clause: ContractClause | None = None
        if numbered is not None and "." in numbered.number:
            parent_number = numbered.number.rsplit(".", 1)[0]
            parent_clause = clauses_by_number.get(parent_number)

        if numbered is not None:
            title = numbered.title
            reason = (
                BoundaryReason.HEADING_AND_NUMBERED
                if headings
                else BoundaryReason.NUMBERED_CLAUSE
            )
        elif headings:
            title = headings[-1].text
            reason = BoundaryReason.HEADING
        elif has_explicit_boundary:
            title = None
            reason = BoundaryReason.DOCUMENT_START
        else:
            title = None
            reason = BoundaryReason.UNSTRUCTURED_BLOCK

        section_path = list(heading_path)
        if parent_clause is not None:
            for part in parent_clause.section_path:
                if part not in section_path:
                    section_path.append(part)
        if title and (not section_path or section_path[-1] != title):
            section_path.append(title)

        clause = ContractClause(
            clause_id=clause_id,
            order=order,
            number=numbered.number if numbered else None,
            title=title,
            text="\n\n".join(block.text.strip() for block in group),
            block_indices=[block.index for block in group],
            parent_clause_id=(
                parent_clause.clause_id if parent_clause is not None else None
            ),
            section_path=section_path,
            boundary_reason=reason,
        )
        clauses.append(clause)
        if numbered is not None:
            clauses_by_number[numbered.number] = clause

    validate_clause_coverage(non_empty_blocks, clauses)
    return clauses


def validate_clause_coverage(
    blocks: list[DocumentBlock],
    clauses: list[ContractClause],
) -> None:
    """Require every non-empty parser block to occur in exactly one clause."""

    expected = [block.index for block in blocks if block.text.strip()]
    actual = [index for clause in clauses for index in clause.block_indices]
    counts = Counter(actual)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))

    if duplicates or missing or unexpected or len(actual) != len(expected):
        raise ClauseSegmentationError(
            "Clause coverage is invalid: "
            f"missing={missing}, duplicates={duplicates}, unexpected={unexpected}"
        )


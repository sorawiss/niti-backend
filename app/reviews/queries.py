from app.documents.models import DocumentBlock
from app.reviews.models import ClauseQuery, ContractClause


class ClauseQueryBuilder:
    def __init__(self, *, max_query_chars: int = 4000) -> None:
        if max_query_chars < 256:
            raise ValueError("max_query_chars must be at least 256")
        self._max_query_chars = max_query_chars

    def build(
        self,
        clause: ContractClause,
        blocks: list[DocumentBlock],
    ) -> list[ClauseQuery]:
        blocks_by_index = {block.index: block for block in blocks}
        source_blocks = [blocks_by_index[index] for index in clause.block_indices]

        context = " > ".join(clause.section_path).strip()
        # Keep context bounded so every query retains room for source text.
        max_context_chars = self._max_query_chars // 3
        prefix = context[:max_context_chars]
        content_limit = max(64, self._max_query_chars - len(prefix) - 1)

        pieces: list[tuple[int, str]] = []
        for block in source_blocks:
            pieces.extend(
                (block.index, piece)
                for piece in self._split_text(block.text.strip(), content_limit)
            )

        windows: list[tuple[list[int], str]] = []
        window_indices: list[int] = []
        window_parts: list[str] = []
        window_length = 0
        for block_index, piece in pieces:
            added_length = len(piece) + (1 if window_parts else 0)
            if window_parts and window_length + added_length > content_limit:
                windows.append((list(dict.fromkeys(window_indices)), "\n".join(window_parts)))
                window_indices = []
                window_parts = []
                window_length = 0
            window_indices.append(block_index)
            window_parts.append(piece)
            window_length += len(piece) + (1 if len(window_parts) > 1 else 0)
        if window_parts:
            windows.append((list(dict.fromkeys(window_indices)), "\n".join(window_parts)))

        queries: list[ClauseQuery] = []
        for query_index, (block_indices, content) in enumerate(windows, start=1):
            text = f"{prefix}\n{content}" if prefix else content
            queries.append(
                ClauseQuery(
                    query_id=f"{clause.clause_id}-q{query_index:02d}",
                    clause_id=clause.clause_id,
                    text=text,
                    block_indices=block_indices,
                )
            )
        if not queries:
            raise ValueError(f"Clause {clause.clause_id} produced no retrieval query")
        return queries

    @staticmethod
    def _split_text(text: str, limit: int) -> list[str]:
        pieces: list[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind(" ", 0, limit + 1)
            if split_at <= 0:
                split_at = limit
            pieces.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces


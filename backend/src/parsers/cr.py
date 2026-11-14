import json
import uuid
from typing import Any, Iterator, List, Mapping
from .base import Parser
from logging import getLogger
from src.utils.langchain_utils import calculate_expected_tokens, get_default_splitter
from src.config import settings


_logger = getLogger(__name__)


class ComprehensiveRulesParser(Parser):
    """
    Parses the Academy Ruins comprehensive rules JSON.

    The source is a flat object keyed by rule number, where each entry carries
    `ruleNumber`, `ruleText` and an optional `examples` list. Examples are folded
    into the rule text under an EXAMPLES header so a rule stays a single document.

    Every chunk gets a deterministic id from its rule number and text, so unchanged
    chunks keep their id when the file is re-parsed. Embedding is left to the caller.
    """

    EXAMPLES_HEADER = "EXAMPLES"

    def parse(self, raw: bytes) -> Iterator[List[Mapping[str, Any]]]:
        data = json.loads(raw)

        for entry in data.values():
            rule_number = entry.get("ruleNumber")
            rule_text = entry.get("ruleText") or ""
            examples = entry.get("examples")

            if examples:
                rule_text = f"{rule_text}\n\n{self.EXAMPLES_HEADER}\n" + "\n".join(examples)

            yield [{
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rule_number}:{chunk}")),
                "rule_number": rule_number,
                "rule_text": chunk,
            } for chunk in self._split(rule_text)]

    def _split(self, rule_text: str) -> list[str]:
        """Splits the rule text into chunks that fit the embedding model's token limit."""
        if calculate_expected_tokens(rule_text) < settings.ollama_max_tokens:
            return [rule_text]

        _logger.info(f"Rule text exceeds max token limit ({settings.ollama_max_tokens}). Splitting into chunks for embedding.")

        chunks = []
        for chunk in rule_text.split("\n\n"):
            if calculate_expected_tokens(chunk) < settings.ollama_max_tokens:
                chunks.append(chunk)
            else:
                chunks.extend(get_default_splitter().split_text(chunk))
        return chunks

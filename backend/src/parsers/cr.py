import json
from typing import Any, Dict, Iterator, List, Mapping
from .base import Parser
from logging import getLogger
from utils.langchain_utils import ollama_embedder, calculate_expected_tokens, get_default_splitter
from config import settings


_logger = getLogger(__name__)

type ParsedEmbeddings = list[tuple[str, list[float]]]


class ComprehensiveRulesParser(Parser):
    """
    Parses the Academy Ruins comprehensive rules JSON.

    The source is a flat object keyed by rule number, where each entry carries
    `ruleNumber`, `ruleText` and an optional `examples` list. Examples are folded
    into the rule text under an EXAMPLES header so a rule stays a single document.
    """

    EXAMPLES_HEADER = "EXAMPLES"

    def parse(self, raw: bytes) -> Iterator[List[Mapping[str, Any]]]:
        data = json.loads(raw)

        for rule_number, entry in data.items():
            rule_text = entry.get("ruleText") or ""
            examples = entry.get("examples")

            if examples:
                examples = self.EXAMPLES_HEADER.join("\n".join(examples))
                rule_text = "\n\n".join([rule_text, *examples])
                del examples

            parse_results: ParsedEmbeddings = self._split_and_embed(rule_text)
            
            try: 
                yield [{
                    "rule_number": entry.get("ruleNumber"),
                    "rule_text": rule,
                    "embedding": embedding
                } for rule, embedding in parse_results]
            except KeyError as e:
                _logger.warning(f"Missing expected key in entry for rule {rule_number}: {e} \n skipping")
                continue
                

    def _split_and_embed(self, rule_text: str) -> ParsedEmbeddings:
        """
        Splits the rule text into chunks and generates embeddings for each chunk.

        Returns a list of tuples containing the chunk text and its corresponding embedding.
        """


        if calculate_expected_tokens(rule_text) < settings.ollama_max_tokens:
            return [(rule_text, ollama_embedder.embed_query(rule_text))]

        _logger.info(f"Rule text exceeds max token limit ({settings.ollama_max_tokens}). Splitting into chunks for embedding.")

        chunks = rule_text.split("\n\n")  

        data = []

        for chunk in chunks:
            if calculate_expected_tokens(chunk) < settings.ollama_max_tokens:
                data.append((chunk, ollama_embedder.embed_query(chunk)))
            else:
                data.extend(self._fallback_split_and_embed(chunk))

        return data
    

    def _fallback_split_and_embed(self, chunk: str) -> ParsedEmbeddings:
        """
        Fallback method to split the rule text into smaller chunks and generate embeddings for each chunk.

        Returns a list of tuples containing the chunk text and its corresponding embedding.
        """
        fallback_splitter = get_default_splitter()
        chunks = fallback_splitter.split_text(chunk)

        embeddings = [ollama_embedder.embed_query(chunk) for chunk in chunks]

        return list(zip(chunks, embeddings))
import json
from unittest.mock import MagicMock

import pytest
import typing

import scripts.tokenize as tokenize
from src.parsers import ComprehensiveRulesParser
from src.qmodels import Rule
from src.utils.qdrant import QdrantSession
from src.connections import StorageConnection
from tests.mocks.ollama import FakeOllamaEmbedder


RULES = {
    "100.1": {"ruleNumber": "100.1", "ruleText": "First rule."},
    "100.2": {"ruleNumber": "100.2", "ruleText": "Second rule.", "examples": ["Example: one."]},
    "100.3": {"ruleNumber": "100.3", "ruleText": "Third rule."},
}


@pytest.fixture
def run_tokenizer(qdrant_client, monkeypatch):
    """Returns a function that runs the Tokenizer on the given rules against the test Qdrant."""
    monkeypatch.setattr(tokenize, "get_qdrant_client", lambda: qdrant_client)

    def run(rules: dict, embedder: FakeOllamaEmbedder) -> None:
        monkeypatch.setattr(tokenize, "ollama_embedder", embedder)
        storage = MagicMock()
        storage.return_value.read.return_value = json.dumps(rules).encode()
        storage = typing.cast(type[StorageConnection], storage)
        tokenize.Tokenizer(
            file_path="cr.json",
            connection_cls=storage,
            parser=ComprehensiveRulesParser(),
            model_cls=Rule,
            text_field="rule_text",
        )()

    return run


def stored_rules(session: QdrantSession) -> dict[str, str]:
    """rule_number -> rule_text of everything in the collection."""
    return {r.rule_number: r.rule_text for batch in session.get_batches(Rule) for r in batch}


class TestTokenizer:
    def test_first_run_embeds_and_stores_every_rule(self, run_tokenizer, session, fake_embedder):
        run_tokenizer(RULES, fake_embedder)

        assert len(fake_embedder.embedded) == 3
        assert stored_rules(session) == {
            "100.1": "First rule.",
            "100.2": "Second rule.\n\nEXAMPLES\nExample: one.",
            "100.3": "Third rule.",
        }

    def test_rerun_with_same_file_embeds_nothing(self, run_tokenizer, session):
        run_tokenizer(RULES, FakeOllamaEmbedder())

        second = FakeOllamaEmbedder()
        run_tokenizer(RULES, second)

        assert second.embedded == []
        assert len(stored_rules(session)) == 3

    def test_changed_file_replaces_edited_deletes_removed_and_adds_new(self, run_tokenizer, session):
        run_tokenizer(RULES, FakeOllamaEmbedder())

        changed = {
            "100.1": RULES["100.1"],                                      
            "100.2": {"ruleNumber": "100.2", "ruleText": "Second rule, edited."},
            "100.4": {"ruleNumber": "100.4", "ruleText": "Fourth rule."},  
        }
        embedder = FakeOllamaEmbedder()
        run_tokenizer(changed, embedder)

        assert sorted(embedder.embedded) == ["Fourth rule.", "Second rule, edited."]
        assert stored_rules(session) == {
            "100.1": "First rule.",
            "100.2": "Second rule, edited.",
            "100.4": "Fourth rule.",
        }

    def test_rerun_after_a_crash_completes_without_re_embedding_stored_batches(
        self, run_tokenizer, session, monkeypatch
    ):
        monkeypatch.setattr(tokenize.Tokenizer, "BATCH_SIZE", 1)
        rules = {f"100.{i}": {"ruleNumber": f"100.{i}", "ruleText": f"Rule {i}."} for i in range(5)}

        crashing = FakeOllamaEmbedder(fail_after_calls=2)
        with pytest.raises(ConnectionError):
            run_tokenizer(rules, crashing)
        assert len(stored_rules(session)) == 2  # batches finished before the crash were kept

        resumed = FakeOllamaEmbedder()
        run_tokenizer(rules, resumed)

        assert len(resumed.embedded) == 3
        assert stored_rules(session) == {f"100.{i}": f"Rule {i}." for i in range(5)}

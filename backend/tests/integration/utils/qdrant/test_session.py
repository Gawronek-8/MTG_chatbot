import math
import uuid

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType

from src.qmodels import Rule
from src.utils.qdrant import QdrantSession
from tests.mocks.ollama import FakeOllamaEmbedder

_embedder = FakeOllamaEmbedder(size=768)


def make_rules(count: int, start: int = 0) -> list[Rule]:
    rules = []
    for i in range(start, start + count):
        text = f"Rule text {i}."
        rules.append(Rule(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, text)),
            rule_number=f"100.{i}",
            rule_text=text,
            vector=_embedder.embed_query(text),
        ))
    return rules


def normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


class TestSession:
    def test_put_and_get_round_trip(self, session: QdrantSession):
        [rule] = make_rules(1)
        session.put(rule, create_missing=True)

        [stored] = session.get(Rule)

        assert stored.id == rule.id
        assert (stored.rule_number, stored.rule_text) == (rule.rule_number, rule.rule_text)
        assert stored.vector == pytest.approx(normalized(rule.vector), abs=1e-6)

    def test_put_raises_when_collection_missing_and_writes_nothing(self, session: QdrantSession):
        with pytest.raises(RuntimeError, match="does not exist"):
            session.put(make_rules(3))

        assert not session.collection_exists(Rule)

    def test_put_creates_missing_collection_with_payload_index(self, session: QdrantSession, qdrant_client):
        session.put(make_rules(1), create_missing=True)

        schema = qdrant_client.get_collection(Rule.__collection__).payload_schema
        assert schema["rule_number"].data_type == PayloadSchemaType.KEYWORD
        assert "rule_text" not in schema

    def test_put_upserts_by_id(self, session: QdrantSession):
        [rule] = make_rules(1)
        session.put(rule, create_missing=True)

        rule.rule_text = "Updated."
        session.put(rule)

        stored = session.get(Rule)
        assert [r.rule_text for r in stored] == ["Updated."]

    def test_put_rejects_wrong_vector_size(self, session: QdrantSession):
        session.put(make_rules(1), create_missing=True)
        [rule] = make_rules(1, start=1)
        rule.vector = [0.5] * 10

        with pytest.raises(UnexpectedResponse):
            session.put(rule)

    def test_get_with_id_field_fetches_neither_payload_nor_vectors(self, session: QdrantSession):
        [rule] = make_rules(1)
        session.put(rule, create_missing=True)

        [stored] = session.get(Rule, fields=["id"])

        assert stored.id == rule.id
        assert stored.rule_number is None and stored.rule_text is None
        assert not any(stored.vector)

    def test_get_filters_by_payload(self, session: QdrantSession):
        session.put(make_rules(3), create_missing=True)

        stored = session.get(Rule, rule_number="100.1")

        assert [r.rule_number for r in stored] == ["100.1"]

    def test_get_rejects_unknown_fields(self, session: QdrantSession):
        with pytest.raises(ValueError, match="not defined"):
            session.get(Rule, fields=["score"])

    def test_get_batches_returns_every_point_exactly_once(self, session: QdrantSession):
        rules = make_rules(7)
        session.put(rules, create_missing=True)

        batches = list(session.get_batches(Rule, batch_size=3, fields=["id"]))

        assert [len(b) for b in batches] == [3, 3, 1]
        assert sorted(r.id for b in batches for r in b) == sorted(r.id for r in rules)

    def test_get_batches_survives_deleting_each_page_while_paging(self, session: QdrantSession):
        """The tokenizer deletes stale ids page by page while it is still paging; no point may be skipped."""
        rules = make_rules(7)
        session.put(rules, create_missing=True)

        seen = []
        for batch in session.get_batches(Rule, batch_size=3, fields=["id"]):
            ids = [r.id for r in batch]
            seen.extend(ids)
            session.refresh_collection(Rule, ids)

        assert sorted(seen) == sorted(r.id for r in rules)
        assert session.get(Rule) == []

    def test_refresh_collection_deletes_only_the_given_ids(self, session: QdrantSession):
        rules = make_rules(4)
        session.put(rules, create_missing=True)

        session.refresh_collection(Rule, [rules[0].id, rules[2].id], batch_size=1)

        assert sorted(r.id for r in session.get(Rule)) == sorted([rules[1].id, rules[3].id])

    def test_collection_exists(self, session: QdrantSession):
        assert not session.collection_exists(Rule)

        session.put(make_rules(1), create_missing=True)

        assert session.collection_exists(Rule)

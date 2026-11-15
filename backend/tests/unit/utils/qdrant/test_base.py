import pytest

from src.qmodels import Rule

RULE_ID = "3f2b8c1e-0000-5000-8000-000000000000"


def make_rule(**overrides) -> Rule:
    fields = {"id": RULE_ID, "rule_number": "100.1", "rule_text": "Text.", "vector": [0.5] * 768}
    return Rule(**(fields | overrides))


class TestQdrantBaseModel:
    def test_to_point_keeps_id_and_vectors_out_of_the_payload(self):
        point = make_rule().to_point()

        assert point.id == RULE_ID
        assert point.payload == {"rule_number": "100.1", "rule_text": "Text."}
        assert point.vector == {"vector": [0.5] * 768}

    def test_to_point_refuses_a_missing_id(self):
        with pytest.raises(ValueError, match="no id"):
            make_rule(id=None).to_point()

    def test_to_point_refuses_a_vector_that_was_never_set(self):
        """A point that was never embedded must not be stored, or it would silently pollute search results."""
        rule = Rule(id=RULE_ID, rule_number="100.1", rule_text="Text.")

        with pytest.raises(ValueError, match="vector 'vector'"):
            rule.to_point()

    def test_to_point_refuses_a_none_vector(self):
        with pytest.raises(ValueError, match="vector 'vector'"):
            make_rule(vector=None).to_point()

    def test_unknown_constructor_fields_are_rejected(self):
        """Catches parser/model key mismatches (e.g. the old "embedding" vs "vector")."""
        with pytest.raises(ValueError, match="embedding"):
            make_rule(embedding=[0.5] * 768)

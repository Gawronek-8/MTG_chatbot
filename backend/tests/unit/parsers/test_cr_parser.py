import json

import pytest

from src.config import settings
from src.parsers import ComprehensiveRulesParser
from src.utils.langchain_utils import calculate_expected_tokens

def parse(data: dict) -> list[dict]:
    """Runs the parser and flattens its per-rule lists into one list of entries."""
    raw = json.dumps(data).encode()
    return [entry for rule in ComprehensiveRulesParser().parse(raw) for entry in rule]


class TestCRParser:
    def test_id_and_text_are_stable(self):
        """
        Golden test: ids are derived from the chunk text, so any change to how the text is built
        (e.g. the EXAMPLES block) changes every id, and the next run deletes and re-embeds the whole
        collection. If this fails on purpose, update the expected values and expect a full re-embed.
        """
        [entry] = parse({
            "702.2b": {
                "ruleNumber": "702.2b",
                "ruleText": "Deathtouch is a static ability.",
                "examples": ["Example: A creature with deathtouch fights another creature."],
            }
        })

        assert entry["rule_text"] == (
            "Deathtouch is a static ability.\n\nEXAMPLES\n"
            "Example: A creature with deathtouch fights another creature."
        )
        assert entry["id"] == "c92669fb-1853-57a9-b998-bdaa8fa0a4b7"

    def test_rule_without_examples_is_one_chunk_of_its_text(self):
        [entry] = parse({"100.1": {"ruleNumber": "100.1", "ruleText": "These rules apply to any game."}})

        assert entry["rule_number"] == "100.1"
        assert entry["rule_text"] == "These rules apply to any game."
        assert "vector" not in entry

    def test_examples_are_listed_one_per_line(self):
        [entry] = parse({
            "100.2": {
                "ruleNumber": "100.2",
                "ruleText": "Rule text.",
                "examples": ["Example: first.", "Example: second."],
            }
        })

        assert entry["rule_text"] == "Rule text.\n\nEXAMPLES\nExample: first.\nExample: second."

    def test_each_rule_is_yielded_as_its_own_list(self):
        rules = list(ComprehensiveRulesParser().parse(json.dumps({
            "100.1": {"ruleNumber": "100.1", "ruleText": "A."},
            "100.2": {"ruleNumber": "100.2", "ruleText": "B."},
        }).encode()))

        assert [[entry["rule_number"] for entry in rule] for rule in rules] == [["100.1"], ["100.2"]]

    def test_same_text_under_different_rule_numbers_gets_different_ids(self):
        entries = parse({
            "100.1": {"ruleNumber": "100.1", "ruleText": "Same text."},
            "100.2": {"ruleNumber": "100.2", "ruleText": "Same text."},
        })

        assert entries[0]["id"] != entries[1]["id"]

    def test_long_rule_is_split_into_chunks_that_fit_the_token_limit(self):
        # Paragraphs of ~60% of the limit each: one fits, three together don't, so the rule is split at the blank lines.
        paragraph = "word " * int(settings.ollama_max_tokens * 0.6 * 4 / 5)
        rule_text = "\n\n".join([paragraph.strip()] * 3)
        assert calculate_expected_tokens(rule_text) >= settings.ollama_max_tokens

        entries = parse({"100.3": {"ruleNumber": "100.3", "ruleText": rule_text}})

        assert len(entries) > 1
        assert all(calculate_expected_tokens(e["rule_text"]) < settings.ollama_max_tokens for e in entries)
        assert all(e["rule_number"] == "100.3" for e in entries)

    def test_single_oversized_paragraph_falls_back_to_the_splitter(self):
        rule_text = "word " * (settings.ollama_max_tokens * 4 // 5 * 2)

        entries = parse({"100.4": {"ruleNumber": "100.4", "ruleText": rule_text}})

        assert len(entries) > 1
        assert all(calculate_expected_tokens(e["rule_text"]) < settings.ollama_max_tokens for e in entries)

    def test_split_chunks_get_distinct_ids(self):
        rule_text = "\n\n".join(f"paragraph {i} " + "word " * (settings.ollama_max_tokens // 2) for i in range(3))

        entries = parse({"100.5": {"ruleNumber": "100.5", "ruleText": rule_text}})

        assert len({e["id"] for e in entries}) == len(entries)

    @pytest.mark.parametrize("entry", [
        {"ruleNumber": "100.6"},
        {"ruleNumber": "100.6", "ruleText": None},
        {"ruleNumber": "100.6", "ruleText": "", "examples": []},
    ])
    def test_missing_text_does_not_crash(self, entry):
        entries = parse({"100.6": entry})

        assert all(isinstance(e["rule_text"], str) for e in entries)

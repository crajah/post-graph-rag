"""Edge-case tests for extractor utilities — no LLM needed."""


from post_graph_rag.extractor import (
    BARE_QUANTITY,
    DEFAULT_ENTITY_TYPES,
    MAX_ENTITY_NAME_CHARS,
    TENSE_PREFIXES,
    VAGUE_PREDICATES,
    Entity,
    ExtractionResult,
    GraphExtractor,
    Triple,
    _clean_date,
    date_sort_key,
    is_phrase_not_entity,
    is_pronominal,
    normalise_predicate,
)


class TestNormalisePredicate:
    def test_lowercase_and_underscore(self):
        assert normalise_predicate("Works At") == "works_at"

    def test_strip_tense_prefix(self):
        assert normalise_predicate("was_appointed_knight_of") == "appointed_knight_of"
        assert normalise_predicate("is_known_for") == "known_for"
        assert normalise_predicate("has_published") == "published"

    def test_preserves_short_after_strip(self):
        # Should not strip if remaining is too short
        assert normalise_predicate("was_at") == "was_at"

    def test_hyphens_to_underscores(self):
        assert normalise_predicate("works-at") == "works_at"

    def test_multiple_spaces(self):
        assert normalise_predicate("works  at   the") == "works_at_the"

    def test_empty_and_none(self):
        assert normalise_predicate("") == ""
        assert normalise_predicate(None) == ""

    def test_leading_trailing_whitespace(self):
        assert normalise_predicate("  works_at  ") == "works_at"

    def test_only_first_tense_prefix_stripped(self):
        assert normalise_predicate("was_has_done") == "has_done"

    def test_mixed_separators(self):
        assert normalise_predicate("was-employed by") == "employed_by"


class TestIsPronomonial:
    def test_pronouns(self):
        for pronoun in ["he", "she", "it", "they", "him", "her", "them"]:
            assert is_pronominal(pronoun) is True

    def test_pronominal_prefixes(self):
        assert is_pronominal("his father") is True
        assert is_pronominal("her company") is True
        assert is_pronominal("their work") is True

    def test_long_phrase_not_pronominal(self):
        assert is_pronominal("his very long extended phrase that goes on") is False

    def test_proper_names(self):
        assert is_pronominal("Charles Babbage") is False
        assert is_pronominal("Ada Lovelace") is False
        assert is_pronominal("Boeing") is False

    def test_empty_and_none(self):
        assert is_pronominal("") is True
        assert is_pronominal(None) is True


class TestIsPhraseNotEntity:
    def test_conjunction_entity(self):
        assert is_phrase_not_entity("Ada Lovelace and Charles Babbage") is True

    def test_proper_name_not_rejected(self):
        assert is_phrase_not_entity("Charles Babbage") is False

    def test_oversized_name(self):
        long_name = "A" * (MAX_ENTITY_NAME_CHARS + 1)
        assert is_phrase_not_entity(long_name) is True

    def test_bare_quantity_dollar(self):
        assert is_phrase_not_entity("$18.4 billion") is True

    def test_bare_quantity_percentage(self):
        assert is_phrase_not_entity("40%") is True

    def test_bare_quantity_number(self):
        assert is_phrase_not_entity("98") is True

    def test_name_with_number_kept(self):
        assert is_phrase_not_entity("737 MAX") is False
        assert is_phrase_not_entity("Boeing Company") is False
        assert is_phrase_not_entity("3M") is False

    def test_possessive_off_by_default(self):
        assert is_phrase_not_entity("Babbage's father") is False

    def test_possessive_on_when_requested(self):
        assert is_phrase_not_entity("Babbage's father", reject_possessive=True) is True

    def test_empty(self):
        assert is_phrase_not_entity("") is False

    def test_at_max_length(self):
        name = "A" * MAX_ENTITY_NAME_CHARS
        assert is_phrase_not_entity(name) is False


class TestCleanDate:
    def test_bare_year(self):
        assert _clean_date("1625") == "1625"

    def test_year_month(self):
        assert _clean_date("1625-06") == "1625-06"

    def test_full_date(self):
        assert _clean_date("1625-06-12") == "1625-06-12"

    def test_padding(self):
        assert _clean_date("5") == "0005"
        assert _clean_date("90-3") == "0090-03"

    def test_none(self):
        assert _clean_date(None) is None

    def test_empty(self):
        assert _clean_date("") is None

    def test_vague_rejected(self):
        assert _clean_date("later") is None
        assert _clean_date("in his youth") is None
        assert _clean_date("unknown") is None
        assert _clean_date("n/a") is None
        assert _clean_date("null") is None
        assert _clean_date("none") is None
        assert _clean_date("-") is None

    def test_negative_year(self):
        assert _clean_date("-500") == "-500"


class TestDateSortKey:
    def test_bare_year_padded(self):
        assert date_sort_key("1625") == "1625-01-01"

    def test_year_month_padded(self):
        assert date_sort_key("1625-06") == "1625-06-01"

    def test_full_date_unchanged(self):
        assert date_sort_key("1625-06-12") == "1625-06-12"

    def test_none_empty(self):
        assert date_sort_key(None) == ""
        assert date_sort_key("") == ""

    def test_ordering(self):
        dates = ["1700", "1625-06-12", "1625", "1625-06"]
        sorted_keys = sorted(dates, key=date_sort_key)
        assert sorted_keys[0] == "1625"
        assert sorted_keys[-1] == "1700"


class TestVaguePredicates:
    def test_all_lower(self):
        for p in VAGUE_PREDICATES:
            assert p == p.lower()

    def test_known_vague(self):
        assert "relates_to" in VAGUE_PREDICATES
        assert "associated_with" in VAGUE_PREDICATES
        assert "connected_to" in VAGUE_PREDICATES


class TestTensePrefixes:
    def test_all_end_with_underscore(self):
        for p in TENSE_PREFIXES:
            assert p.endswith("_")


class TestDefaultEntityTypes:
    def test_has_common_types(self):
        assert "Person" in DEFAULT_ENTITY_TYPES
        assert "Organization" in DEFAULT_ENTITY_TYPES
        assert "Concept" in DEFAULT_ENTITY_TYPES


class TestBareQuantityPattern:
    def test_dollar_amounts(self):
        assert BARE_QUANTITY.match("$1,326")
        assert BARE_QUANTITY.match("$18.4 billion")
        assert BARE_QUANTITY.match("$ 2.5 million")

    def test_percentages(self):
        assert BARE_QUANTITY.match("40%")
        assert BARE_QUANTITY.match("12.7 percent")

    def test_parenthesized(self):
        assert BARE_QUANTITY.match("(1,204)")

    def test_non_quantities_not_matched(self):
        assert not BARE_QUANTITY.match("717 aircraft")
        assert not BARE_QUANTITY.match("Boeing Company")
        assert not BARE_QUANTITY.match("737 MAX")


class TestExtractionResultModel:
    def test_defaults(self):
        r = ExtractionResult()
        assert r.entities == []
        assert r.triples == []

    def test_with_entities(self):
        e = Entity(name="Zeus", type="Person", description="King of gods")
        r = ExtractionResult(entities=[e])
        assert len(r.entities) == 1
        assert r.entities[0].name == "Zeus"


class TestTripleModel:
    def test_defaults(self):
        t = Triple(subject="A", predicate="knows", object="B")
        assert t.negated is False
        assert t.confidence == 1.0
        assert t.valid_from is None
        assert t.valid_to is None
        assert t.description is None

    def test_with_all_fields(self):
        t = Triple(
            subject="A", predicate="ruled", object="B",
            description="A ruled B", negated=True, confidence=0.8,
            valid_from="1625", valid_to="1649"
        )
        assert t.negated is True
        assert t.confidence == 0.8


class TestEntityModel:
    def test_defaults(self):
        e = Entity(name="Test", type="Concept", description="A thing")
        assert e.aliases == []

    def test_with_aliases(self):
        e = Entity(name="Charles Babbage", type="Person",
                   description="Mathematician", aliases=["Babbage"])
        assert "Babbage" in e.aliases


class TestGraphExtractorPrompt:
    def test_system_prompt_generated(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm)
        prompt = ext.system_prompt
        assert "ENTITIES" in prompt
        assert "TRIPLES" in prompt

    def test_custom_prompt_used(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm, system_prompt="Custom prompt here")
        assert ext.system_prompt == "Custom prompt here"

    def test_entity_types_in_prompt(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm, entity_types=["Animal", "Plant"])
        assert "'Animal'" in ext.system_prompt
        assert "'Plant'" in ext.system_prompt

    def test_vocabulary_in_prompt(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm, predicate_vocabulary=["works_at", "knows"])
        assert "'works_at'" in ext.system_prompt
        assert "'knows'" in ext.system_prompt

    def test_validity_disabled(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm, extract_validity=False)
        assert "Do not populate valid_from" in ext.system_prompt


class TestGraphExtractorContextBlock:
    def test_no_context(self):
        assert GraphExtractor._context_block(None) == ""

    def test_with_title(self):
        from post_graph_rag.models import DocumentContext
        ctx = DocumentContext(title="My Doc")
        block = GraphExtractor._context_block(ctx)
        assert "My Doc" in block

    def test_with_known_entities(self):
        from post_graph_rag.models import DocumentContext
        ctx = DocumentContext(known_entities=["Alice", "Bob"])
        block = GraphExtractor._context_block(ctx)
        assert "Alice" in block
        assert "Bob" in block

    def test_empty_context(self):
        from post_graph_rag.models import DocumentContext
        ctx = DocumentContext()
        assert GraphExtractor._context_block(ctx) == ""


class TestGraphExtractorCanonicalPredicate:
    def test_normalises(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm)
        assert ext._canonical_predicate("Was Employed By") == "employed_by"

    def test_snaps_to_vocabulary(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm, predicate_vocabulary=["works_at"])
        assert ext._canonical_predicate("works_at_company") == "works_at"

    def test_alias_applied(self):
        from conftest import FakeLLM, make_config
        config = make_config()
        llm = FakeLLM(config)
        ext = GraphExtractor(llm, predicate_aliases={"employed_at": "works_at"})
        assert ext._canonical_predicate("employed_at") == "works_at"


class TestGraphExtractorMerge:
    def test_merge_entities_deduped(self):
        base = ExtractionResult(
            entities=[Entity(name="Zeus", type="Person", description="King")],
            triples=[]
        )
        extra = ExtractionResult(
            entities=[Entity(name="Zeus", type="God", description="")],
            triples=[]
        )
        merged = GraphExtractor._merge(base, extra)
        assert len(merged.entities) == 1
        assert merged.entities[0].type == "Person"  # non-Concept preferred

    def test_merge_triples_deduped(self):
        t1 = Triple(subject="A", predicate="knows", object="B")
        t2 = Triple(subject="A", predicate="knows", object="B", valid_from="1625")
        base = ExtractionResult(entities=[], triples=[t1])
        extra = ExtractionResult(entities=[], triples=[t2])
        merged = GraphExtractor._merge(base, extra)
        assert len(merged.triples) == 1
        assert merged.triples[0].valid_from == "1625"

    def test_merge_aliases_accumulated(self):
        e1 = Entity(name="Charles Babbage", type="Person", description="Mathematician",
                     aliases=["Babbage"])
        e2 = Entity(name="Charles Babbage", type="Person", description="",
                     aliases=["C. Babbage"])
        base = ExtractionResult(entities=[e1], triples=[])
        extra = ExtractionResult(entities=[e2], triples=[])
        merged = GraphExtractor._merge(base, extra)
        aliases = merged.entities[0].aliases
        assert "Babbage" in aliases
        assert "C. Babbage" in aliases


class TestGraphExtractorParseJson:
    def test_valid_json(self):
        data = '{"entities": [{"name": "A", "type": "T", "description": "D"}], "triples": []}'
        result = GraphExtractor._parse_json_result(data)
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 1

    def test_json_with_fences(self):
        data = '```json\n{"entities": [], "triples": []}\n```'
        result = GraphExtractor._parse_json_result(data)
        assert isinstance(result, ExtractionResult)

    def test_empty(self):
        assert GraphExtractor._parse_json_result("") is None
        assert GraphExtractor._parse_json_result("   ") is None
        assert GraphExtractor._parse_json_result(None) is None

    def test_invalid_json(self):
        assert GraphExtractor._parse_json_result("not json") is None

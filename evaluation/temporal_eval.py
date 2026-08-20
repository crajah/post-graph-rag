#!/usr/bin/env python3
"""Evaluate temporal handling on a corpus whose facts change over time.

    python evaluation/fetch_corpus.py --series dumas --out evaluation/dumas
    python evaluation/temporal_eval.py --corpus evaluation/dumas --realm dumas_kb

Books are indexed in publication order, which is what supersession relies on: a
later book's account of a relationship replaces the earlier one. Files are named
with a numeric prefix so a plain sort gives chronological order.
"""
import argparse
import asyncio
import glob
import json
import logging
import os
import time

from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig, RAGError

# Alliances in this series genuinely reverse: former comrades end up on opposing
# sides. Declaring the incompatible pairs lets a later book close an earlier
# claim instead of both reading as currently true.
RELATIONSHIP_GROUPS = [
    # All the ways this text expresses "on the same side" versus "opposed",
    # collapsed by ALIASES onto these canonical forms.
    {"friend_of", "enemy_of", "rival_of", "ally_of", "opponent_of"},
    {"serves", "betrays", "opponent_of"},
    {"employed_by", "former_employer"},
]

VOCABULARY = [
    "friend_of", "enemy_of", "rival_of", "ally_of", "opponent_of",
    "serves", "betrays", "employed_by", "former_employer",
    "father_of", "child_of", "married_to", "sibling_of",
    "fights", "rescues", "meets", "travels_to", "located_in",
    "member_of", "commands", "loves", "protects",
]

# Chosen from what the model actually produced on this corpus, not guessed:
# `opposes` and `opponent_of` appeared on the same pair, and `befriends` is far
# commoner here than `friend_of`.
ALIASES = {
    "companion_of": "friend_of", "comrade_of": "friend_of", "befriends": "friend_of",
    "allied_with": "ally_of", "adversary_of": "enemy_of", "foe_of": "enemy_of",
    "opposes": "opponent_of", "opposed_by": "opponent_of", "fights": "opponent_of",
    "works_for": "employed_by", "in_service_of": "serves", "serves_under": "serves",
    "son_of": "child_of", "daughter_of": "child_of",
}

QUESTIONS = [
    "What is the relationship between d'Artagnan and Athos?",
    "Who are the enemies of d'Artagnan?",
    "How do the loyalties of the musketeers change over time?",
]

# --------------------------------------------------------------------------
# Financial filings. Annual reports are a natural temporal corpus: the same
# line items recur every year while their significance inverts. A programme
# that "generates record cash flow" in one filing "drives negative cash flow"
# in another, and the filings say so in prose rather than in a dated field.
FINANCE_VOCABULARY = [
    "generates_cash_flow", "consumes_cash", "increases_revenue", "reduces_revenue",
    "improves_margin", "erodes_margin", "incurs_charge", "recognises_gain",
    "delays", "accelerates", "grounds", "certifies", "suspends", "resumes",
    "caps_production", "increases_production", "delivers", "cancels_order",
    "invests_in", "acquires", "divests", "settles_with", "compensates",
    "defers_cost", "writes_off", "reports_loss", "reports_profit",
    "regulates", "investigates", "penalises", "part_of", "supplies",
]

# Predicates that cannot both hold between the same pair, so a later filing
# closes an earlier claim rather than sitting beside it. This is the whole
# point on financial data: "737 generates cash flow" and "737 consumes cash"
# are both true historically but not simultaneously.
FINANCE_GROUPS = [
    {"generates_cash_flow", "consumes_cash"},
    {"increases_revenue", "reduces_revenue"},
    {"improves_margin", "erodes_margin"},
    {"grounds", "certifies"},
    {"suspends", "resumes"},
    {"caps_production", "increases_production"},
    {"delays", "accelerates"},
    {"reports_profit", "reports_loss"},
]

FINANCE_ALIASES = {
    "generates_free_cash_flow": "generates_cash_flow",
    "generates_record_free_cash_flow": "generates_cash_flow",
    "produces_cash": "generates_cash_flow", "funds": "generates_cash_flow",
    "drives_negative_fcf": "consumes_cash", "burns_cash": "consumes_cash",
    "drains_cash": "consumes_cash", "reduces_cash": "consumes_cash",
    "incurs_supply_chain_charges": "incurs_charge", "takes_charge": "incurs_charge",
    "records_charge": "incurs_charge", "charges": "incurs_charge",
    "maximizes_operating_margins": "improves_margin", "improves_margins": "improves_margin",
    "grounded": "grounds", "grounding_of": "grounds",
    "recertifies": "certifies", "approves": "certifies",
    "deferred_production_costs": "defers_cost", "defers_production_cost": "defers_cost",
    "halts_production": "suspends", "pauses": "suspends",
    "limits_production": "caps_production", "restricts_production": "caps_production",
    # Derived from the predicates the model actually emitted on the 10-K corpus,
    # not guessed: on the Dumas corpus guessed aliases fired zero supersessions.
    "generates_substantial_cash_flow": "generates_cash_flow",
    "increases_cash": "generates_cash_flow",
    "increases_cash_from_operations": "generates_cash_flow",
    "reports_cash_from_operations": "generates_cash_flow",
    "receives_cash_from": "generates_cash_flow",
    "recorded_charges_at": "incurs_charge", "recorded_charges_of": "incurs_charge",
    "recorded_charges_for": "incurs_charge", "recorded_charges_on": "incurs_charge",
    "recognised_charges": "incurs_charge", "incurred_charges": "incurs_charge",
    "incurred_charges_for": "incurs_charge",
    "increases_risk_of_delay": "delays",
    "lower_margins_than": "erodes_margin",
}

FINANCE_QUESTIONS = [
    "How did deferred production costs change from a minor line item to a driver of cash burn?",
    "What caused Boeing's cash flow to turn negative?",
    "How did the relationship between the 737 programme and Boeing's cash flow change over time?",
    "What role did the FAA play across these filings?",
]

PRESETS = {
    "fiction": (VOCABULARY, ALIASES, RELATIONSHIP_GROUPS, QUESTIONS, ["1625", "1648", "1660"]),
    "finance": (FINANCE_VOCABULARY, FINANCE_ALIASES, FINANCE_GROUPS, FINANCE_QUESTIONS,
                ["2006", "2012", "2018", "2020", "2024"]),
}


async def run(args):
    vocabulary, aliases, groups, questions, default_as_of = PRESETS[args.preset]
    as_of_dates = args.as_of if args.as_of is not None else default_as_of
    config = RAGConfig(
        model=args.model,
        max_retries=args.max_retries,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        realm=args.realm,
        schema_per_realm=True,
        gleaning_passes=args.gleaning_passes,
        chunk_chars=args.chunk_chars,
        max_concurrent_chunks=args.max_concurrent_chunks,
        predicate_vocabulary=vocabulary,
        predicate_aliases=aliases,
        exclusive_predicate_groups=groups,
        extract_validity=True,
    )
    rag = GraphRAG(config)
    await rag.store.connect()
    if args.reset:
        await rag.store.client._execute(f'DROP SCHEMA IF EXISTS "{args.realm}" CASCADE;')
    await rag.initialize()

    paths = sorted(glob.glob(os.path.join(args.corpus, "*.txt")))
    if not paths:
        raise SystemExit(f"No .txt files in {args.corpus}. Run fetch_corpus.py --series dumas first.")

    print("=" * 74)
    print(f"TEMPORAL INDEXING (realm={args.realm}, model={args.model})")
    print("=" * 74)

    t_all = time.time()
    for path in paths:
        title = os.path.basename(path)[:-4]
        text = open(path).read()
        # Sample evenly across the whole novel. Taking the first N chunks only
        # reads the opening, where relationships are being established rather
        # than reversed — which is the opposite of what this corpus is for.
        all_chunks = rag.chunker(text)
        if args.max_chunks and len(all_chunks) > args.max_chunks:
            step = len(all_chunks) / args.max_chunks
            chunks = [all_chunks[int(i * step)] for i in range(args.max_chunks)]
        else:
            chunks = all_chunks
        body = "\n".join(chunks)
        print(f"\n--- {title}: {len(body):,} chars, {len(chunks)} chunks ---", flush=True)
        t0 = time.time()
        try:
            results = await rag.index_text(body, metadata=DocumentMetadata(
                source=f"gutenberg://{title}", document=title,
                category="fiction", collection="dumas",
            ))
            superseded = sum(r.get("relations_superseded", 0) for r in results)
            print(f"    {len(results)} chunks | "
                  f"{sum(r['entities_extracted'] for r in results)} entities | "
                  f"{sum(r['triples_extracted'] for r in results)} triples | "
                  f"{superseded} relations superseded | {time.time()-t0:.0f}s", flush=True)
        except RAGError as e:
            print(f"    FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)

    print(f"\n=== indexed in {(time.time()-t_all)/60:.1f} min ===")

    S = f'"{args.realm}"'
    q = rag.store.client._fetch

    print("\n" + "=" * 74)
    print("TEMPORAL STATE OF THE GRAPH")
    print("=" * 74)
    stats = (await q(
        f"SELECT (SELECT count(*) FROM {S}.entities) e,"
        f" (SELECT count(*) FROM {S}.relations) r,"
        f" (SELECT count(*) FROM {S}.relations WHERE payload->>'superseded_by' IS NOT NULL) sup,"
        f" (SELECT count(*) FROM {S}.relations WHERE payload->>'valid_from' IS NOT NULL"
        f"    OR payload->>'valid_to' IS NOT NULL) dated,"
        f" (SELECT count(*) FROM {S}.entities WHERE payload->>'dormant_since' IS NOT NULL) dormant"))[0]
    print(f"   entities={stats['e']}  relations={stats['r']}  "
          f"superseded={stats['sup']}  with stated validity={stats['dated']}  dormant={stats['dormant']}")
    print(f"   -> {stats['r'] - stats['dated']} relations carry no stated period and are "
          f"treated as always valid")

    rows = await q(
        f"SELECT f.payload->>'name' a, t.payload->>'name' b, r.relation_type k,"
        f"       r.payload->>'superseded_by' s"
        f" FROM {S}.relations r"
        f" JOIN {S}.entities f ON f.id = r.from_id JOIN {S}.entities t ON t.id = r.to_id"
        f" WHERE r.payload->>'superseded_by' IS NOT NULL ORDER BY r.id LIMIT 12")
    print(f"\n   Relationships a later book replaced ({len(rows)} shown):")
    for r in rows:
        print(f"     ({r['a']}) --[{r['k']}]--> ({r['b']})   superseded by edge {r['s']}")
    if not rows:
        print("     none — no incompatible pair was asserted twice")

    rows = await q(
        f"SELECT f.payload->>'name' a, t.payload->>'name' b, r.relation_type k,"
        f"       r.payload->>'valid_from' vf, r.payload->>'valid_to' vt"
        f" FROM {S}.relations r"
        f" JOIN {S}.entities f ON f.id = r.from_id JOIN {S}.entities t ON t.id = r.to_id"
        f" WHERE r.payload->>'valid_from' IS NOT NULL OR r.payload->>'valid_to' IS NOT NULL"
        f" ORDER BY r.id LIMIT 12")
    print(f"\n   Relations with a period the text actually stated ({len(rows)} shown):")
    for r in rows:
        print(f"     ({r['a']}) --[{r['k']}]--> ({r['b']})   {r['vf'] or '?'} .. {r['vt'] or 'open'}")
    if not rows:
        print("     none — the prose states few explicit dates, which is the expected case")

    print("\n" + "=" * 74)
    print("AS-OF RETRIEVAL")
    print("=" * 74)
    for as_of in as_of_dates:
        res = await rag.query_data(questions[0], param=QueryParam(
            mode="mix", top_k=6, as_of=as_of))
        rels = res["data"]["relationships"]
        print(f"\n   as_of={as_of}: {len(rels)} relations")
        for r in rels[:5]:
            period = f"{r.get('valid_from') or '?'}..{r.get('valid_to') or 'open'}"
            print(f"     ({r['src_id']}) --[{r['relation_type']}]--> ({r['tgt_id']})  [{period}]")

    print("\n" + "=" * 74)
    print("CURRENT vs FULL HISTORY")
    print("=" * 74)
    for question in questions:
        cur = await rag.query_data(question, param=QueryParam(mode="mix", top_k=6))
        hist = await rag.query_data(question, param=QueryParam(
            mode="mix", top_k=6, include_superseded=True))
        print(f"\n   Q: {question}")
        print(f"      current only : {len(cur['data']['relationships'])} relations")
        print(f"      with history : {len(hist['data']['relationships'])} relations")
        for r in cur["data"]["relationships"][:4]:
            print(f"        now: ({r['src_id']}) --[{r['relation_type']}]--> ({r['tgt_id']})")

    if args.synthesise:
        print("\n" + "=" * 74)
        print("SYNTHESISED ANSWERS")
        print("=" * 74)
        for question in questions:
            out = await rag.query(question, param=QueryParam(mode="mix", top_k=6))
            print(f"\n   Q: {question}\n   {out['answer'][:600]}")

    if args.stats_out:
        json.dump(dict(stats), open(args.stats_out, "w"), indent=2, default=str)
    await rag.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="evaluation/dumas")
    ap.add_argument("--realm", default=os.getenv("RAG_EVAL_REALM", "dumas_kb"))
    ap.add_argument("--model", default=os.getenv("RAG_MODEL", "google/gemma-4-26b-a4b-it-maas"))
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("RAG_MAX_RETRIES", "8")))
    ap.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001"))
    ap.add_argument("--embedding-dim", type=int, default=int(os.getenv("RAG_EMBEDDING_DIM", "1536")))
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--max-chunks", type=int, default=15, help="Per book; keeps runs bounded")
    ap.add_argument("--max-concurrent-chunks", type=int, default=6)
    ap.add_argument("--gleaning-passes", type=int, default=1)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fiction",
                    help="Vocabulary, exclusivity groups and questions for the corpus type")
    ap.add_argument("--as-of", nargs="*", default=None)
    ap.add_argument("--synthesise", action="store_true")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--no-reset", dest="reset", action="store_false")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

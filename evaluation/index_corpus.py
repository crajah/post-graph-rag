#!/usr/bin/env python3
"""Index a multi-document corpus and report indexing throughput.

    python evaluation/index_corpus.py
    python evaluation/index_corpus.py --max-chunks 5 --model DeepSeek-V3.2 \
        --fallback-models Meta-Llama-3.3-70B-Instruct MiniMax-M2.7

Every knob is a flag or environment variable; nothing about the corpus or the
model line-up is baked in.
"""
import argparse
import asyncio
import glob
import json
import logging
import os
import time

from post_graph_rag import DocumentContext, DocumentMetadata, GraphRAG, RAGConfig, RAGError

# A compact, domain-agnostic predicate set plus the synonyms that collapse onto
# it. The vocabulary steers the model and snaps morphological variants; the alias
# map is what merges genuinely different wordings for the same relation.
# Inverse relations are given their own canonical predicate rather than being
# mapped onto their converse, which would silently reverse their direction.
BIOGRAPHY_VOCABULARY = [
    "created", "designed", "built", "wrote", "published", "invented", "proposed",
    "worked_with", "corresponded_with", "met", "influenced", "inspired_by",
    "studied", "studied_under", "taught", "educated_at",
    "member_of", "founded", "employed_by", "awarded",
    "located_in", "born_in", "died_in",
    "child_of", "parent_of", "married_to", "sibling_of",
    "part_of", "has_component", "uses", "used_for", "can_calculate",
    "succeeded", "based_on", "translated",
]

BIOGRAPHY_ALIASES = {
    "collaborated_with": "worked_with", "contact_of": "worked_with",
    "worked_on": "worked_with", "acquaintances": "met", "acquainted_with": "met",
    "became_friends_with": "met", "friend_of": "met",
    "developed": "created", "made": "created", "sought_to_build": "designed",
    "construct": "built", "constructed": "built",
    "authored": "wrote", "published_translation_with": "translated",
    "translated_and_annotated": "translated",
    "educated_by": "studied_under", "taught_by": "studied_under",
    "student_of": "studied_under", "attended": "educated_at",
    "appointed_knight_of": "awarded", "conferred": "awarded",
    "son_of": "child_of", "daughter_of": "child_of",
    "father_of": "parent_of", "mother_of": "parent_of",
    "brother_of": "sibling_of", "sister_of": "sibling_of",
    "component": "has_component", "features": "has_component",
    "incorporated": "has_component", "contains": "has_component",
    "anticipated": "influenced", "inspired": "influenced",
    "read_works_of": "influenced_by", "influenced_by": "inspired_by",
}

VOCABULARY_PRESETS = {
    "biography": (BIOGRAPHY_VOCABULARY, BIOGRAPHY_ALIASES),
}


async def run(args):
    config = RAGConfig(
        model=args.model,
        fallback_models=args.fallback_models,
        max_retries=args.max_retries,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        realm=args.realm,
        schema_per_realm=True,
        embed_relations=args.embed_relations,
        gleaning_passes=args.gleaning_passes,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap,
        predicate_vocabulary=args.predicate_vocabulary,
        predicate_aliases=args.predicate_aliases,
    )
    rag = GraphRAG(config)
    await rag.store.connect()
    if args.reset:
        await rag.store.client._execute(f'DROP SCHEMA IF EXISTS "{args.realm}" CASCADE;')
    await rag.initialize()

    stats, t_all = [], time.time()
    paths = sorted(glob.glob(os.path.join(args.corpus, "*.txt")))
    if not paths:
        raise SystemExit(f"No .txt files in {args.corpus}. Run fetch_corpus.py first.")

    for path in paths:
        title = os.path.basename(path)[:-4].replace("_", " ")
        source = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        text = open(path).read()
        chunks = rag.chunker(text)[: args.max_chunks]
        print(f"\n=== {title}: {len(text):,} chars -> {len(chunks)} chunks ===", flush=True)

        # Thread context through the document so references that only resolve
        # against earlier text attach to the right entity.
        context = DocumentContext(title=title, source=source)

        for i, ch in enumerate(chunks, 1):
            t0 = time.time()
            try:
                res = await rag.index_document(ch, metadata=DocumentMetadata(
                    source=source,
                    category=args.category,
                    collection=args.collection,
                    document=f"{title}.txt",
                    paragraph=i,
                ), context=context)
                dt = time.time() - t0
                stats.append({"doc": title, "chunk": i, "secs": dt,
                              "entities": res["entities_extracted"],
                              "triples": res["triples_extracted"]})
                print(f"  [{i:2}/{len(chunks)}] {res['entities_extracted']:2}e "
                      f"{res['triples_extracted']:2}t  {dt:5.1f}s", flush=True)
                for name in res["entities"]:
                    if name not in context.known_entities:
                        context.known_entities.append(name)
                del context.known_entities[: max(0, len(context.known_entities) - config.context_entity_limit)]
            except RAGError as e:
                print(f"  [{i:2}/{len(chunks)}] FAILED {type(e).__name__}: {str(e)[:110]}", flush=True)
                stats.append({"doc": title, "chunk": i, "error": type(e).__name__})

    ok = [s for s in stats if "secs" in s]
    elapsed = time.time() - t_all
    print(f"\n=== INDEXED {len(ok)}/{len(stats)} chunks in {elapsed/60:.1f} min ===")
    if ok:
        secs = [s["secs"] for s in ok]
        print(f"    mean {sum(secs)/len(secs):.1f}s/chunk (min {min(secs):.1f}, max {max(secs):.1f}) | "
              f"{sum(s['entities'] for s in ok)} entities, {sum(s['triples'] for s in ok)} triples")

    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)
        print(f"    stats -> {args.stats_out}")

    await rag.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="evaluation/corpus")
    ap.add_argument("--realm", default=os.getenv("RAG_EVAL_REALM", "wiki_kb"))
    ap.add_argument("--model", default=os.getenv("RAG_MODEL", "DeepSeek-V3.2"))
    ap.add_argument("--fallback-models", nargs="*", default=[
        m for m in os.getenv("RAG_FALLBACK_MODELS", "").split(",") if m])
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("RAG_MAX_RETRIES", "5")))
    ap.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    ap.add_argument("--embedding-dim", type=int, default=int(os.getenv("RAG_EMBEDDING_DIM", "1536")))
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--gleaning-passes", type=int, default=int(os.getenv("RAG_GLEANING_PASSES", "1")))
    ap.add_argument("--predicate-vocabulary", nargs="*", default=[
        p for p in os.getenv("RAG_PREDICATE_VOCABULARY", "").split(",") if p])
    ap.add_argument("--predicate-aliases", default=None,
                    help="JSON object or path to a JSON file mapping synonym -> canonical predicate")
    ap.add_argument("--vocabulary-preset", choices=sorted(VOCABULARY_PRESETS),
                    help="Use a bundled vocabulary + alias map")
    ap.add_argument("--max-chunks", type=int, default=10, help="Per document; keeps eval runs bounded")
    ap.add_argument("--category", default="history_of_computing")
    ap.add_argument("--collection", default="eval_corpus")
    ap.add_argument("--embed-relations", action="store_true")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--no-reset", dest="reset", action="store_false", help="Append to the existing realm")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.vocabulary_preset:
        vocab, aliases = VOCABULARY_PRESETS[args.vocabulary_preset]
        args.predicate_vocabulary = args.predicate_vocabulary or list(vocab)
        args.predicate_aliases = dict(aliases)
    elif args.predicate_aliases:
        raw = args.predicate_aliases
        if os.path.exists(raw):
            raw = open(raw).read()
        args.predicate_aliases = json.loads(raw)
    else:
        args.predicate_aliases = {}

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

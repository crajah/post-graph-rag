"""Index the shared evaluation corpus with LightRAG, for comparison with post-graph-rag.

Same corpus, same LLM, same embedding model, same chunk sizing and gleaning depth
where the two libraries expose an equivalent knob.

LightRAG is not a dependency of this project. Install it separately, ideally in
its own virtualenv:

    pip install lightrag-hku
    python evaluation/fetch_corpus.py
    RAG_MODEL=MiniMax-M2.7 python evaluation/compare_lightrag.py

Then index the same corpus with post-graph-rag and diff the two:

    python evaluation/index_corpus.py --realm lr_compare --model MiniMax-M2.7
    python evaluation/compare_realms.py lr_compare
"""
import asyncio
import glob
import json
import os
import shutil
import time

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

BASE = os.getenv("OPENAI_API_BASE", "http://localhost:4000/v1")
KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL = os.getenv("RAG_MODEL", "MiniMax-M2.7")
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.getenv("CORPUS", os.path.join(HERE, "corpus"))
WORKDIR = os.getenv("LIGHTRAG_WORKDIR", os.path.join(HERE, "lightrag_work"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "20000"))  # matches 10 chunks x 2000 chars

QUESTIONS = [
    "What was the relationship between Ada Lovelace and Charles Babbage?",
    "How does the Analytical Engine differ from the Difference Engine?",
    "What are the main themes across these documents?",
]


# The router exhausts one provider's credits mid-run. post-graph-rag has retry
# and model failover built in; LightRAG does not, so the equivalent is added here
# to keep the comparison about extraction quality rather than resilience.
FALLBACKS = [m for m in os.getenv(
    "RAG_FALLBACK_MODELS", "gemma-4-31B-it,DeepSeek-V3.2,gpt-oss-120b,DeepSeek-V3.1"
).split(",") if m]
MAX_RETRIES = 5
RETRYABLE = ("402", "429", "500", "502", "503", "504", "run out of credits",
             "rate limit", "overloaded", "timeout")


def _retryable(exc) -> bool:
    text = str(exc).lower()
    return any(m.lower() in text for m in RETRYABLE)


async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    kwargs.pop("hashing_kv", None)
    last = None
    for model in [MODEL, *FALLBACKS]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await openai_complete_if_cache(
                    model, prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    base_url=BASE, api_key=KEY, **kwargs,
                )
            except Exception as e:
                last = e
                if not _retryable(e):
                    raise
                await asyncio.sleep(2.0 * attempt)
    raise RuntimeError(f"All models exhausted: {last}")


async def embed_func(texts):
    client = AsyncOpenAI(base_url=BASE, api_key=KEY)
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.embeddings.create(model=EMB_MODEL, input=texts)
            return np.array([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
        except Exception as e:
            last = e
            if not _retryable(e):
                raise
            await asyncio.sleep(2.0 * attempt)
    raise RuntimeError(f"Embedding failed: {last}")


async def main():
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR)

    rag = LightRAG(
        working_dir=WORKDIR,
        llm_model_func=llm_func,
        llm_model_name=MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=EMB_DIM, func=embed_func),
        chunk_token_size=500,          # ~2000 chars, matching post-graph-rag
        chunk_overlap_token_size=50,   # ~200 chars
        entity_extract_max_gleaning=1, # matching post-graph-rag's gleaning_passes=1
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    docs, names = [], []
    for path in sorted(glob.glob(f"{CORPUS}/*.txt")):
        docs.append(open(path).read()[:MAX_CHARS])
        names.append(os.path.basename(path))
    print(f"Indexing {len(docs)} documents ({sum(len(d) for d in docs):,} chars) with {MODEL}", flush=True)

    t0 = time.time()
    await rag.ainsert(docs, file_paths=names)
    elapsed = time.time() - t0
    print(f"\n=== LIGHTRAG INDEXED in {elapsed/60:.1f} min ===", flush=True)

    stats = {"index_seconds": elapsed, "documents": len(docs),
             "chars": sum(len(d) for d in docs), "model": MODEL}

    # Read the resulting graph straight off disk.
    gpath = os.path.join(WORKDIR, "graph_chunk_entity_relation.graphml")
    if os.path.exists(gpath):
        import xml.etree.ElementTree as ET
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        root = ET.parse(gpath).getroot()
        graph = root.find("g:graph", ns)
        nodes = graph.findall("g:node", ns)
        edges = graph.findall("g:edge", ns)
        keys = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", ns)}

        def attrs(el):
            return {keys.get(d.get("key"), d.get("key")): (d.text or "") for d in el.findall("g:data", ns)}

        etypes, descs = {}, 0
        for n in nodes:
            a = attrs(n)
            t = a.get("entity_type", "UNKNOWN").strip('"')
            etypes[t] = etypes.get(t, 0) + 1
            if a.get("description"):
                descs += 1
        keywords = {}
        for e in edges:
            a = attrs(e)
            for kw in (a.get("keywords") or "").split(","):
                kw = kw.strip().strip('"')
                if kw:
                    keywords[kw] = keywords.get(kw, 0) + 1

        stats.update(entities=len(nodes), relations=len(edges),
                     entity_types=len(etypes), entities_with_description=descs,
                     distinct_edge_keywords=len(keywords),
                     top_keywords=sorted(keywords.items(), key=lambda kv: -kv[1])[:12],
                     top_types=sorted(etypes.items(), key=lambda kv: -kv[1])[:12])
        print(f"    {len(nodes)} entities | {len(edges)} relations | "
              f"{len(etypes)} entity types | {len(keywords)} distinct edge keywords")

    print("\n=== RETRIEVAL ===", flush=True)
    answers = {}
    for q in QUESTIONS:
        for mode in ("mix", "global"):
            try:
                t = time.time()
                res = await rag.aquery(q, param=QueryParam(mode=mode, top_k=5))
                answers[f"{mode}: {q}"] = {"secs": round(time.time() - t, 1),
                                           "answer": res[:900] if isinstance(res, str) else str(res)[:900]}
                print(f"\n[{mode}] {q}\n{str(res)[:500]}\n", flush=True)
            except Exception as e:
                answers[f"{mode}: {q}"] = {"error": str(e)[:300]}
                print(f"\n[{mode}] {q}\n  ERROR: {str(e)[:300]}\n", flush=True)

    stats["answers"] = answers
    json.dump(stats, open(os.path.join(HERE, "lightrag_stats.json"), "w"), indent=2, default=str)
    print(f"\nstats -> {os.path.join(HERE, 'lightrag_stats.json')}")
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())

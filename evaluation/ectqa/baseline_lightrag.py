"""Run LightRAG on the same ECT-QA slice post-graph-rag is measured on.

The comparison in the README cited LightRAG's published number against a
different corpus slice. This removes that caveat: same six companies, same
sixteen quarters each, same 78 questions, same answering model, same
embedding model, and the answers are scored afterwards by the same
element-wise judge.

Isolation is matched too, which matters more than it sounds. post-graph-rag
scopes a company question to that company's space, so a fair baseline gets
the same benefit: one LightRAG working directory per company, plus one
holding all six for the cross-company questions. Indexing everything into a
single store and then asking about one company would be measuring our
multi-tenancy rather than their retrieval.

LightRAG is not a dependency of this project; install it in its own
environment (pip install lightrag-hku) and run this with that interpreter.
"""
import argparse
import asyncio
import glob
import json
import os
import pathlib
import time

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

BASE = os.getenv("OPENAI_API_BASE", "http://localhost:4000/v1")
KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL = os.getenv("RAG_MODEL", "gemini-3.6-flash")
EMB_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
EMB_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1536"))
FALLBACKS = [m for m in os.getenv("RAG_FALLBACK_MODELS", "gemini-3.5-flash-lite,gpt-oss-120b").split(",") if m]
MAX_RETRIES = int(os.getenv("LR_MAX_RETRIES", "10"))
RETRYABLE = ("402", "429", "500", "502", "503", "504", "run out of credits",
             "rate limit", "overloaded", "timeout")

# The same instruction post-graph-rag is given, so the two systems are asked
# for the same thing in the same words.
ANSWER_INSTRUCTION = None    # filled from run.py at import time


def _retryable(exc):
    text = str(exc).lower()
    return any(m.lower() in text for m in RETRYABLE)


async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    kwargs.pop("hashing_kv", None)
    last = None
    for model in [MODEL, *FALLBACKS]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await openai_complete_if_cache(
                    model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    base_url=BASE, api_key=KEY, **kwargs)
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
            resp = await client.embeddings.create(model=EMB_MODEL, input=texts,
                                                  encoding_format="float")
            return np.array([d.embedding for d in
                             sorted(resp.data, key=lambda x: x.index)])
        except Exception as e:
            last = e
            if not _retryable(e):
                raise
            await asyncio.sleep(2.0 * attempt)
    raise RuntimeError(f"Embedding failed: {last}")


def transcripts_for(data_dir, code):
    """Every quarter of one company, in chronological order."""
    out = []
    for p in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.loads(pathlib.Path(p).read_text())
        if d.get("stock_code") != code:
            continue
        out.append((f"{d['year']}-q{d['quarter']}",
                    d.get("cleaned_content") or d.get("raw_content") or ""))
    out.sort(key=lambda x: x[0])
    return out


async def build(workdir, docs, names):
    if os.path.exists(workdir):
        import shutil
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)
    rag = LightRAG(
        working_dir=workdir,
        llm_model_func=llm_func, llm_model_name=MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=EMB_DIM, func=embed_func),
        chunk_token_size=500, chunk_overlap_token_size=50,
        entity_extract_max_gleaning=1)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.ainsert(docs, file_paths=names)
    return rag


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="a post-graph-rag results file; its questions are reused verbatim")
    ap.add_argument("--data", default=str(pathlib.Path(__file__).parent / "data"))
    ap.add_argument("--workroot", required=True)
    ap.add_argument("--mode", default="mix")
    ap.add_argument("--top-k", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = json.loads(pathlib.Path(args.results).read_text())
    rows = src["results"]
    spaces = src["spaces"]                       # {"WDC": "wdc", ...}
    codes = list(spaces)
    print(f"{len(rows)} questions | {len(codes)} companies | model {MODEL}", flush=True)

    # one store per company, plus one holding everything for cross-company
    stores = {}
    for code in codes:
        quarters = transcripts_for(args.data, code)
        docs = [text for _q, text in quarters]
        names = [f"{code}-{q}" for q, _t in quarters]
        t0 = time.time()
        stores[code] = await build(os.path.join(args.workroot, spaces[code]), docs, names)
        print(f"  indexed {code}: {len(docs)} quarters "
              f"({sum(len(d) for d in docs):,} chars) in {time.time()-t0:.0f}s", flush=True)

    all_docs, all_names = [], []
    for code in codes:
        for q, text in transcripts_for(args.data, code):
            all_docs.append(text); all_names.append(f"{code}-{q}")
    t0 = time.time()
    stores["__ALL__"] = await build(os.path.join(args.workroot, "all"), all_docs, all_names)
    print(f"  indexed ALL: {len(all_docs)} quarters in {time.time()-t0:.0f}s", flush=True)

    by_space = {v: k for k, v in spaces.items()}
    results, failed = [], []
    for i, r in enumerate(rows, 1):
        space = r["space"]
        code = by_space.get(space)
        store = stores.get(code) if code else stores["__ALL__"]
        try:
            answer = await store.aquery(
                f"{ANSWER_INSTRUCTION}\n\nQuestion: {r['question']}",
                param=QueryParam(mode=args.mode, top_k=args.top_k))
            answer = (answer or "").strip()
        except Exception as e:
            failed.append({"question": r["question"][:80],
                           "reason": f"{type(e).__name__}: {e}"})
            print(f"  [{i}/{len(rows)}] FAILED {type(e).__name__}", flush=True)
            continue
        results.append({"space": space, "companies": r["companies"], "type": r["type"],
                        "question": r["question"], "gold": r["gold"],
                        "answer": answer[:800]})
        print(f"  [{i}/{len(rows)}] {r['type']:<20} {len(answer):>5} chars", flush=True)

    pathlib.Path(args.out).write_text(json.dumps({
        "benchmark": "ECT-QA", "system": "LightRAG", "lightrag_mode": args.mode,
        "model": MODEL, "embedding_model": EMB_MODEL, "top_k": args.top_k,
        "source_questions": args.results, "spaces": spaces,
        "failed": failed, "reportable": not failed,
        "results": results}, indent=2))
    print(f"\n  {len(results)} answered, {len(failed)} failed -> {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    # reuse the exact instruction post-graph-rag is given
    import re
    src = (pathlib.Path(__file__).resolve().parent / "run.py").read_text()
    m = re.search(r"ANSWER_INSTRUCTION = \((.*?)\n\)\n", src, re.S)
    ANSWER_INSTRUCTION = eval("(" + m.group(1) + ")")
    asyncio.run(main())

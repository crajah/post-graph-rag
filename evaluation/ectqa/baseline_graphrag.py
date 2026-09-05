"""Run Microsoft GraphRAG on the same ECT-QA slice post-graph-rag is measured on.

Companion to baseline_lightrag.py, with the same fairness rules: one index per
company plus one holding all six, so a company question is answered from that
company's index rather than from a store where six companies' quarters are
mixed. post-graph-rag gets that isolation from spaces; a baseline compared
against it should get the equivalent, or the comparison measures our
multi-tenancy rather than their retrieval.

Chunking is set to 500 tokens with 50 overlap to match our ~2000-character
chunks, gleaning to 1, and both models to whatever the router serves us --
the same settings the LightRAG baseline uses.

GraphRAG is not a dependency of this project; install it in its own
environment (pip install graphrag, Python <3.14) and run this with that
interpreter.
"""
import argparse
import asyncio
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL = os.getenv("RAG_MODEL", "gemini-3.6-flash")
EMB_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
BASE = os.getenv("OPENAI_API_BASE", "http://localhost:4000/v1")
KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

SETTINGS = """\
completion_models:
  default_completion_model:
    model_provider: openai
    model: {model}
    auth_method: api_key
    api_key: {key}
    api_base: {base}
    concurrent_requests: 4
    retry:
      type: exponential_backoff
      max_retries: 10

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: {emb}
    auth_method: api_key
    api_key: {key}
    api_base: {base}
    concurrent_requests: 4
    retry:
      type: exponential_backoff
      max_retries: 10

input:
  type: text

chunking:
  type: tokens
  size: 500
  overlap: 50
  encoding_model: o200k_base

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output/lancedb

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 1

summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

community_reports:
  completion_model_id: default_completion_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"

local_search:
  chat_model_id: default_completion_model
  embedding_model_id: default_embedding_model

global_search:
  chat_model_id: default_completion_model
"""


def transcripts_for(data_dir, code):
    import glob
    out = []
    for p in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.loads(pathlib.Path(p).read_text())
        if d.get("stock_code") != code:
            continue
        out.append((f"{d['year']}-q{d['quarter']}",
                    d.get("cleaned_content") or d.get("raw_content") or ""))
    out.sort(key=lambda x: x[0])
    return out


def prepare(root, docs):
    """A GraphRAG project rooted at *root* with *docs* as {name: text}."""
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "input"), exist_ok=True)
    subprocess.run([sys.executable, "-m", "graphrag", "init", "--root", root],
                   input="\n\n", text=True, capture_output=True)
    for name, text in docs.items():
        pathlib.Path(root, "input", f"{name}.txt").write_text(text)
    pathlib.Path(root, "settings.yaml").write_text(
        SETTINGS.format(model=MODEL, emb=EMB_MODEL, key=KEY, base=BASE))
    return root


def index(root):
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", root],
                       capture_output=True, text=True)
    ok = p.returncode == 0
    return ok, time.time() - t0, (p.stderr or p.stdout)[-600:]


def query(root, question, method="local"):
    p = subprocess.run(
        [sys.executable, "-m", "graphrag", "query", "--root", root,
         "--method", method, question],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-300:])
    out = p.stdout
    # the CLI prefixes the answer with a banner; take what follows it
    m = re.split(r"SUCCESS: (?:Local|Global) Search Response:?", out)
    return (m[-1] if len(m) > 1 else out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--workroot", required=True)
    ap.add_argument("--method", default="local")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = json.loads(pathlib.Path(args.results).read_text())
    rows, spaces = src["results"], src["spaces"]
    instruction = _answer_instruction()

    roots = {}
    all_docs = {}
    for code, space in spaces.items():
        docs = {f"{space}-{q}": text for q, text in transcripts_for(args.data, code)}
        all_docs.update(docs)
        root = prepare(os.path.join(args.workroot, space), docs)
        ok, secs, tail = index(root)
        print(f"  indexed {code}: {len(docs)} quarters, ok={ok} in {secs:.0f}s", flush=True)
        if not ok:
            print(f"    {tail}", flush=True)
        roots[code] = root
    root = prepare(os.path.join(args.workroot, "all"), all_docs)
    ok, secs, tail = index(root)
    print(f"  indexed ALL: {len(all_docs)} quarters, ok={ok} in {secs:.0f}s", flush=True)
    roots["__ALL__"] = root

    by_space = {v: k for k, v in spaces.items()}
    results, failed = [], []
    for i, r in enumerate(rows, 1):
        code = by_space.get(r["space"])
        target = roots.get(code, roots["__ALL__"])
        try:
            answer = query(target, f"{instruction}\n\nQuestion: {r['question']}",
                           args.method)
        except Exception as e:
            failed.append({"question": r["question"][:80], "reason": str(e)[:200]})
            print(f"  [{i}/{len(rows)}] FAILED", flush=True)
            continue
        results.append({"space": r["space"], "companies": r["companies"],
                        "type": r["type"], "question": r["question"],
                        "gold": r["gold"], "answer": answer[:800]})
        print(f"  [{i}/{len(rows)}] {r['type']:<20} {len(answer):>5} chars", flush=True)

    pathlib.Path(args.out).write_text(json.dumps({
        "benchmark": "ECT-QA", "system": "GraphRAG", "method": args.method,
        "model": MODEL, "embedding_model": EMB_MODEL,
        "source_questions": args.results, "spaces": spaces,
        "failed": failed, "reportable": not failed,
        "results": results}, indent=2))
    print(f"\n  {len(results)} answered, {len(failed)} failed -> {args.out}")


def _answer_instruction():
    src = (HERE / "run.py").read_text()
    m = re.search(r"ANSWER_INSTRUCTION = \((.*?)\n\)\n", src, re.S)
    return eval("(" + m.group(1) + ")")


if __name__ == "__main__":
    main()

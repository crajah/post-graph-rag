#!/usr/bin/env python3
"""Download a corpus of related articles to evaluate multi-document indexing.

The articles are fetched rather than vendored so the repository stays small and
the text keeps its original licensing at the source.

    python evaluation/fetch_corpus.py
    python evaluation/fetch_corpus.py --articles Marie_Curie Pierre_Curie Radium
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

# Default corpus: heavily overlapping entities, which is what makes it a useful
# test of cross-document entity resolution rather than of extraction alone.
DEFAULT_ARTICLES = [
    "Ada_Lovelace",
    "Charles_Babbage",
    "Analytical_engine",
    "Difference_engine",
]

API = "https://en.wikipedia.org/w/api.php"


def fetch(title: str) -> tuple:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",  # without this, non-canonical titles return an empty extract
        "format": "json",
        "titles": title,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "post-graph-rag-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        pages = json.load(resp)["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("title", title), page.get("extract", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--articles", nargs="+", default=DEFAULT_ARTICLES, help="Wikipedia article titles")
    ap.add_argument("--out", default="evaluation/corpus", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for title in args.articles:
        resolved, text = fetch(title)
        if not text:
            print(f"  !! {title}: empty extract, skipped")
            continue
        path = os.path.join(args.out, f"{resolved.replace(' ', '_')}.txt")
        with open(path, "w") as f:
            f.write(text)
        total += len(text)
        print(f"  {resolved}: {len(text):,} chars -> {path}")
    print(f"\n{total:,} chars total in {args.out}")


if __name__ == "__main__":
    main()

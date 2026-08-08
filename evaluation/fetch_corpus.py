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

# A temporal corpus: three novels following the same characters across roughly
# forty years, during which alliances genuinely reverse. Public domain.
GUTENBERG_SERIES = {
    "dumas": [
        # (order, title, Gutenberg id, in-story period)
        (1, "The Three Musketeers", 1257, "1625"),
        (2, "Twenty Years After", 1259, "1648"),
        (3, "The Vicomte de Bragelonne", 2759, "1660"),
    ],
}
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the Project Gutenberg header and licence footer."""
    start = text.find("*** START OF")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find("*** END OF")
    return text[start: end if end != -1 else len(text)].strip()


def fetch_series(name: str, out_dir: str, max_chars: int) -> int:
    """Download a Gutenberg series, named so publication order sorts correctly."""
    total = 0
    for order, title, book_id, period in GUTENBERG_SERIES[name]:
        url = GUTENBERG_URL.format(id=book_id)
        req = urllib.request.Request(url, headers={"User-Agent": "post-graph-rag-eval/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        body = strip_gutenberg_boilerplate(raw)[:max_chars]
        # The numeric prefix keeps chronological order under a plain sort, which
        # is what supersession relies on: the later book must be indexed last.
        slug = f"{order:02d}_{title.replace(' ', '_')}_{period}.txt"
        path = os.path.join(out_dir, slug)
        with open(path, "w") as f:
            f.write(body)
        total += len(body)
        print(f"  {order}. {title} ({period}): {len(body):,} chars -> {path}")
    return total


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
    ap.add_argument("--series", choices=sorted(GUTENBERG_SERIES),
                    help="Fetch a Gutenberg novel series instead of Wikipedia articles")
    ap.add_argument("--max-chars", type=int, default=200_000,
                    help="Per-book truncation for series downloads")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.series:
        total = fetch_series(args.series, args.out, args.max_chars)
        print(f"\n{total:,} chars total in {args.out}")
        return
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

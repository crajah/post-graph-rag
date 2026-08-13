#!/usr/bin/env python3
"""Extract the article's Markdown tables to CSV for gist embedding.

Markdown emphasis is stripped and tick/cross symbols become words, because
GitHub interprets neither inside a CSV — an unconverted "❌ discouraged" renders
as the meaningless "No discouraged".
"""
import csv, pathlib, re, sys

# Positional: tables are zipped with this list in document order, so a new
# table means inserting its name at the right index, not appending.
NAMES = [
    "capability-comparison", "community-detector-comparison",
    "multi-hop-retrieval-by-depth",
    "relation-channel-on-topic-share", "retrieval-quota-by-question-shape",
    "lightrag-encyclopedic-prose",
    "lightrag-narrative-prose", "lightrag-financial-filings",
    "retrieval-recall-granularity-fix", "model-sensitivity",
]


def clean(cell: str) -> str:
    cell = cell.strip()
    cell = re.sub(r"\*\*(.+?)\*\*", r"\1", cell)
    cell = re.sub(r"\*(.+?)\*", r"\1", cell)
    cell = cell.replace("`", "")
    cell = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", cell)
    for sym, word in (("✅", "Yes"), ("❌", "No"), ("⚠️", "Partial")):
        cell = re.sub(re.escape(sym) + r"\s+(\S)", word + r" — \1", cell)
        cell = cell.replace(sym, word)
    return cell.strip()


def tables(md: str):
    out, cur = [], []
    for line in md.split("\n"):
        if line.lstrip().startswith("|"):
            cur.append(line)
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def main():
    here = pathlib.Path(__file__).parent
    md = (here.parent / "docs" / "index.md").read_text()
    found = tables(md)
    if len(found) != len(NAMES):
        sys.exit(f"expected {len(NAMES)} tables, found {len(found)} — update NAMES")
    for name, tbl in zip(NAMES, found):
        rows = []
        for line in tbl:
            cells = [clean(c) for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.strip() or "-") for c in cells):
                continue
            rows.append(cells)
        with (here / f"{name}.csv").open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"{name}.csv  ({len(rows) - 1} rows)")


if __name__ == "__main__":
    main()

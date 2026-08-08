#!/usr/bin/env python3
"""Download SEC 10-K annual filings as a temporal evaluation corpus.

    python evaluation/fetch_sec.py --company boeing --out evaluation/sec_boeing

Annual filings make an unusually good temporal corpus: the same line items recur
year after year while their meaning changes completely. A "deferred production
cost" that is a footnote in one filing can be the central driver of cash burn in
another, and the filings say so in their own words rather than in a dated field.

Filings are named with a numeric prefix so a plain sort gives chronological
order, which is what relation supersession resolves from.
"""
import argparse
import html
import json
import os
import re
import time
import urllib.request

SEC_UA = os.getenv("SEC_USER_AGENT", "post-graph-rag-eval research@example.com")
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
PAGED = "https://data.sec.gov/submissions/CIK{cik}-submissions-{page:03d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"

# Years chosen to span the operational cycles rather than sampled evenly: two
# peaks and three troughs, so relationships genuinely reverse between filings.
COMPANIES = {
    "boeing": {
        "cik": "0000012927",
        "label": "Boeing",
        # fiscal year -> what that filing is expected to show
        "years": {
            2006: "record deliveries, strong margins",
            2012: "787 supply chain and battery crisis",
            2018: "record revenue and free cash flow",
            2020: "737 MAX grounding, negative cash flow",
            2024: "quality crisis and FAA production caps",
        },
    },
}


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def html_to_text(raw: str) -> str:
    """Reduce filing HTML to readable prose.

    Filings are mostly tables of figures; the narrative sections are what carry
    the relationships worth extracting, so markup and cell padding are stripped
    rather than preserved.
    """
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?is)</(p|div|tr|h[1-6]|li)>", "\n", raw)
    raw = re.sub(r"(?is)</t[dh]>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Drop lines that are almost entirely digits: table rows carry no relations.
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 30:
            continue
        digits = sum(c.isdigit() for c in line)
        if digits > len(line) * 0.35:
            continue
        kept.append(line)
    return "\n".join(kept)


# Management's Discussion and Analysis is where a filing explains *why* the
# numbers moved — cash flow drivers, programme charges, concessions. Taking the
# head of the document instead yields business description and risk factors,
# which are boilerplate and barely change between years.
MDNA_START = re.compile(r"(?i)item\s*7[^0-9A-Za-z]{0,4}\s*management.{0,20}discussion")
MDNA_END = re.compile(r"(?i)item\s*9[^0-9A-Za-z]{0,4}\s*(changes\s+in|controls)")


def sample_lines(text: str, max_chars: int) -> str:
    """Take whole lines spread evenly across a block of text.

    Sampling by character stride shreds the prose into unreadable fragments —
    "Item 7 Management's Discussion" becomes "Itm7 aaeetsDsuso". Lines are the
    smallest unit that keeps sentences intact.
    """
    if len(text) <= max_chars:
        return text
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text[:max_chars]
    kept, used, step = [], 0, 1
    # Walk the lines at a stride that lands close to the character budget.
    avg = max(1, len(text) // max(1, len(lines)))
    step = max(1, round(len(lines) / max(1, max_chars // avg)))
    for i in range(0, len(lines), step):
        line = lines[i]
        if used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)


def select_narrative(text: str, max_chars: int) -> str:
    """Prefer MD&A through the financial statements; fall back to the whole filing.

    Filings repeat the Item 7 heading in the table of contents, so the *last*
    match is the section itself rather than its index entry.
    """
    starts = list(MDNA_START.finditer(text))
    if starts:
        begin = starts[-1].start()
        tail = text[begin:]
        end = MDNA_END.search(tail)
        section = tail[: end.start()] if end else tail
        if len(section) >= 20_000:
            return sample_lines(section, max_chars)

    # No usable MD&A: spread across the whole filing so later narrative sections
    # are represented, not just the boilerplate opening.
    return sample_lines(text, max_chars)


def list_filings(cik: str) -> list:
    """All 10-K filings, newest first, across the paginated submission files."""
    out = []
    data = get_json(SUBMISSIONS.format(cik=cik))
    recent = data["filings"]["recent"]
    out += [
        (recent["filingDate"][i], recent["accessionNumber"][i], recent["primaryDocument"][i])
        for i, f in enumerate(recent["form"]) if f == "10-K"
    ]
    for extra in data["filings"].get("files", []):
        page = get_json(f"https://data.sec.gov/submissions/{extra['name']}")
        time.sleep(0.2)                      # SEC asks for <10 req/s
        out += [
            (page["filingDate"][i], page["accessionNumber"][i], page["primaryDocument"][i])
            for i, f in enumerate(page["form"]) if f == "10-K"
        ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", default="boeing", choices=sorted(COMPANIES))
    ap.add_argument("--out", default="evaluation/sec_boeing")
    ap.add_argument("--max-chars", type=int, default=180_000, help="Per-filing truncation")
    ap.add_argument("--years", nargs="*", type=int, help="Override the fiscal years fetched")
    args = ap.parse_args()

    spec = COMPANIES[args.company]
    cik, cik_int = spec["cik"], str(int(spec["cik"]))
    wanted = args.years or sorted(spec["years"])
    os.makedirs(args.out, exist_ok=True)

    filings = list_filings(cik)
    print(f"{spec['label']}: {len(filings)} 10-K filings on EDGAR\n")

    total = 0
    for order, fy in enumerate(sorted(wanted), start=1):
        # A fiscal year is reported in the following calendar year's filing.
        match = next((f for f in filings if f[0].startswith(str(fy + 1))), None)
        if match is None:
            print(f"  !! FY{fy}: no filing found, skipped")
            continue
        filed, accession, doc = match
        url = ARCHIVE.format(cik_int=cik_int, accession=accession.replace("-", ""), doc=doc)
        req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        time.sleep(0.3)

        body = select_narrative(html_to_text(raw), args.max_chars)
        path = os.path.join(args.out, f"{order:02d}_{spec['label']}_10K_FY{fy}.txt")
        with open(path, "w") as f:
            f.write(body)
        total += len(body)
        note = spec["years"].get(fy, "")
        print(f"  {order}. FY{fy} (filed {filed}): {len(body):,} chars -> {os.path.basename(path)}")
        if note:
            print(f"       expected: {note}")

    print(f"\n{total:,} chars total in {args.out}")


if __name__ == "__main__":
    main()

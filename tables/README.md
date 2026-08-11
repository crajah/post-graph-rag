# Article tables, as CSV

Medium supports neither Markdown import nor tables. GitHub renders `.csv` gists
as real tables, and Medium's gist embed shows that rendering — so each file here
becomes one table in the published article.

For each: create a gist at https://gist.github.com containing the file, then
paste the gist URL on its own line in the Medium editor. The gist filename is
visible in the embed, which is why these are named for their content.

| File | Table in the article |
| :--- | :--- |
| `capability-comparison.csv` | GraphRAG / LightRAG / post-graph-rag capabilities |
| `community-detector-comparison.csv` | Leiden vs label propagation |
| `multi-hop-retrieval-by-depth.csv` | Relations retrieved at 1, 2 and 3 hops |
| `lightrag-encyclopedic-prose.csv` | Wikipedia corpus vs LightRAG |
| `lightrag-narrative-prose.csv` | Dumas trilogy vs LightRAG |
| `lightrag-financial-filings.csv` | Boeing 10-K corpus vs LightRAG |
| `retrieval-recall-granularity-fix.csv` | Recall before and after the granularity rule |
| `model-sensitivity.csv` | Extraction quality by model |

Generated from `docs/index.md`; regenerate with `python3 tables/extract.py` after
editing the article.

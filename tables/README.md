# Article tables, as CSV

Medium supports neither Markdown import nor tables. GitHub renders `.csv` gists
as real tables, and Medium's gist embed shows that rendering — so each file here
becomes one table in the published article.

Paste the gist URL on its own line in the Medium editor and press Enter; it
expands into the table. The gist filename is visible in the embed, which is why
these are named for their content.

| File | Table in the article | Gist |
| :--- | :--- | :--- |
| `capability-comparison.csv` | GraphRAG / LightRAG / post-graph-rag capabilities | [fc5bbe32](https://gist.github.com/crajah/fc5bbe32b020b2afe1abcd61adaf57c4) |
| `community-detector-comparison.csv` | Leiden vs label propagation | [b48ce8a7](https://gist.github.com/crajah/b48ce8a7d504df7692e35e195bc37d18) |
| `multi-hop-retrieval-by-depth.csv` | Relations retrieved at 1, 2 and 3 hops | [07e3cd69](https://gist.github.com/crajah/07e3cd695085f70459da3f22483e7531) |
| `lightrag-encyclopedic-prose.csv` | Wikipedia corpus vs LightRAG | [9ddca709](https://gist.github.com/crajah/9ddca7090f0c1797b692f76c865aab70) |
| `lightrag-narrative-prose.csv` | Dumas trilogy vs LightRAG | [b5d8c200](https://gist.github.com/crajah/b5d8c20068a3eb80bfade1d440b91059) |
| `lightrag-financial-filings.csv` | Boeing 10-K corpus vs LightRAG | [48d83f71](https://gist.github.com/crajah/48d83f711a46ce02a44d0cc61572a240) |
| `retrieval-recall-granularity-fix.csv` | Recall before and after the granularity rule | [5c0e8999](https://gist.github.com/crajah/5c0e8999d6a6659b02907d06988ed3e0) |
| `model-sensitivity.csv` | Extraction quality by model | [a83c724e](https://gist.github.com/crajah/a83c724e8ecfc958d68d62d3a9e0f7e9) |

Note: **post-graph** has a table with the same filename but different content and
a different gist — check the repo before embedding a `capability-comparison`.

Editing a CSV here does not update its gist. Push the change with
`gh gist edit <id> tables/<file>.csv`, which updates any Medium embed in place.

Regenerate from the article with `python3 tables/extract.py`. It fails if the
article's table count drifts from the expected eight, rather than silently
writing the wrong content to a familiar filename.

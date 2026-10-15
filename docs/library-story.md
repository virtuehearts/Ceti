# Library & paper writer

Two halves of one idea: drop source material into CETI, and the
oracle writes long-form original work informed by it.

## Library (`/ingest`)

`data/library/` is a drop folder. Put `.pdf`, `.txt`, or `.md`
files in it; `/ingest` (or `Library.scan()`) picks them up:

1. **Extract** — `pdftotext -layout` for PDFs, direct read for
   text/markdown.
2. **Chunk** — paragraph-aware, ~600 tokens with ~120 overlap.
3. **Embed** — embeddinggemma-2 via the local embedding server
   (`:11435`). ~4s/chunk on CPU — a full book is a ~30 min
   background job; let it run.
4. **Store** — `data/library.sqlite3`, `doc_chunks` table
   (separate from the Q&A knowledge base). Files are tracked by
   sha256; re-drops are skipped.

`Library.search(query, top_k)` does cosine search over chunks.
The embedding server must be up (`llama-server --embedding`
on :11435) — `/ingest` says so loudly if it isn't.

The library is a **style/idea reference**, not a copy machine:
retrieved passages go into prompts as voice inspiration, and the
oracle is instructed never to reproduce them.

## Paper writer (`/story`)

StoryBlocks-style long-form generation. A story is a sequence of
blocks (~200 words/page, 1–10 pages per block, retried if short):

```
/story new <premise>        # writes the story bible, prints the id
/story write <id> [pages] [count]
/story status <id>
/story compile <id>         # story.md + story.txt
/story list
```

Coherence across 10–50 pages comes from three layers:

1. **Story bible** — written first: title, logline, characters,
   three-act arc, 12 chapter beats. Each block follows the next
   beat.
2. **Rolling summary** — every chapter is summarized into
   `summary.md`; the next chapter is generated against it.
   Summaries compress when they get long.
3. **Style retrieval** — top-k library chunks for the current
   beat go into the prompt as voice reference.

Stories live as plain files under `data/stories/<id>/`:
`bible.md`, `summary.md`, `blocks/01.md`…, `story.md`.

Needs a cloud backend with an API key — the oracle does the
writing. A 50-page run is an overnight job: run
`/story write <id> 3 15` and let the box work. Temperature 0.8
for prose, 0.3 for summaries.

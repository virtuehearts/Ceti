# CETI Modules — the modular knowledge standard

CETI Modules extend AIML with four things: a module header, explicit
facts, provenance, and domains as directories. A module is one file,
one assigned domain. The file is the unit of authorship, signing,
versioning, distribution, and enable/disable. The knowledge graph is a
DERIVED INDEX built from the modules at boot/ingest (compile-first).
Files are the source of truth; edit a file, re-index that module. The
graph never becomes a second store that drifts from the files.

## Format

```xml
<?xml version="1.0" encoding="UTF-8"?>
<module domain="geography" name="countries" version="1-partial"
        source="browser search (Wikipedia infoboxes, ISO 3166-1, Statvoo, country.io)"
        confidence="1.0">
  <fact subject="France" predicate="capital" object="Paris"/>
  <fact subject="France" predicate="currency" object="Euro"/>
  <!-- standard AIML categories for narrative / conversational pairs -->
  <category>
    <pattern>WHAT IS FRANCE</pattern>
    <template>France is a country in Europe.</template>
  </category>
</module>
```

- `<module domain name version source confidence>`: header. `domain`
  is the assigned domain; `name` is the module name; `version` is a
  string (e.g. `1`, `1-partial`); `source` records where the facts
  came from; `confidence` is the module's default confidence (0–1).
  A file with no header loads as a LEGACY module (confidence 0.6,
  domain by filename keywords).
- `<fact subject predicate object/>`: explicit structured facts.
  Predicates: the fact-store vocabulary (introduced, released,
  founded, created, ram, cpu, price, units_sold, author, creator,
  designer, inventor, is, named_after, successor, predecessor) plus
  the attribute predicates (capital, currency, currency_code,
  population, region, languages, iso_alpha2, iso_alpha3).
- `<category>`: standard AIML pattern/template pairs, for narrative,
  conversational, and smalltalk knowledge the graph can't express.
- Domain precedence: explicit header domain > subdirectory name >
  filename inference.

## Layout

```
data/cassettes/<domain>/<module>.aiml
data/cassettes/<domain>/<module>.aiml.gz   (gzip, loads transparently)
data/cassettes/*.aiml                      (legacy flat files, still work)
```

Domains: geography, literature, poetry, personality, math, bbs,
computing (C64 etc. under computing/retro), plus keyword-inferred
domains for legacy files (telecom, demoscene, general, ...).
Existing flat cassettes stay flat — the cassette rack only reads
top-level files, so flat files are loaded as legacy modules with
logical domain assignment. New modules go in domain subdirs.

## Lobes on the bus

- **Occipital (recognizer):** the AIML pattern engine (`ceti/aiml.py`).
  Turns text into recognized patterns. Kept as-is.
- **Temporal (memory):** the store — graph index + cassettes
  (narrative) + library (vector) + reference modules. Built from
  module files, compile-first.
- **Parietal (computer):** the calculator (Python AST, never the LLM)
  + structured operations (graph traversal, unit handling). Operates
  on knowledge, does not store it.
- **Frontal (reasoner/composer):** derives intent (action + entity +
  attribute), gathers pieces from the other lobes, composes the
  answer, grounds it (every claim traces to a piece). Serve-time
  composition is RULES ONLY, no model. Heavy reasoning (learning,
  judging, distilling) happens at idle/compile time. This preserves
  zero-token serve.

Serving order in `Brain._tier_memory`: graph lookup (any phrasing that
extracts to the same subject/predicate) → composer explain (graph
facts + a library hit) → library → knowledge (vector) → local model.
The graph lookup is a stored fact's render, so it is grounded by
construction; the composer checks grounding and falls back to quoting
the best piece verbatim when grounding fails. `memory_source` in the
trace reports the fact's ORIGIN: `facts` for learned facts, `graph`
for module-only facts, `composer` for composed answers, `library` /
`knowledge` for the older paths.

## Learning: one gate

All structured learning goes through `Brain.learn_fact(question,
answer, score, source)`. The graph checks the fact against module +
learned facts (it is a superset of the fact store: modules + replayed
learned facts), so a graph pass implies a fact-store pass. On pass,
the fact is persisted to the fact store. On block, the caller drops
the pair. `Brain._learn` and the researcher (`_record`) both use it.

Conflict rule (unchanged from the fact store): a conflicting fact
with >= confidence blocks the whole learn; higher confidence
supersedes (old fact kept, marked superseded, audit trail).
Reference modules (confidence 1.0) outrank learned facts.

Personality is a module like any other: the persona lives in a file,
not in code. Swapping personas = swapping a file. Identity separation
is structural — the personality module never loads other domains, and
no other domain loads it. (The current persona stays in `persona.md`
for now; the module format supports a personality domain when it is
migrated. Do NOT migrate it until the tests that depend on `persona.md`
are updated.)

Math module holds reference facts only (constants, units,
conversions). The calculator stays Python code.

## Compression

- **Dictionary encoding (interning):** the graph stores each unique
  string once (`_strings` + `_ids`) and references facts by integer
  ID (~50–70% off the fact store).
- **gzip:** module files load transparently from `.aiml.gz`;
  signed dataset bundles are already gzip (`tarfile "w:gz"`).
- **Deferred:** library chunk gzip (would change the vectormem
  SQLite format; risky). Documented gap, not a silent omission.

## Reference data

`data/cassettes/geography/countries.aiml` — 30 countries x 8
predicates = 240 facts (capital, currency, currency_code, region,
population, languages, iso_alpha2, iso_alpha3), confidence 1.0,
version `1-partial`. Source: verified individually via browser search
(Wikipedia infoboxes, ISO 3166-1 alpha-3 list, Statvoo, country.io)
because restcountries v3.1 is deprecated (returns a deprecation
error; v5 needs an account + API key). Populations are mixed-vintage
estimates, not one snapshot. Full ~195 coverage needs a keyed v5
fetch or another bulk source.

## Benchmark

`tests/test_learning_benchmark.py` + `tests/testdata_learning_benchmark.json`:
78 held-out questions + 78 learning pairs (parallel order, disjoint
phrasing). Measured on the lobe system: before 0/78 (0.0%) -> after
78/78 (100.0%), delta +100.0%. Leakage check clean (0 leaks).
See the benchmark module for the leakage rules.

## Honest limits

- Entity linking is rule-based. The composer title-cases entities,
  which mangles acronyms ("USA" -> "Usa"); the Brain path matches
  entities case-insensitively as mitigation.
- The composer's compare is not wired into Brain yet (it needs both
  entities made explicit by the caller); explain is wired.
- Handler grounding is a coarse filter (numbers + 40% word overlap,
  8-word knowledge floor); subtle inventions can still pass. The
  fact/graph layer is the backstop for structured facts.
- Truth judge is the same small model checking its own output, not
  a guarantee.
- The fact extractor is rule-based with a closed predicate
  vocabulary; pairs outside it learn as raw pairs (no worse than
  before, just not structured).

# Facts — learning as structured knowledge, not flat Q&A pairs.

## The problem with pairs

`Brain._learn` stored a flat (question, answer) pair. That has three
holes:

1. **Paraphrases don't match.** Learning "when was the C64 introduced →
   1982" can't answer "what year did the C64 come out" — different
   wording, no AIML hit, falls through to the model.
2. **Conflicts hide behind wording.** The contradiction check needed
   exact-question equality, so "when was the C64 introduced → 1977"
   learned under a different phrasing sailed past it.
3. **No structure, no provenance.** A pair carries no confidence, no
   source, no way to supersede it when better knowledge arrives.

## The fix

The fact base is the source of truth for learned knowledge. AIML is a
compiled index over it, not the store itself.

- **Fact** = (subject, predicate, object) + confidence + source +
  created + superseded. A closed predicate vocabulary (see
  `PREDICATES`) keeps extraction deterministic.
- **Extractor** maps (question, answer) -> Fact, rule-based, no model.
  Pairs that don't extract stay as raw Q&A — nothing is lost.
- **Conflict resolution at the (subject, predicate) level.** A new fact
  with the same subject + predicate but a different object is blocked
  if the existing fact's confidence >= the new one; otherwise the new
  fact supersedes the old. Conflicts are detected by meaning, not
  wording — this closes the paraphrase-inconsistency hole.
- **AIML compilation.** One fact -> several paraphrase patterns, one
  rendered template. `compile_aiml()` returns the (pattern, template)
  pairs; the Brain registers them in-memory (`learn_compiled`) on
  startup and on every learn/supersede — they are NOT written to
  `learned.aiml` (that file holds raw pairs only). Serving is
  unchanged: the AIML tier matches the compiled patterns (exact
  hits are O(1)). A superseded fact re-registers its patterns with
  `override=True`, so the instant tier answers from the new fact.
- **The researcher uses the same gate.** The idle researcher's
  `_record()` routes every distilled pair through `FactStore.learn()`
  (source="research", confidence 0.9) before writing it to a
  cassette. A conflicting distilled pair is dropped — this closes
  the distiller-bypass that used to write cassettes with no gates
  at all. Pairs that don't extract to a fact still pass through,
  so the human audit before shipping still applies to those.
- **Fact answering.** `answer(question)` extracts (subject, predicate)
  from the question and renders the fact. Any phrasing that extracts
  to the same (subject, predicate) hits — even a novel paraphrase the
  compiler never generated a pattern for.

## Confidence

- research cassette / curated source: 0.9
- cloud oracle: 0.8
- local model: 0.6
- raw pair (no fact extracted): not a fact, no confidence

A fact with higher confidence supersedes a lower one. Equal
confidence: the existing fact wins (first-write wins on ties, so a
repeat learn can't flip a fact by re-asserting it).

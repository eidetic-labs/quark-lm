---
title: Unknown Policy
description: How QuarkLM should answer outside the corpus.
---

# Unknown Policy

QuarkLM answers a question only when the admitted corpus can support the answer.
When the asked-for fact is outside that corpus, the correct output is `unknown.`,
not a plausible-looking guess. This rail is what keeps the prototype honest about
the edge of its own closed world.

The deterministic responder is the surface that enforces this. It either answers
from admitted knowledge with provenance or returns `unknown.` — it never invents
an answer from a nearby surface form or from the surrounding world. For where the
responder sits in the system, see [Build](../build/index.md).

## The default is `unknown.`

The responder matches a question against a fixed set of admitted patterns:
glossary definitions, story facts, self-facts, learning rules, and
training-data-status questions. If none of those patterns is satisfied by the
admitted corpus, the responder falls through to `unknown.`. Declining is the
default, and answering is the exception that has to be earned by an admitted
source.

```text
question
   -> matches an admitted pattern, and the corpus holds the answer? -> grounded answer + provenance
   -> otherwise                                                      -> unknown.
```

This ordering matters. A model that guesses by default looks more capable on any
single prompt, but its confidence stops tracking what it actually has evidence
for. QuarkLM inverts that: an honest `unknown.` is a correct output, not a
failure, because it reports the true boundary of the admitted corpus.

## Training-data-status questions

One question shape has a stricter contract than a plain `unknown.`. A
training-data-status question asks whether a particular fact is part of the
admitted training set. For these, the responder reports membership directly:

| Asked about | Admitted in corpus | Answer |
| --- | --- | --- |
| A fact named in the corpus | yes | `yes.` |
| A fact outside the corpus | no | `no.` |

The distinction is deliberate. A plain content question about an unknown fact
returns `unknown.` because the responder genuinely cannot supply the fact. A
training-data-status question about an unknown fact returns `no.` because the
responder *can* answer it with certainty: the fact is not in the admitted set.
Knowing what is absent is itself grounded knowledge, and the policy keeps the two
cases from collapsing into one.

## What the rail refuses to do

| Behavior | Policy |
| --- | --- |
| Answer from a near-miss surface form when the corpus has no matching fact | Refused; returns `unknown.` |
| Borrow an answer from world knowledge outside the admitted corpus | Refused; there is no such knowledge to borrow from. |
| Report `yes.` for a fact that is not admitted | Refused; absent facts answer `no.` |
| Produce a fluent guess to avoid saying `unknown.` | Refused; `unknown.` is the correct output. |

The responder moves no weights when it answers or declines. A grounded answer
proves the admitted corpus contains the fact; it does not prove the transformer
learned it. That separation — `memory-served` versus `weight-consolidated` — is
the same one the rest of the project holds throughout; see the three evidence
states in [Language model](../learn/language-model.md).

## Why this is a security surface

The unknown policy is not a quality-of-answer feature. It is the part of the
[epistemic boundary](./index.md) that faces outward, toward questions the corpus
was never given. If the responder were allowed to guess past its evidence, the
project could no longer claim that every answer traces to a ledgered source. The
[purity boundary](./purity-boundary.md) governs what may enter the weights; the
unknown policy governs what may leave the responder. Both exist so that a claim
and its evidence never drift apart.

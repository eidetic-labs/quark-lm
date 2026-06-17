---
title: Unknown Policy
description: How QuarkLM should answer outside the corpus.
---

# Unknown Policy

<p className="qlm-meta"><span>5 min read</span><span>Security</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why declining is the default and an answer is the earned exception.
- How a plain `unknown.` differs from the stricter `yes.` / `no.` contract for training-data-status questions.
- What the rail refuses to do, and why a fluent guess is the wrong output.
- Why the unknown policy is the outward-facing half of the epistemic boundary.

</div>

QuarkLM answers a question only when the admitted corpus can support the answer.
When the asked-for fact is outside that corpus, the correct output is `unknown.`,
not a plausible-looking guess. This rail is what keeps the prototype honest about
the edge of its own closed world.

The deterministic responder is the surface that enforces this. It either answers
from admitted knowledge with provenance or returns `unknown.` — it never invents
an answer from a nearby surface form or from the surrounding world. For where the
responder sits in the system, see [Build](../build/index.md).

<div className="qlm-keypoint">

**Declining is the default; answering is the earned exception**

A model that guesses by default looks more capable on any single prompt, but its
confidence stops tracking what it actually has evidence for. QuarkLM inverts
that: an honest `unknown.` is a correct output, not a failure, because it reports
the true boundary of the admitted corpus.

</div>

## The default is `unknown.`

The responder matches a question against a fixed set of admitted patterns:
glossary definitions, story facts, self-facts, learning rules, and
training-data-status questions. If none of those patterns is satisfied by the
admitted corpus, the responder falls through to `unknown.`. Declining is the
default, and answering is the exception that has to be earned by an admitted
source.

```text title="The responder's answer-or-decline ordering"
question
   -> matches an admitted pattern, and the corpus holds the answer? -> grounded answer + provenance
   -> otherwise                                                      -> unknown.
```

This ordering matters. The grounded branch carries provenance back to an admitted
source; the fall-through branch reports the boundary plainly. Neither branch
fabricates content the corpus cannot supply.

## Training-data-status questions

One question shape has a stricter contract than a plain `unknown.`. A
training-data-status question asks whether a particular fact is part of the
admitted training set. For these, the responder reports membership directly.

<div className="qlm-grid">
<div><h4><code>yes.</code></h4><p>A fact <strong>named in the corpus</strong> and admitted: the responder confirms membership directly.</p></div>
<div><h4><code>no.</code></h4><p>A fact <strong>outside the corpus</strong> and not admitted: the responder denies membership with certainty.</p></div>
</div>

The distinction is deliberate. A plain content question about an unknown fact
returns `unknown.` because the responder genuinely cannot supply the fact. A
training-data-status question about an unknown fact returns `no.` because the
responder *can* answer it with certainty: the fact is not in the admitted set.

<div className="qlm-keypoint">

**Knowing what is absent is itself grounded knowledge**

A plain content probe and a membership probe must not collapse into one output.
`unknown.` reports that the responder cannot supply a fact; `no.` reports the
certain truth that the fact was never admitted. The policy keeps the two cases
apart.

</div>

## What the rail refuses to do

<div className="qlm-grid">
<div><h4>No near-miss answers</h4><p>Answering from a near-miss surface form when the corpus has no matching fact is refused; the responder returns <code>unknown.</code></p></div>
<div><h4>No borrowed world knowledge</h4><p>Borrowing an answer from world knowledge outside the admitted corpus is refused; there is no such knowledge to borrow from.</p></div>
<div><h4>No false membership</h4><p>Reporting <code>yes.</code> for a fact that is not admitted is refused; absent facts answer <code>no.</code></p></div>
<div><h4>No fluent guessing</h4><p>Producing a fluent guess to avoid saying <code>unknown.</code> is refused; <code>unknown.</code> is the correct output.</p></div>
</div>

The responder moves no weights when it answers or declines. A grounded answer
proves the admitted corpus contains the fact; it does not prove the transformer
learned it.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

A grounded answer proves the corpus holds the fact; it does not prove the weights
learned it. That separation — `memory-served` versus `weight-consolidated` — is
the same one the rest of the project holds throughout; see the three evidence
states in [Language model](../learn/language-model.md).

</div>

## Why this is a security surface

The unknown policy is not a quality-of-answer feature. It is the part of the
[epistemic boundary](./index.md) that faces outward, toward questions the corpus
was never given. If the responder were allowed to guess past its evidence, the
project could no longer claim that every answer traces to a ledgered source. The
[purity boundary](./purity-boundary.md) governs what may enter the weights; the
unknown policy governs what may leave the responder. Both exist so that a claim
and its evidence never drift apart.

:::note

The purity boundary and the unknown policy are two halves of the same contract.
One controls admission *into* the weights; the other controls what may *leave*
the responder. Read them together to see the full boundary.

:::

<div className="qlm-next">

<a href="./purity-boundary.md"><strong>Read next</strong><span>Purity boundary</span><small>What may enter the weights, and why the ledger is the admission gate.</small></a>

<a href="./index.md"><strong>Step back</strong><span>Secure</span><small>The epistemic boundary and the three claims the secure pages hold in place.</small></a>

<a href="../learn/language-model.md"><strong>Go deeper</strong><span>Language model</span><small>The three evidence states behind memory-served versus weight-consolidated.</small></a>

</div>

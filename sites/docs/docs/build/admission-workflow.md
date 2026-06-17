---
title: Admission Workflow
description: Admit new knowledge before it can be trained.
---

# Admission Workflow

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will do**

- Admit a single fact to the ledgered corpus, or admit a batch.
- Satisfy the six-field fact contract and let duplicate ids be rejected.
- Sync admission probes, read the admission status block, and run the cycle that follows.

</div>

Admission is how a new fact enters QuarkLM's closed world. A fact is not training
data because someone typed it in chat. It becomes training-eligible only after it
is admitted to the ledgered corpus, and it becomes learned weight behavior only
after a guarded update is evaluated and promoted. Admission is the first of those
steps and the only one this page covers.

The `admit` module appends a fact to `corpus/admissions.jsonl`, which is named in
`corpus/ledger.json` as the `admissions-v0` source and marked allowed for both
curriculum generation and training. Writing the fact through `admit` is what
makes it part of the admitted corpus; the [purity boundary](../secure/purity-boundary.md)
is the reason no other path may introduce training text.

<div className="qlm-keypoint">

**Admitted is not weight-consolidated**

Writing a fact through `admit` makes it curriculum- and training-*eligible*. It
moves no neural weights. An admitted fact is `memory-served` once curriculum and
retrieval are rebuilt; it is `weight-consolidated` only if a later run promotes a
guarded update that learned it.

</div>

## What admission does, and does not do

<div className="qlm-grid">
<div><h4>Does: append</h4><p>Appends a validated fact to the admitted corpus.</p></div>
<div><h4>Does not: move weights</h4><p>Moves no neural weights.</p></div>
<div><h4>Does: reject duplicates</h4><p>Rejects duplicate ids before writing.</p></div>
<div><h4>Does not: train</h4><p>Does not train, evaluate, or promote anything.</p></div>
<div><h4>Does: sync probes</h4><p>Regenerates admission probes from the new corpus (default paths).</p></div>
<div><h4>Does not: rebuild artifacts</h4><p>Does not regenerate <code>build/train.txt</code> or memory cards.</p></div>
<div><h4>Does: make eligible</h4><p>Makes the fact curriculum- and training-<em>eligible</em>.</p></div>
<div><h4>Does not: consolidate</h4><p>Does not make the fact <code>weight-consolidated</code>.</p></div>
</div>

Admission writes one log line. Everything downstream — curriculum text,
retrieval memory cards, candidate proposals, guarded updates — is run separately
and audited separately. Those two states (`memory-served` and `weight-consolidated`)
are kept distinct everywhere; see [Build](./index.md).

## Admit one fact

Run from the project root with `PYTHONPATH=src` set.

```bash title="Admit a single fact"
PYTHONPATH=src python3 -m admit \
  --id learned-child-book \
  --person child \
  --object book \
  --color blue \
  --relation on \
  --container table
```

## Admit a batch

```bash title="Admit a batch from a JSONL file"
PYTHONPATH=src python3 -m admit \
  --batch path/to/new_admissions.jsonl
```

Each line of the batch file is one fact in the same shape as the single-fact
fields.

## The fact contract

Every admitted fact carries the same six required fields. The id may contain
lowercase letters, digits, and hyphens, and must start with a letter; the other
five fields must be lowercase letters only.

<div className="qlm-grid">
<div><h4><code>id</code></h4><p>Stable admission id, unique across the corpus. Allowed: <code>[a-z][a-z0-9-]*</code>.</p></div>
<div><h4><code>person</code></h4><p>Who the fact is about. Allowed: lowercase letters.</p></div>
<div><h4><code>object</code></h4><p>The thing. Allowed: lowercase letters.</p></div>
<div><h4><code>color</code></h4><p>Its color. Allowed: lowercase letters.</p></div>
<div><h4><code>relation</code></h4><p>Where it sits. Allowed: lowercase letters.</p></div>
<div><h4><code>container</code></h4><p>What it sits on or in. Allowed: lowercase letters.</p></div>
</div>

A fact missing any required field is rejected, as is any field whose characters
fall outside the pattern. This keeps admitted facts inside the nursery
vocabulary the corpus tokenizer learns from.

:::note

Ids are the corpus's primary key. Before writing, `admit` rejects an id that
already exists in `corpus/admissions.jsonl` and rejects an id that appears more
than once within the same batch. Nothing is written when a duplicate is found,
so a rejected batch leaves the corpus unchanged.

:::

## Probe sync

When the admission writes to the default `corpus/admissions.jsonl`, `admit` also
regenerates the direct and paraphrase admission probes so admitted-memory evals
stay aligned with admitted facts. Pass `--no-sync-probes` to skip that
regeneration, or `--path` to write to a non-default file (in which case probes
are not synced unless probe paths are given explicitly). How those probes are
generated and checked is described in [Generated probes](./generated-probes.md).

## What admission reports

The command prints the admitted record(s) and a status block. The status is
`admitted_pending_weight_update`: the fact is in the corpus and answerable once
curriculum and memory are rebuilt, but no weights have changed. The reported next
step is to run a self-improvement answer cycle, which is where a guarded update is
attempted and a promotion gate decides whether anything is kept.

```text title="The path a fact travels after admission"
admit -> corpus/admissions.jsonl  (admitted, no weights moved)
      -> regenerate curriculum + memory  (memory-served)
      -> self_improve answer-cycle  (guarded update, may be rejected)
      -> promotion gate  (weight-consolidated only if promoted)
```

## After admission

Rebuild the curriculum and retrieval artifacts so the fact is answerable, then
run a self-improvement cycle that compares against the last promoted report.

<ol className="qlm-steps">
<li><strong>Rebuild curriculum and retrieval</strong><p>Regenerate the artifacts so the admitted fact is answerable as <code>memory-served</code>.</p></li>
<li><strong>Run the answer cycle</strong><p>Compare against the last promoted report; a guarded update is attempted here.</p></li>
<li><strong>Let the promotion gate decide</strong><p>A guarded update is kept only if the recorded promotion gate passes — completing a run does not promote it.</p></li>
</ol>

```bash title="Run a self-improvement answer cycle"
PYTHONPATH=src python3 -m self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.38/self_improvement_report.json
```

The cycle does not promote a run because it completed. Candidates built from the
admitted fact are held in [candidate quarantine](../operate/candidate-quarantine.md)
until the [closed-world verifier](../operate/closed-world-verifier.md) clears
them, and any guarded update is kept only if the recorded promotion gate passes.

<div className="qlm-next">
<a href="./quickstart.md"><strong>Read next</strong><span>Quickstart</span><small>The full run sequence end to end.</small></a>
<a href="../learn/self-improvement-loop.md"><strong>Read next</strong><span>Self-improvement loop</span><small>The lifecycle contract this admission feeds.</small></a>
<a href="./generated-probes.md"><strong>Reference</strong><span>Generated probes</span><small>How admission probes are generated and checked.</small></a>
</div>

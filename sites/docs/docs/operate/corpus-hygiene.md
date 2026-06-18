---
title: Corpus Hygiene
description: Corpus hygiene and training-plan artifacts for QuarkLM runs.
---

# Corpus Hygiene

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- What `corpus_hygiene.json` and `training_plan.json` record, and why a run writes both before its metrics count.
- Why hygiene is descriptive while the verifier and the promotion gate are decisive.
- How the candidate ratio keeps generated examples out of training until they are admitted.
- Why the hygiene trail is part of the evidence bundle, not a promotion shortcut.

</div>

Corpus hygiene is the first evidence surface a run produces. Before any metric is
read, it makes the *shape of the data* visible: where the training text came
from, whether examples are duplicated, and whether the eval set overlaps the
training set. A run whose numbers look good for the wrong reason — leaked eval
prompts, a lopsided source mixture — is caught here, not after promotion.

`src/corpus_hygiene.py` (added in v0.73) writes two required run artifacts:

<div className="qlm-grid">
<div><h4>corpus_hygiene.json</h4><p>What the data looks like — sources, duplicates, overlap, coverage.</p></div>
<div><h4>training_plan.json</h4><p>What the run is permitted to do with it, inside the closed-world boundary.</p></div>
</div>

Self-improvement answer cycles and transformer answer-training runs write both
artifacts before their metrics are treated as evidence. The artifacts do not
promote or reject a model. They make data risk legible so the gates downstream —
the [closed-world verifier](./closed-world-verifier.md) and constraint-first
promotion — can act on it.

## Corpus growth preflight

`corpus_growth_plan.json` is the pre-admission companion to hygiene. It is
written before a proposed batch is appended to `corpus/admissions.jsonl`, so the
project can inspect growth pressure before it changes the corpus.

```bash title="Preflight an admission batch"
PYTHONPATH=src python3 -m corpus_growth_plan \
  --batch batches/new-facts.jsonl \
  --output build/corpus_growth_plan.json
```

<div className="qlm-grid">
<div><h4>Source provenance</h4><p>Batch path, admissions path, corpus directory, and eval files used for the check.</p></div>
<div><h4>Duplicate checks</h4><p>Duplicate ids inside the batch, conflicts with existing admissions, and repeated person/object fact keys.</p></div>
<div><h4>Train/eval split checks</h4><p>Generated direct and paraphrase probes are compared against eval prompts before admission.</p></div>
<div><h4>Retention probes</h4><p>Existing admitted facts are sampled so the next training run must retain prior knowledge.</p></div>
<div><h4>Unknown-policy probes</h4><p>Outside-corpus prompts are proposed with the expected `unknown` target.</p></div>
<div><h4>Tokenizer stress strings</h4><p>Longer corpus-derived strings are listed for tokenizer compression pressure.</p></div>
</div>

The growth plan is read-only. A passing report means the batch is ready to be
admitted by the normal admission command; it does not itself admit data or train
weights.

## Where hygiene sits in the chain

```text title="Corpus hygiene in the evidence chain"
admitted corpus (ledger.json)
  -> corpus_hygiene.json     describe the data: sources, duplicates, overlap, coverage
  -> training_plan.json      declare what the run may train on, inside the boundary
  -> closed_world_verifier   decide whether the plan may influence the next step
  -> constraint-first gate   only then are quality metrics allowed to count
```

<div className="qlm-keypoint">

**Descriptive, not decisive**

Hygiene describes the data; the verifier and the gate decide. Keeping the two
separate is deliberate: a report that could itself promote a run would be a
quality shortcut, not a check.

</div>

## Corpus hygiene report

`corpus_hygiene.json` records the measurable properties of the data the run drew
from:

<div className="qlm-grid">
<div><h4>Corpus source counts</h4><p>How many records came from each admitted source.</p></div>
<div><h4>Training text path and character count</h4><p>The exact text the run trained on, and its size.</p></div>
<div><h4>Training-example source mixture</h4><p>The proportion of examples drawn from each source family.</p></div>
<div><h4>Duplicate training examples</h4><p>Repeated training examples that could inflate apparent learning.</p></div>
<div><h4>Duplicate admission and eval ids</h4><p>Repeated identifiers across admitted and eval records.</p></div>
<div><h4>Train/eval prompt overlap</h4><p>Eval prompts that also appear in training — the contamination check.</p></div>
<div><h4>Protected heldout prompt overlap</h4><p>Overlap against the protected heldout set that may never be trained on.</p></div>
<div><h4>Candidate-example ratio</h4><p>The share of examples that originate from generated candidates.</p></div>
<div><h4>Rare-profile coverage</h4><p>Whether low-frequency answer profiles are represented at all.</p></div>
</div>

The report does not promote or reject a model by itself. It makes data risk
visible before promotion gates or transformer screens interpret metrics. A high
train/eval overlap, for example, does not fail the run on its own — it tells the
verifier and the reader why an exact-answer count should be distrusted.

## Training plan

`training_plan.json` is the run's declared scope. Where the hygiene report
describes the data, the plan states what the run is permitted to do with it,
inside the closed-world boundary.

<div className="qlm-grid">
<div><h4>Component and run id</h4><p>Which component is training, and the run it belongs to.</p></div>
<div><h4>Allowed data sources</h4><p>The admitted sources the run may draw from, stated up front.</p></div>
<div><h4>Closed-world data boundary</h4><p>The flags asserting no external weights, tokenizer, embeddings, or text.</p></div>
<div><h4>Hygiene report path</h4><p>The <code>corpus_hygiene.json</code> this plan was built against.</p></div>
<div><h4>Eval-set counts</h4><p>The size of each eval set the run will be scored on.</p></div>
<div><h4>Base and scheduled example mixture</h4><p>The intended source mixture, before and after scheduling.</p></div>
<div><h4>Candidate policy status</h4><p>Whether generated candidates are excluded from training.</p></div>
<div><h4>Training recipe path and summary</h4><p>The reproducible <a href="../training-recipes/">recipe</a>, when written.</p></div>
<div><h4>Replay-plan path and summary</h4><p>The profile-aware replay plan, when one is written.</p></div>
<div><h4>Closed-world verifier path and summary</h4><p>The <a href="../closed-world-verifier/">verifier</a> approval, when written.</p></div>
<div><h4>Planned artifacts</h4><p>The evidence files the run commits to emitting.</p></div>
</div>

<div className="qlm-keypoint">

**The candidate ratio is the load-bearing field**

Generated or proposed examples are reported here, but reporting them is not the
same as admitting them: a candidate cannot become training data without a later
admission to the ledger and a verification path. The plan records the lane; the
[candidate quarantine](./candidate-quarantine.md) enforces it.

</div>

## How the surfaces grew, and what they prove

The hygiene and training-plan artifacts were extended across many versions, but
the additions follow one shape: each new surface attaches more evidence to the
same plan without renaming the existing artifacts or letting any of them double
as a promotion shortcut.

| Version | Addition |
| --- | --- |
| v0.73 | Hygiene report and training plan; candidate ratio reported. |
| v0.75 | `candidate_quarantine.json`, linked from the plan, so generated examples cannot enter training without admission and verification. |
| v0.76 | `closed_world_verifier.json`, so the plan can be approved or rejected before its evidence influences the next version. |
| v0.77 | `training_recipe.json`, so the plan links a recipe that can reconstruct the run. |
| v0.78–v0.80 | Transformer responsibility, model/config, checkpoint, and eval surfaces that consume the same plan, recipe, and verifier without changing their names. |

From v0.81 onward, successive versions used these surfaces to attempt
profile-targeted and baseline-floor repairs of the transformer's
target-routing problem. The decisive outcome is consistent: the same artifact
trail kept rejecting promotion whenever a trained snapshot lost target-token
coverage or missed the baseline floor — including every `200/200` retry, which
is a memory-served count, not learned weights. Only small, guarded
source-profile updates that preserved the baseline floor (first one
`bridge:owner` update, then several profile-scale updates) were allowed to
become trusted model state.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

A `200/200` retry is a memory-served count, not learned weights. The corpus-hygiene
trail demonstrates, run after run, that broader coverage or a better-looking
number cannot become trusted model state by itself — only a guarded update that
keeps the data boundary and the baseline floor intact can.

</div>

This is the point of the report. The corpus-hygiene trail is part of the
evidence bundle, not a quality promotion shortcut. The from-scratch transformer
remains unpromoted on `branch_diversity_target`; the hygiene trail is part of
why that claim is honest.

:::note
See [Transformer](../build/transformer.md) for the routing problem itself, and
[Build](../build/index.md) for the `memory-served` versus `weight-consolidated`
distinction these artifacts protect.
:::

## What is next

<div className="qlm-next">

<a href="../closed-world-verifier/"><strong>Read next</strong><span>Closed-world verifier</span><small>The deterministic gate that decides whether a plan may influence the next step.</small></a>

<a href="../candidate-quarantine/"><strong>Read</strong><span>Candidate quarantine</span><small>How generated candidates are kept out of training until admitted.</small></a>

<a href="../../build/transformer/"><strong>Concept</strong><span>Transformer</span><small>The target-routing problem the hygiene trail keeps unpromoted.</small></a>

</div>

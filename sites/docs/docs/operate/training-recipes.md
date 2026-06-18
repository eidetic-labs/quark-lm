---
title: Training Recipes
description: Reproducible recipes and constraint-first promotion reports for QuarkLM runs.
---

# Training Recipes

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- The two required artifacts every self-improvement answer cycle and transformer answer-training run must emit.
- What the recipe records, and why it never names an external weight, tokenizer, embedding, or dataset.
- How constraint-first promotion orders boundary constraints ahead of quality checks.
- Why the transformer is currently rejected at this gate, and what is blocking it.

</div>

v0.77 adds `src/training_recipe.py` and two required artifacts that every
self-improvement answer cycle and transformer answer-training run must emit:

<div className="qlm-grid">
<div><h4>training_recipe.json</h4><p>Records how to reproduce a run.</p></div>
<div><h4>constraint_first_promotion.json</h4><p>Records whether a run is even allowed to let quality metrics affect promotion.</p></div>
</div>

The two split a single question into its two honest halves. The recipe answers
"what was run, and how would it be run again." The constraint-first report
answers "did the run stay inside the closed-world boundary before any score was
allowed to count."

<div className="qlm-keypoint">

**Neither file promotes a run.**

They make the run auditable against its own declared plan rather than a
description reconstructed afterward. One preserves reproducibility, the other
gates promotion.

</div>

These artifacts sit downstream of the [experiment
registry](./experiment-registry.md), which opens a run's intent, and downstream
of the [closed-world verifier](./closed-world-verifier.md), which approves the
training plan. The recipe and the constraint-first report close the loop: one
preserves reproducibility, the other gates promotion.

## The recipe artifact

`training_recipe.json` captures the run's configuration so a later screen can be
reconstructed from the artifact and the admitted project state, not from hidden
argparse memory.

<div className="qlm-grid">
<div><h4>recipe_id / version</h4><p>The recipe identity and the QuarkLM version it belongs to.</p></div>
<div><h4>component / run_id</h4><p>The component under test and the run the recipe describes.</p></div>
<div><h4>model</h4><p>Model configuration for the screen.</p></div>
<div><h4>tokenizer</h4><p>Tokenizer provenance — corpus-trained, never pretrained.</p></div>
<div><h4>data</h4><p>The admitted data sources the run is permitted to draw from.</p></div>
<div><h4>objective</h4><p>The training objective and its settings.</p></div>
<div><h4>optimizer</h4><p>Optimizer settings.</p></div>
<div><h4>replay</h4><p>Replay status, replay-plan reference, and replay-mixture report summary.</p></div>
<div><h4>artifacts</h4><p>The evidence files the run commits to emitting.</p></div>
<div><h4>gates</h4><p>The required gates the run must clear.</p></div>
<div><h4>rerun</h4><p>The rerun surface — enough to reproduce the run.</p></div>
</div>

The artifact also carries `uses_external_model: false` alongside the same
no-pretrained-weights, no-pretrained-tokenizer, and no-external-embeddings
posture enforced everywhere else. A recipe that named an external weight,
tokenizer, embedding, or dataset would cross the [purity
boundary](../secure/purity-boundary.md); recipes are reproduction records for
closed-world runs only.

Transformer recipes also point to two control artifacts. `sweep_plan.json`
records the tokenizer, transformer profile, context, width, heads, layers,
optimizer, and step-budget axes for the screen. `replay_mixture_report.json`
records whether the run exposed new lessons, retained facts, glossary/self
facts, unknown-policy probes, tokenizer stress strings, and heldout/paraphrase
evidence. Answer-training screens now record tokenizer type and manifest hash
from the governed tokenizer path, so char and closed-world subword trials are
comparable without importing a pretrained vocabulary. Together these artifacts
replace "turning knobs" with a comparable trial record.

`answer-sweep` collects multiple declared `answer-train` trials under one run
root and writes `sweep_report.json`. The sweep report summarizes the axis values,
trial directories, tokenizer evidence, and constraint-first status for each
trial; each trial still keeps its own `sweep_plan.json`, verifier, recipe, and
promotion report.

## Constraint-first promotion

`constraint_first_promotion.json` separates closed-world *constraints* from
quality *checks*, and enforces the order between them. Constraints are
pass/fail facts about the data boundary. Quality checks are scores. The report
will not let a score count until every required constraint has passed.

```text title="Constraint-first promotion order"
constraints (boundary facts)  ->  must all pass
        |
        v  (only then)
quality checks (scores)        ->  may affect promotion
        |
        v
status: blocked_before_quality_metrics  |  considered
```

For a transformer answer-training run the constraints are:

<div className="qlm-grid">
<div><h4>baseline_snapshot_recorded / final_snapshot_recorded</h4><p>The run captured comparable before/after evidence.</p></div>
<div><h4>closed_world_training_data</h4><p>Training drew only from admitted sources.</p></div>
<div><h4>closed_world_verifier</h4><p>The deterministic <a href="../closed-world-verifier/">verifier</a> approved the plan.</p></div>
<div><h4>controlled_sweep_plan</h4><p>The run declared its comparison axes before quality metrics were interpreted.</p></div>
<div><h4>replay_mixture_report</h4><p>The run declared its new, retained, unknown-policy, tokenizer-stress, and heldout/paraphrase evidence mixture.</p></div>
<div><h4>no_pretrained_weights</h4><p>No imported weights.</p></div>
<div><h4>no_pretrained_tokenizer</h4><p>No imported tokenizer.</p></div>
<div><h4>no_external_embeddings</h4><p>No imported embeddings.</p></div>
<div><h4>direct_answer_evidence_present</h4><p>The run recorded direct-answer evidence to judge.</p></div>
<div><h4>branch_context_gate</h4><p>Branch-context coverage holds.</p></div>
<div><h4>branch_diversity_target</h4><p>Predictions route across distinct branch tokens.</p></div>
<div><h4>target_coverage_preserved</h4><p>Trained snapshots did not lose target-token coverage.</p></div>
</div>

Only after those constraints pass can the exact direct-answer quality check
(`direct_greedy_exact`) affect promotion. Loss, NLL, rank, and top-k movement
stay advisory throughout: the report states plainly that these metrics are
advisory until all closed-world constraints pass. When a required constraint
fails, the report status is `blocked_before_quality_metrics` and
`quality_metrics_considered` is false — the scores are recorded, but they are
not allowed to influence the decision.

<div className="qlm-keypoint">

**A run can produce better numbers and still be rejected.**

This is the mechanism behind the constraint-first rule stated on
[Operate](./index.md). A good score behind a failed constraint is not evidence
of promotion.

</div>

## Current status

Self-improvement answer cycles promote through their existing exact-eval
discipline, now with a constraint-first report attached to each attempt.

Transformer answer-training runs now have a promotion gate, and it is rejecting.
Recent screens block on `branch_diversity_target`: the constraints report marks
that constraint failed, leaves quality metrics unconsidered, and closes with
status `blocked_before_quality_metrics`. The transformer is therefore not
promoted, and the docs say so — see [Transformer](../build/transformer.md) for
why branch diversity is the blocker and how it differs from the retrieval
memory that answers admitted probes exactly.

<div className="qlm-keypoint">

**Exact retrieval is `memory-served`, not `weight-consolidated`.**

The gate does not confuse the two. Retrieval answering an admitted probe proves
the corpus contains the answer; it does not prove the weights learned it.

</div>

<div className="qlm-next">
<a href="../closed-world-verifier/"><strong>Read next</strong><span>Closed-world verifier</span><small>The deterministic gate that approves the training plan upstream of this report.</small></a>
<a href="../../build/transformer/"><strong>Read next</strong><span>The transformer</span><small>Why branch diversity is the blocker, and why it differs from memory-served retrieval.</small></a>
<a href="../experiment-registry/"><strong>Reference</strong><span>Experiment registry</span><small>Where a run's intent is opened, upstream of the recipe.</small></a>
</div>

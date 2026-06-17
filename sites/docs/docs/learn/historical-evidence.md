---
title: Historical Evidence Archive
description: Earlier QuarkLM run evidence moved out of GOAL.md.
---

# Historical Evidence Archive

<p className="qlm-meta"><span>8 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why QuarkLM keeps failed and superseded runs as versioned evidence instead of discarding them
- How the responder / learned-component track stays separate from the transformer screen track
- What the early learned-component, self-improvement, and transformer phases actually proved
- Where current release evidence lives, and the rule that keeps archival detail from drifting back into `GOAL.md`

</div>

This page preserves older run evidence that used to live in `GOAL.md`. It is a
record, not a status report. `GOAL.md` is now the durable goal contract; current
status, release posture, and the latest transformer evidence live in
[Current evidence](./current-evidence.mdx), `STATUS.md`, and
`sites/shared/current-state.json`.

The archive exists so that evidence is never lost when it leaves the
current-state surfaces. QuarkLM keeps failed and superseded runs as versioned
diagnostic evidence rather than discarding them, so the chain of what was tried,
what passed, and what was rejected stays inspectable. Moving an entry here is how
that history is retained without letting old numbers drift back into the goal
contract or the README.

## How to read this archive

The same distinction enforced everywhere in QuarkLM applies to every entry
below. Two tracks ran in parallel, and the archive keeps them apart.

<div className="qlm-keypoint">

**A passing probe proves the corpus, not the weights**

A run that answers a probe correctly proves the corpus *contains* the answer; it
does not prove the transformer *learned* it. Responder-track results are
`memory-served` or learned-classifier evidence; only raw greedy transformer
answers would count toward neural promotion.

</div>

<div className="qlm-grid">
<div><h4>Responder / learned-component track</h4><p>The deterministic responder, retrieval memory, the learned answer model, and the answer decoder serve admitted knowledge. Exact results here are <code>memory-served</code> or learned-classifier evidence, not from-scratch transformer promotion.</p></div>
<div><h4>Transformer screen track</h4><p>The from-scratch decoder-only transformer is the <code>weight-consolidation</code> path. Its screens stayed separate from promoted responder evidence and are still blocked on <code>branch_diversity_target</code>.</p></div>
</div>

See [Language model](./language-model.md) for the three evidence states and
[Build](../build/index.md) for the two paths that produce them.

## Early learned components

These runs built and tightened the responder track's learned components before
the transformer architecture work began. They move classifier and decoder
exactness, not from-scratch transformer weights.

<div className="qlm-grid">
<div><h4><code>runs/context64-v0.2/</code></h4><p>Character model validation NLL moved <code>3.4968 → 2.6545</code>; known QA target NLL moved <code>3.4979 → 2.4155</code>; held-out target NLL moved <code>3.4978 → 2.5788</code>; free-form exact remained <code>0</code>.</p></div>
<div><h4><code>runs/answer-v0.1/</code></h4><p>Learned answer model moved QA, unknown, held-out, and paraphrase exactness from weak baselines to full exactness before stricter unseen-paraphrase tightening.</p></div>
<div><h4><code>runs/answer-v0.2/</code></h4><p>Learned answer model passed stricter unseen paraphrase probes: QA <code>8/8</code>, unknown <code>4/4</code>, held-out <code>8/8</code>, paraphrase <code>8/8</code>.</p></div>
<div><h4><code>runs/decoder-v0.2/</code></h4><p>Generative answer decoder moved from <code>0/8</code>, <code>0/4</code>, <code>0/8</code>, <code>0/8</code> exactness to QA <code>8/8</code>, unknown <code>4/4</code>, held-out <code>8/8</code>, paraphrase <code>8/8</code>.</p></div>
</div>

## Early self-improvement runs

These runs built the admission, audit, and self-diagnosis discipline that the
loop still depends on. They show the corpus growing through ledgered admissions
and the probes being generated from those admitted sources, so a passing probe is
evidence the corpus can answer it rather than a hand-written test.

<div className="qlm-grid">
<div><h4><code>self-improve-v0.9/</code></h4><p>Stricter lesson split kept held-out facts out of exact held-out prompt training; prompt leakage audit passed; answer model and decoder passed QA, unknown, held-out, and paraphrase evals.</p></div>
<div><h4><code>self-improve-v0.12/</code></h4><p>Added operational self and learning-admission concepts plus the first admitted memory event; answer model and decoder passed owner, self, learning, and admissions evals.</p></div>
<div><h4><code>self-improve-v0.14/</code></h4><p>Expanded admitted memory log to two facts; admission probes expanded to <code>8</code>; forgetting and prompt leakage audits passed.</p></div>
<div><h4><code>self-improve-v0.16/</code></h4><p>Moved provenance code into <code>provenance</code>; wrote corpus snapshots and diffs; forgetting and leakage audits passed.</p></div>
<div><h4><code>self-improve-v0.17/</code></h4><p>Generated admission probes from <code>corpus/admissions.jsonl</code>; probe sync passed with zero missing, extra, or mismatched ids.</p></div>
<div><h4><code>self-improve-v0.18/</code></h4><p>Renamed the product to QuarkLM, added <code>quark-lm-*</code> script aliases, and generated admission paraphrase probes.</p></div>
<div><h4><code>self-improve-v0.19/</code></h4><p>Added glossary word <code>stone</code> and admitted <code>learned-ivy-stone</code>; direct probes reached <code>12/12</code>, paraphrase probes <code>21/21</code>, and bridge lessons protected held-out transfer.</p></div>
<div><h4><code>self-improve-v0.20/</code></h4><p>Generated glossary probes from <code>corpus/glossary.json</code>; glossary probes passed <code>20/20</code>; exact eval audit and promotion gate passed.</p></div>
<div><h4><code>self-improve-v0.21/</code></h4><p>Added glossary words <code>shell</code>, <code>coin</code>, and <code>drum</code>; admitted three new memories; direct probes reached <code>24/24</code>, paraphrase probes <code>42/42</code>, glossary probes <code>26/26</code>; rule-based self-diagnosis reported <code>uses_external_model: false</code>.</p></div>
<div><h4><code>self-improve-v0.22/</code></h4><p>Expanded operational self facts and learning rules, added self-diagnosis corpus facts, and exposed the need to preserve failed-attempt evidence.</p></div>
<div><h4><code>self-improve-v0.23/</code></h4><p>Attempt archives became part of the loop so failed gates remain preserved instead of being overwritten by repair attempts.</p></div>
<div><h4><code>self-improve-v0.24/</code></h4><p>First transformer architecture work was kept separate from promoted responder evidence.</p></div>
<div><h4><code>self-improve-v0.25/</code> – <code>v0.42/</code></h4><p>Continued the promoted responder track while transformer screens stayed separate until neural promotion gates mature. Current promoted responder evidence remains <code>runs/self-improve-v0.42/</code>.</p></div>
</div>

:::note
The run paths above are abbreviated; each lives under `runs/`. The complete
ledgered admission and probe history is regenerated from the corpus files, not
hand-maintained.
:::

## Transformer evidence index

The transformer run history is now documented primarily in
[Transformer](../build/transformer.md), [Provenance](../operate/provenance.md),
and [Current evidence](./current-evidence.mdx). This index keeps the major phases
the old `GOAL.md` evidence section recorded, as a map into that detail.

<div className="qlm-keypoint">

**None of these phases cleared the branch-diversity gate**

Candidate-selector and generator results are auxiliary evidence. Raw greedy
transformer answers are the only signal that would count toward neural
promotion, and across every phase below they stayed weak.

</div>

<ol className="qlm-steps">
<li><strong>Architecture start</strong><p>Runs <code>transformer-v0.24/</code>, <code>transformer-v0.25/</code>: tiny decoder-only transformer from random weights using the corpus-trained character tokenizer.</p></li>
<li><strong>Answer training start</strong><p>Runs <code>transformer-answer-v0.26/</code>, <code>transformer-answer-v0.27/</code>: first transformer answer-training and faster eval-scoped candidate evaluator.</p></li>
<li><strong>Choice / selector path</strong><p>Runs <code>transformer-answer-v0.28-choice-prefix-pilot/</code>, <code>v0.29-selector-fast/</code>, <code>v0.30-selector-emission/</code>: candidate-selector evidence improved answer selection while raw greedy generation stayed weak.</p></li>
<li><strong>Generator path</strong><p>Run <code>transformer-answer-v0.31-generator-weighted-lr035-80k/</code>: no-candidate auxiliary generator moved exact generation from <code>0/219 → 219/219</code>; this remains generator evidence, not transformer greedy promotion.</p></li>
<li><strong>Direct-answer repair</strong><p>Runs <code>transformer-answer-v0.32-direct-base-context32/</code> through <code>v0.42-branch-repair-contrast50-dim8-context32/</code>: direct-answer modes improved distributional metrics and candidate behavior but did not make raw greedy transformer answers reliable.</p></li>
<li><strong>Branch diagnostics</strong><p>Runs <code>transformer-answer-v0.43-branch-profile-smoke-dim4-context16/</code> through <code>v0.43-branch-diversity-target-smoke-dim4-context80/</code>: branch profiles, context coverage, and branch-diversity targets exposed prompt-independent first-token collapse.</p></li>
<li><strong>Representation screens</strong><p>Runs <code>transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/</code> through <code>v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/</code>: context summaries, projections, prompt attention, prompt-position projections, and representation contrast moved measured surfaces but did not pass branch diversity.</p></li>
<li><strong>Structure audit and pre-layer norm</strong><p><code>STRUCTURE_AUDIT.md</code>, runs <code>transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/</code> and <code>v0.44-target-balanced-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/</code>: open-source structure was studied as reference only; pre-layer-norm partially cracked non-QA collapse but remained rejected because formal branch-diversity gates failed.</p></li>
</ol>

The generator reaching `219/219` while the direct transformer stayed at `0/219`
is the archived form of the distinction the project still holds today: the system
could already *serve* every answer while the neural weights had not yet *learned*
to route them.

```text title="The archived distinction in one line"
generator exact: 219/219   ->   could SERVE every answer
direct transformer: 0/219  ->   weights had not yet LEARNED to route them
```

The complete version-by-version log continues in
[Transformer screen history](../build/transformer-screen-history.md).

## Archive rule

<div className="qlm-keypoint">

**Archival detail belongs here, not in the goal contract**

Historical evidence should not drift back into `GOAL.md` or the README.
Version-specific detail belongs on this page only when it is archival context.

</div>

Current release evidence belongs in [Current evidence](./current-evidence.mdx),
the shared current state, and the relevant Build or Operate docs.

<div className="qlm-next">
<a href="../current-evidence/"><strong>Read next</strong><span>Current evidence</span><small>The live release posture and latest transformer screens.</small></a>
<a href="../../build/transformer-screen-history/"><strong>Go deeper</strong><span>Transformer screen history</span><small>The complete version-by-version transformer log.</small></a>
<a href="../language-model/"><strong>Reference</strong><span>Language model</span><small>The three evidence states and the two paths that produce them.</small></a>
</div>

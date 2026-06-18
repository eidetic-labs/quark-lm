---
title: Operate
description: Promote QuarkLM releases with evidence.
slug: /operate/
---

# Operate

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- When a run becomes a release, and why completion alone never earns it
- The constraint-first rule that blocks quality metrics until the boundary holds
- The evidence bundle every promoted run carries, and the controls that gate it
- Why docs are part of the promotion gate, not a byproduct of it

</div>

Operating QuarkLM means deciding when a run becomes a release and proving that
the decision was earned. A run is not promoted because it completed. It is
promoted only when it preserves the closed-world boundary, passes the recorded
gates, and leaves behind machine-checkable evidence — and when the docs that
describe current state move with it.

The pages under Operate cover three jobs: keeping every run's evidence
auditable, deciding what is eligible to influence the next weight update, and
keeping the public surfaces honest about what is and is not learned.

## The discipline in one rule

Promotion is constraint-first. Closed-world constraints — data boundary,
candidate exclusion, quarantine validity, branch coverage, branch diversity,
target-coverage preservation — must pass *before* any loss, NLL, rank, or
exact-quality number is allowed to count toward promotion. Quality metrics stay
advisory until the constraint report says they are eligible.

<div className="qlm-keypoint">

**Better numbers do not earn promotion**

This is why a run can produce better numbers and still be rejected, and why the
transformer is not promoted while `branch_diversity_target` fails. Constraints
are checked first; quality is allowed to count only after they pass.

</div>

Two distinctions are enforced across every page here.

<div className="qlm-grid">
<div><h4>A candidate is not training data</h4><p>A generated lesson, probe, repair, or memory proposal is not training data until it is admitted to <code>corpus/ledger.json</code>. Candidate records carry history; the candidate store is not a training source.</p></div>
<div><h4>Memory-served is not weight-consolidated</h4><p>Retrieval answering a probe is <code>memory-served</code>, not <code>weight-consolidated</code>. Exact retrieval proves the corpus <em>contains</em> the answer; it does not prove the transformer <em>learned</em> it. See <a href="../build/">Build</a>.</p></div>
</div>

## Operating surfaces

Each promoted run carries a fixed bundle of evidence artifacts. These are JSON
written during the run, validated deterministically, with no hidden promotion
behavior.

<div className="qlm-grid">
<div><h4><code>experiment_intent.json</code></h4><p>Hypothesis, allowed data, planned artifacts, acceptance gates, failure criteria, and the closing decision for a run.</p></div>
<div><h4><code>corpus_hygiene.json</code></h4><p>Source mixture, duplicates, train/eval overlap, candidate ratio, and rare-profile coverage.</p></div>
<div><h4><code>training_plan.json</code></h4><p>Allowed data sources, the closed-world data boundary, scheduled example mixture, replay summary, and planned artifacts.</p></div>
<div><h4><code>candidate_quarantine.json</code></h4><p>Candidate lifecycle state and proof that generated candidates are not training data until admitted.</p></div>
<div><h4><code>closed_world_verifier.json</code></h4><p>Deterministic pass/fail evidence for candidate checks and training-plan approval.</p></div>
<div><h4><code>training_recipe.json</code></h4><p>Reproducible model, tokenizer, data, objective, optimizer, artifact, and gate recipe.</p></div>
<div><h4><code>tokenizer_manifest.json</code></h4><p>Corpus hash, source files, append-only vocabulary, merge rules, and accepted or rejected tokenizer candidates.</p></div>
<div><h4><code>constraint_first_promotion.json</code></h4><p>The gate that blocks quality metrics until closed-world constraints pass.</p></div>
<div><h4><code>corpus_snapshot.json</code></h4><p>Current ledger source hashes and record counts.</p></div>
<div><h4><code>corpus_diff.json</code></h4><p>Comparison to the previous promoted run.</p></div>
</div>

Alongside the per-run artifacts sit the control documents that govern whether a
run is allowed to become a tag, and the hosting controls that keep the public
surfaces in sync.

<div className="qlm-grid">
<div><h4>Promotion controls</h4><p><code>ALPHA_GATE.md</code> / <code>RC_SPEC.md</code> / <code>RC_GAP_AUDIT.md</code> / <code>RC_CHECKLIST.md</code> — alpha gate, release-candidate track, gap, checklist, and forbidden-claim controls.</p></div>
<div><h4>Hosting controls</h4><p><code>.readthedocs.yaml</code> / <code>sites/DEPLOYMENT.md</code> — docs-on-Read-the-Docs and marketing-on-GitHub-Pages hosting controls.</p></div>
<div><h4>Public state</h4><p>README / Docusaurus / marketing — public state that must not drift.</p></div>
</div>

:::note

Run directories are versioned and immutable: `runs/self-improve-*` holds
promoted responder evidence, and unpromoted transformer screens keep their
original names so provenance stays exact. Failed runs are not discarded; they
remain as versioned diagnostic evidence.

:::

## Docs are a promotion gate

Docs are part of the loop, not a byproduct of it. The release is not complete
until the docs are complete: if a page references current eval counts, commands,
run ids, hosting targets, or roadmap commitments, that page must move with the
version.

<div className="qlm-keypoint">

**Anti-drift is a discipline, not a training input**

Promoted-version facts are read from the shared
`sites/shared/current-state.json` source wherever possible, so README,
Docusaurus, and marketing cannot quietly disagree. The docs move with the
version; they do not feed the model.

</div>

## Where to read next

<div className="qlm-grid">
<div><h4><a href="./release-candidate/">Release readiness</a></h4><p>The alpha and release-candidate tracks, the current decision, required commands and artifacts, and the claims that are forbidden until the evidence supports them.</p></div>
<div><h4><a href="./release-discipline/">Release discipline</a></h4><p>The full checklist a promoted run must satisfy, and SemVer release-identifier rules.</p></div>
<div><h4><a href="./experiment-registry/">Experiment registry</a></h4><p>Why every run starts with an explicit intent instead of a loose command.</p></div>
<div><h4><a href="./corpus-hygiene/">Corpus hygiene</a></h4><p>How data risk is made visible before any metric is interpreted.</p></div>
<div><h4><a href="./tokenizer-manifests/">Tokenizer manifests</a></h4><p>How corpus-only vocabulary changes are recorded before they influence a run.</p></div>
<div><h4><a href="./candidate-quarantine/">Candidate quarantine</a></h4><p>The candidate lifecycle and the rule that keeps generated material out of training until admitted.</p></div>
<div><h4><a href="./closed-world-verifier/">Closed-world verifier</a></h4><p>The deterministic check that decides whether a candidate or plan may influence the next learning step.</p></div>
<div><h4><a href="./training-recipes/">Training recipes</a></h4><p>Reproducible recipes and the constraint-first promotion report.</p></div>
<div><h4><a href="./provenance/">Provenance</a></h4><p>Corpus snapshots and diffs, recorded next to weight and eval changes.</p></div>
<div><h4><a href="./docs-drift/">Docs drift</a></h4><p>The rule that keeps docs and marketing synchronized with releases.</p></div>
</div>

<div className="qlm-next">
<a href="./release-candidate/"><strong>Read next</strong><span>Release readiness</span><small>The current decision, required artifacts, and forbidden claims.</small></a>
<a href="./candidate-quarantine/"><strong>Go deeper</strong><span>Candidate quarantine</span><small>The rule that keeps generated material out of training until admitted.</small></a>
<a href="./closed-world-verifier/"><strong>The deciding check</strong><span>Closed-world verifier</span><small>What may influence the next learning step, decided deterministically.</small></a>
</div>

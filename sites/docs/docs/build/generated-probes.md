---
title: Generated Probes
description: Probes generated from admitted memory and glossary sources.
---

# Generated Probes

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why evaluation probes are generated from admitted sources instead of hand-written.
- Which generators run, what each one reads, and what it writes.
- How to regenerate the eval files and check them against the corpus.
- How the resulting audits gate promotion.

</div>

Evaluation probes that check admitted knowledge are not written by hand. They are
generated from the admitted sources themselves, so the evals cannot drift away
from the facts the corpus actually ledgers. Two generators do this work:

<div className="qlm-grid">
<div><h4>admission_probes</h4><p>Builds answer probes from <code>corpus/admissions.jsonl</code>, the admitted-memory log.</p></div>
<div><h4>glossary_probes</h4><p>Builds definition probes from the <code>probe_words</code> list in <code>corpus/glossary.json</code>, the admitted glossary.</p></div>
</div>

<div className="qlm-keypoint">

**A probe is a question, not training data**

A probe is an evaluation question, not training data. These files exercise both
rails — retrieval memory and the trained weights — against the same admitted
facts. Admitting a fact through the [admission workflow](./admission-workflow.md)
is what changes what is learnable; generating probes only changes what is
measured.

</div>

## Why generate instead of hand-write

Hand-written evals rot. A fact is admitted or edited in `corpus/`, the eval that
checks it is forgotten, and the suite slowly stops describing the corpus it
claims to test. Generating probes from the admitted source removes that gap by
construction: the expected probe set is a deterministic function of the admitted
facts, so any divergence is a bug to be fixed rather than a judgement call.

This is the same anti-drift discipline the docs apply to prose. The generated
files are an artifact of the admitted corpus, not an independent input to it.

## What gets generated

Each generator reads its admitted source and writes a fixed expansion of probes
per source record.

<div className="qlm-grid">
<div><h4>evals/admissions.jsonl</h4><p>Generated from <code>corpus/admissions.jsonl</code>. Place, color, owner, and training-data-status probes per admitted fact.</p></div>
<div><h4>evals/admission_paraphrases.jsonl</h4><p>Generated from <code>corpus/admissions.jsonl</code>. Alternate surface forms of the same admitted facts.</p></div>
<div><h4>evals/glossary.jsonl</h4><p>Generated from <code>corpus/glossary.json</code> probe words. A "what does X mean" probe and a "define X" probe per word.</p></div>
</div>

The training-data-status probe is deliberate. For each admitted fact it asks
whether that fact is part of the training data and expects the answer `yes` —
because the fact was admitted to `corpus/ledger.json`. The paraphrase file widens
coverage to alternate phrasings of the admitted facts, so a passing screen
reflects the fact rather than one memorized prompt string.

## Current counts

Counts are derived from the current admitted sources, so they move only when the
corpus does.

<div className="qlm-grid">
<div><h4>Direct admission probes</h4><p><code>48</code> probes from <code>12</code> admitted facts.</p></div>
<div><h4>Admission paraphrase probes</h4><p><code>84</code> probes from <code>12</code> admitted facts.</p></div>
<div><h4>Glossary probes</h4><p><code>38</code> probes from <code>19</code> glossary probe words.</p></div>
</div>

Four direct probes per admitted fact and two probes per glossary probe word
account for these totals. Adding a fact or a probe word raises the counts on the
next sync; nothing else does.

## Check that the evals match the corpus

Both generators run in two modes. With no flag they regenerate the eval files
from the admitted sources. With `--check` they regenerate the expected probe set
in memory, compare it against the files on disk by probe id, and report any
`missing`, `extra`, or `mismatched` ids without writing.

<ol className="qlm-steps">
<li><strong>Regenerate the eval files</strong><p>Rebuild both eval files from the admitted sources.</p></li>
<li><strong>Verify without writing</strong><p>Compare the expected probe set against the checked-in files by probe id.</p></li>
</ol>

```bash title="Regenerate the eval files from admitted sources"
PYTHONPATH=src python3 -m admission_probes
PYTHONPATH=src python3 -m glossary_probes
```

```bash title="Verify the checked-in files still match the corpus (no writes)"
PYTHONPATH=src python3 -m admission_probes --check
PYTHONPATH=src python3 -m glossary_probes --check
```

A `--check` run exits non-zero when the files have drifted from the admitted
sources, which makes it usable as a guard. The
[admit](./admission-workflow.md) command keeps the direct and paraphrase
admission probes in sync automatically when it writes to the default
`corpus/admissions.jsonl`, unless `--no-sync-probes` is passed; `--check` then
confirms that sync after the fact rather than trusting it.

:::tip
Wire the `--check` runs into CI. Because the run exits non-zero on any drift,
it rejects a corpus edit that left the generated evals stale before the change
can land.
:::

## How the audits gate promotion

The [self-improvement report](../learn/self-improvement-loop.md) carries two
audits built from these probes: a combined `admission_probe_audit` over the
direct and paraphrase results, and a separate `glossary_probe_audit`. Each audit
reports whether its generated files still match the admitted sources, and both
must pass before a run can be promoted. They sit alongside the
[closed-world verifier](../operate/closed-world-verifier.md) among the
constraint-first promotion checks, so a screen whose evals have drifted from the
corpus is rejected before any quality number is allowed to count.

<div className="qlm-keypoint">

**Generated probes are evaluation material, not training input**

Generated probes are derived from admitted corpus files and checked into the
repo so evals stay honest, but they are not training data unless explicitly
admitted to the ledger.

</div>

## What's next

<div className="qlm-next">
<a href="./admission-workflow.md"><strong>Read next</strong><span>The admission workflow</span><small>How admitting a fact changes what is learnable.</small></a>
<a href="../operate/closed-world-verifier.md"><strong>Reference</strong><span>The closed-world verifier</span><small>The other constraint-first promotion check.</small></a>
<a href="../learn/self-improvement-loop.md"><strong>Read</strong><span>The self-improvement loop</span><small>Where the probe audits gate a run's promotion.</small></a>
</div>

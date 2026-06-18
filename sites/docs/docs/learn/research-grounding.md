---
title: Research Grounding
description: Paper-grounded guidance for QuarkLM's closed-world self-improvement loop.
---

# Research Grounding

<p className="qlm-meta"><span>12 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Which published research informs QuarkLM's design, and why none of it enters the model as weights, tokenizers, embeddings, datasets, or output.
- How the closest published pattern — a closed-loop self-improvement lifecycle — is constrained at every stage by the closed-world boundary.
- Why `memory-served` is not `weight-consolidated`, and why the transformer stays unpromoted while `branch_diversity_target` fails.
- The design rules, controls, and adopt/defer decisions the research implies, plus the novelty boundary QuarkLM is allowed to claim.

</div>

This page maps the published research that informs QuarkLM's design and records
the rules that research implies. It is a guidance document, not a dependency
list. None of the cited work enters QuarkLM as weights, tokenizers, embeddings,
datasets, or model output; the citations shape decisions only.

QuarkLM is closest to continual learning and lifelong pretraining research, but
it draws a narrower boundary than most published systems. The model starts from
random weights, uses a tokenizer trained only on admitted text, and treats "I
learned something new" as a corpus admission event followed by a versioned,
rejectable training run.

<div className="qlm-keypoint">

**The closed-world boundary is the one constant**

No pretrained weights, pretrained tokenizers, external embeddings, unledgered
datasets, or external model outputs may enter training. Every entry below is the
research read through that boundary.

</div>

For where that boundary is enforced in code, see
[Purity boundary](../secure/purity-boundary.md).

## The closest published pattern

The nearest published shape is a closed-loop self-improvement lifecycle:
acquisition, selection, optimization, inference refinement, and autonomous
evaluation. QuarkLM adopts that loop shape but constrains every stage with the
closed-world boundary.

<div className="qlm-grid">
<div><h4>Acquisition</h4><p>Open-world: open web scraping or broad corpus ingestion. QuarkLM: admitted corpus events, ledgered with provenance.</p></div>
<div><h4>Selection</h4><p>Open-world: model taste or learned data filters. QuarkLM: deterministic provenance and quality gates.</p></div>
<div><h4>Optimization</h4><p>Open-world: unbounded weight updates toward a loss. QuarkLM: auditable, guarded weight updates from admitted curricula.</p></div>
<div><h4>Inference refinement</h4><p>Open-world: retrieval and tuning blended into one score. QuarkLM: retrieval and exact responders labeled separately from learned weights.</p></div>
<div><h4>Evaluation</h4><p>Open-world: aggregate benchmark movement. QuarkLM: promotion gates that can reject an apparently better run on retention, diversity, unknown-policy, or leakage evidence.</p></div>
</div>

In QuarkLM terms the operating sequence is:

```text title="QuarkLM operating sequence"
new lesson -> corpus -> retrieval memory -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

That sequence inverts the usual large-model starting point. Most large models
begin with broad knowledge already compressed into pretrained weights and then
add retrieval or tuning around that base. QuarkLM begins with an auditable
corpus boundary, lets retrieval memory serve admitted knowledge first, and
treats neural consolidation as a later, guarded, rejectable event. This is the
`memory-served` versus `weight-consolidated` distinction enforced across the
docs; see the three evidence states in
[Language model](./language-model.md).

The research does not make QuarkLM's thesis impossible. It makes the operating
discipline stricter: accumulated admitted data must never be replaced by
self-generated text, replay and retention gates must be first-class, and any
future self-judge must prove itself inside the closed world before it can admit
or grade lessons.

## From papers to mechanics

The paper map below is paired with a code-and-systems comparison layer. The
open-source mechanics audit studies public implementations as design references
only: nanoGPT, minGPT, LitGPT, Hugging Face tokenizers, Avalanche, LLM360, OLMo,
OLMo 2, Self-Instruct, STaR, Reflexion, InsCL, and deep generative replay. The
audit reads their structure, not their weights, tokenizers, or data. Its
governing decision is that QuarkLM's next transformer bottleneck is trainer
mechanics, not another global branch-loss term. The full gap matrix is in
[Open-source mechanics audit](./open-source-mechanics-audit.md).

That audit set the direction for the work that followed: build the
self-improvement operating system before adding more training objectives. The
operating-system surfaces that resulted are the experiment registry, replay
planning, corpus hygiene and training-plan artifacts, the research
implementation map, candidate quarantine, deterministic verifier checks, and
constraint-first promotion gates. Each surface has its own page under
[Operate](../operate/index.md), and the version that introduced it is recorded
in the experiment registry rather than narrated here.

The forward and deep-research reviews keep that map current. The forward plan
cross-references papers, public implementation mechanics, and the QuarkLM
codebase; the deep research review expands the literature, mechanics, and gap
review before each implementation step. See
[Forward research plan](./forward-research-plan.md) and
[Deep research review](./deep-research-review.md).

## The branch-diversity blocker

The transformer has a single open blocker, and it is the case that most directly
exercises these design rules. Direct-answer snapshots emit a
`branch_diversity_target` gate that fails when multi-target eval profiles
collapse to too few predicted branch tokens: the model learns to predict one
dominant token instead of routing each prompt to its own answer. From the v0.112
screens onward the failure is classified as a critical `target_routing_gap`,
with low representation separation across `9/9` multi-target profiles and
dominant-token wins traced to hidden-projection pressure.

Many guarded direct-answer objectives have been tried against this gate. They
have moved coverage and diagnostics forward without clearing it, and every
rejected snapshot was kept as versioned diagnostic evidence rather than
discarded. The latest candidate, the v0.115 hidden-projection-margin repair,
lowers the collapsed-token hidden advantage and is still rejected on
`branch_diversity_target`. The screen-by-screen evidence for this arc lives in
two archives: [Branch diversity research](./branch-diversity-research.md) for
the root-cause analysis and external citations, and
[Transformer screen history](../build/transformer-screen-history.md) for the
complete version-by-version log.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

The same screens that reject neural promotion also record that retrieval memory
answers `219/219` eval probes exactly, with provenance and no weight updates.
That is evidence for the memory rail, not for the weights — the system can serve
every admitted answer while the transformer has not yet learned to route them.
`memory-served` is not `weight-consolidated`, and the transformer is not
promoted while `branch_diversity_target` fails.

</div>

## Paper map

Each entry pairs a representative line of published work with the implication it
carries into QuarkLM. The work is read for structure and direction only; nothing
here enters training.

<div className="qlm-grid">
<div><h4>Self-improvement lifecycle</h4><p><a href="https://arxiv.org/abs/2603.25681">Self-Improvement of Large Language Models</a> — model self-improvement should be modeled as a loop with acquisition, selection, optimization, inference refinement, and evaluation layers. QuarkLM already has pieces of this loop; the next work is making the loop explicit and measurable.</p></div>
<div><h4>Continual learning for LLMs</h4><p><a href="https://arxiv.org/abs/2402.01364">Continual Learning for LLMs: A Survey</a> and <a href="https://arxiv.org/abs/2404.16789">A Comprehensive Survey</a> — treat self-improvement as staged learning: corpus learning, instruction-like answer behavior, and alignment or policy behavior need separate gates.</p></div>
<div><h4>Lifelong pretraining</h4><p><a href="https://arxiv.org/abs/2110.08534">Lifelong Pretraining</a> — evaluate both plasticity and retention whenever new corpus data is admitted. Every admission batch needs a forgetting audit, not just final accuracy.</p></div>
<div><h4>Catastrophic forgetting</h4><p><a href="https://arxiv.org/abs/1612.00796">Overcoming Catastrophic Forgetting in Neural Networks</a> — add explicit protection for prior behavior through replay, retention metrics, or weight-importance penalties before larger admission streams.</p></div>
<div><h4>Synaptic consolidation</h4><p><a href="https://arxiv.org/abs/1703.04200">Continual Learning Through Synaptic Intelligence</a> — weight-importance penalties are a possible later stabilizer, but replay and evaluation gates are easier to audit first in a tiny from-scratch codebase.</p></div>
<div><h4>Replay</h4><p><a href="https://arxiv.org/abs/1705.08690">Continual Learning with Deep Generative Replay</a> — replay should be explicit, stratified, and provenance-bound. Generated replay is only acceptable if it is reconstructable from admitted facts and verified before use.</p></div>
<div><h4>Language lifelong learning</h4><p><a href="https://arxiv.org/abs/1909.03329">LAMOL</a> — replay can be generated by a model, but QuarkLM must only train on replay that is derived from admitted data and verified against ledgered probes.</p></div>
<div><h4>External memory and retrieval</h4><p><a href="https://arxiv.org/abs/2005.11401">Retrieval-Augmented Generation</a> — keep corpus memory explicit and provenance-rich. Retrieval can be used as an auditable responder rail, but it is not the same as weight learning.</p></div>
<div><h4>Self-reflective retrieval</h4><p><a href="https://arxiv.org/abs/2310.11511">Self-RAG</a> — adaptive retrieve, generate, and critique behavior is relevant, but QuarkLM needs closed-world reflection tokens or verifier records before a self-critique can affect training.</p></div>
<div><h4>Reasoning self-improvement</h4><p><a href="https://arxiv.org/abs/2203.14465">STaR</a> — self-generated lessons can be candidate training material only after an objective verifier proves they are correct against admitted sources.</p></div>
<div><h4>Self-reward loops</h4><p><a href="https://arxiv.org/abs/2401.10020">Self-Rewarding Language Models</a> — a future QuarkLM judge must be trained and evaluated inside the closed world before it can grade candidate lessons or repairs.</p></div>
<div><h4>Synthetic recursion risk</h4><p><a href="https://arxiv.org/abs/2305.17493">The Curse of Recursion</a> and <a href="https://arxiv.org/abs/2404.01413">Is Model Collapse Inevitable?</a> — do not let model-generated material replace the original admitted corpus. Accumulate ledgered originals, label synthetic candidates, and preserve rare records through coverage-aware replay.</p></div>
<div><h4>Agentic skill libraries</h4><p><a href="https://arxiv.org/abs/2305.16291">Voyager</a> and <a href="https://arxiv.org/abs/2310.02304">STOP</a> — store improvements as auditable artifacts first. Do not treat scaffold changes, generated code, or self-reports as knowledge unless tests and admission rules accept them.</p></div>
<div><h4>Model editing</h4><p><a href="https://arxiv.org/abs/2104.08164">KnowledgeEditor</a> and <a href="https://arxiv.org/abs/2210.07229">MEMIT</a> — direct weight edits are useful reference points, but QuarkLM should prefer retrainable corpus-to-weights paths until edit locality and side effects are measurable.</p></div>
<div><h4>Small-data language learning</h4><p><a href="https://arxiv.org/abs/2305.07759">TinyStories</a> and <a href="https://arxiv.org/abs/2504.08165">BabyLM findings</a> — small models can learn useful language behavior from constrained, simple corpora, but QuarkLM should favor human-authored or admitted text over external-model synthetic stories.</p></div>
<div><h4>Data quality and mixtures</h4><p><a href="https://arxiv.org/abs/2107.06499">Deduplicating Training Data Makes Language Models Better</a> and <a href="https://arxiv.org/abs/2305.10429">DoReMi</a> — add corpus hygiene, duplicate checks, source balance, and domain/lesson mixture reporting before increasing corpus size.</p></div>
<div><h4>Transformer architecture</h4><p><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a> — the transformer is the right backbone for contextual token binding, but QuarkLM's implementation must stay from scratch and small enough to audit.</p></div>
<div><h4>Modern transformer mechanics</h4><p><a href="https://arxiv.org/abs/2104.09864">RoPE</a>, <a href="https://arxiv.org/abs/1910.07467">RMSNorm</a>, and <a href="https://arxiv.org/abs/2002.05202">GLU variants</a> — v0.51's rotary, RMSNorm, and gated-MLP options match mainstream mechanics while preserving from-scratch weights and dependency-free training.</p></div>
<div><h4>Tokenization</h4><p><a href="https://arxiv.org/abs/1508.07909">Subword Units</a> and <a href="https://arxiv.org/abs/1808.06226">SentencePiece</a> — corpus-derived subword tokenization is now an opt-in governed path. Character tokens remain the baseline; subword evidence must prove compression, round-trip safety, manifest purity, and no full-answer-token leakage before it informs model claims.</p></div>
<div><h4>Evaluation contamination</h4><p><a href="https://arxiv.org/abs/2406.04244">Benchmark Data Contamination Survey</a> and <a href="https://arxiv.org/abs/2311.06233">Data Contamination Quiz</a> — keep protected held-out prompts out of training, track generated probe lineage, and treat exact-match success as invalid if the prompt itself leaked into curriculum.</p></div>
</div>

## Best-practice control matrix

Each control pairs a documented failure mode in the literature with the policy
QuarkLM holds against it.

<div className="qlm-grid">
<div><h4>Ledgered admission</h4><p>Continual systems drift when new data is not versioned. Every learnable fact, lesson, probe, and generated candidate needs provenance, source type, and admission status.</p></div>
<div><h4>Accumulation over replacement</h4><p>Synthetic-recursion work shows collapse risk when generated data replaces original data. Original admitted data remains permanent unless explicitly deprecated by a ledger event; generated candidates can supplement only after verification.</p></div>
<div><h4>Replay and retention</h4><p>Continual-learning work treats forgetting as the central failure mode. Every weight update should mix new material with representative old material and report backward retention.</p></div>
<div><h4>Coverage-aware sampling</h4><p>Rare facts and minority targets are easy to lose. Sampling should report per-profile, per-target, and per-source coverage, not only average loss.</p></div>
<div><h4>Data hygiene</h4><p>Duplicates and repeated strings inflate memorization and distort evals. Add duplicate detection and source balance metrics before expanding the corpus materially.</p></div>
<div><h4>Separate memory rails</h4><p>RAG-style memory improves grounding but does not prove weights learned. Exact responder, retrieval, classifier, decoder, and transformer metrics stay separate in evidence reports.</p></div>
<div><h4>Closed-world verifier</h4><p>Self-training works only when incorrect generations are filtered. A generated lesson cannot become training data until a deterministic or internally validated verifier accepts it.</p></div>
<div><h4>Promotion gates</h4><p>Continual systems often improve one metric while regressing another. Promotion must require retention, unknown-policy, leakage, branch diversity, target coverage, and current-task evidence.</p></div>
</div>

## Current design rules

<ol className="qlm-steps">
<li><strong>Corpus and belief stay separate</strong><p>Corpus admission and model belief stay separate.</p></li>
<li><strong>Weights follow versioned curriculum</strong><p>Weight updates happen only after admitted data is converted into versioned curriculum.</p></li>
<li><strong>Self-generated text must pass the verifier</strong><p>Self-generated text can propose lessons, probes, or repairs, but cannot become training data until a deterministic verifier accepts it against the corpus.</p></li>
<li><strong>Preserve prior behavior every batch</strong><p>Every new training batch must preserve prior accepted behavior through forgetting checks, replay, or an explicit retention gate.</p></li>
<li><strong>Measure more than exact match</strong><p>Evaluation must measure more than final exact match: retention, unknown policy, prompt leakage, branch diversity, target coverage, and calibration need to be tracked as the model grows.</p></li>
<li><strong>Treat responders as grounding rails</strong><p>Retrieval and exact responders are grounding rails. They can explain and verify answers, but success there is not proof that the transformer weights learned the behavior.</p></li>
<li><strong>Defer direct weight editing</strong><p>Direct weight editing remains deferred until QuarkLM can measure locality, generalization to paraphrases, and damage to unrelated admitted facts.</p></li>
<li><strong>Label generated candidates as candidates</strong><p>Model-generated candidates must be labeled as candidates, kept out of the permanent curriculum by default, and promoted only through admission checks.</p></li>
<li><strong>Report coverage before scaling corpus</strong><p>Corpus growth should include duplicate, mixture, and rare-record coverage reports before the next larger training run.</p></li>
</ol>

## Research-informed architecture direction

QuarkLM should keep four lanes visible in reports.

<div className="qlm-grid">
<div><h4>Corpus lane</h4><p>Admitted originals, generated candidates, rejected candidates, duplicate checks, source balance, and curriculum mixtures.</p></div>
<div><h4>Memory lane</h4><p>Exact responder and retrieval-style rails that answer from explicit corpus artifacts without implying parametric learning.</p></div>
<div><h4>Weight lane</h4><p>Tokenizer, neural answer models, decoder, and transformer checkpoints trained only from admitted curricula.</p></div>
<div><h4>Evaluation lane</h4><p>Retention, leakage, unknown-policy, target coverage, diversity, calibration, and verifier-quality evidence.</p></div>
</div>

This framing lets QuarkLM improve continuously without confusing "I stored
something," "I retrieved something," "I generated a plausible candidate," and
"my weights learned something." Those are different claims, and the research
suggests that collapsing them into one score is where many self-improvement
loops become fragile.

## Adopt next

- Add replay as a first-class training primitive for admitted facts, glossary
  facts, self facts, learning rules, QA lessons, and transformer branch
  targets.
- Add a retention report that tracks old eval performance, new eval
  performance, backward transfer, and any tradeoff introduced by a repair.
- Add corpus hygiene reports: duplicate detection, source mixture counts,
  candidate/original ratios, and rare-record coverage.
- Teach the self-diagnosis layer to emit candidate lessons and candidate probes
  that are verified before admission.
- Continue the transformer repair under the profile-aware replay plan, treating
  the replay-plan artifact as a constraint: reject any snapshot that improves
  rank or loss while sacrificing per-profile target coverage.
- Keep branch-diversity and target-coverage gates in the transformer path. The
  current failure is collapse under weight updates, not lack of loss movement,
  so a falling loss is not by itself a reason to promote.
- Keep expanding the closed-world verifier lane. It is deterministic and
  rule-based today, and can become a from-scratch model only after its judgments
  can be audited against admitted sources.
- Keep routing repairs narrow. The `target_routing_gap` diagnosis points at
  hidden-projection and representation separation, so prefer a guarded
  representation candidate over a broad branch-loss knob, and scale a repair only
  after coverage and branch-diversity gates confirm it is not another collapse.

## Defer

- Self-rewarded training without an independent verifier.
- Model editing or patching individual associations directly in transformer
  weights.
- Promoting subword tokenization as model progress before tokenizer manifests,
  long-answer diagnostics, retention, and branch-diversity evidence support it.
- Any external-model judge for promotion decisions. External research can guide
  humans, but QuarkLM's own promotion artifacts must stay closed-world.
- Training on model-generated material that replaced, summarized, or diluted
  the original admitted records.
- Letting retrieval success count as transformer learning evidence.

## Novelty boundary

QuarkLM should not claim that continual learning, self-improvement, or
closed-domain models are new. The research gap it explores is narrower:

<div className="qlm-keypoint">

**The thesis is the combination, not any single piece**

- no pretrained model weights;
- no pretrained tokenizer;
- no external embeddings;
- an explicitly admitted corpus as the only training source;
- self-improvement reports that separate corpus changes, generated candidates,
  verifier decisions, weight updates, and forgetting audits;
- a long-term path toward self-improvement that does not require an external
  model to shape the model's future lessons.

</div>

That combination is the project thesis. The implementation should keep proving
it one admitted batch and one versioned run at a time.

:::note

This page is guidance, not a dependency list. The citations shape decisions; the
cited work never enters QuarkLM as weights, tokenizers, embeddings, datasets, or
model output.

:::

## What is next

<div className="qlm-next">
<a href="../language-model/"><strong>Read next</strong><span>The language model</span><small>The three evidence states behind the memory-served versus weight-consolidated distinction.</small></a>
<a href="../open-source-mechanics-audit/"><strong>Read</strong><span>Open-source mechanics audit</span><small>The code-and-systems comparison layer and the full gap matrix.</small></a>
<a href="../../secure/purity-boundary/"><strong>Secure</strong><span>Purity boundary</span><small>Where the closed-world boundary is enforced in code.</small></a>
</div>

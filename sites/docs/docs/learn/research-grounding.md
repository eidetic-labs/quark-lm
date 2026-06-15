---
title: Research Grounding
description: Paper-grounded guidance for QuarkLM's closed-world self-improvement loop.
---

# Research Grounding

Last reviewed: 2026-06-14.

QuarkLM is closest to continual learning and lifelong pretraining research, but
it intentionally uses a narrower boundary than most published systems. The
model starts from random weights, uses a tokenizer trained only on admitted
data, and treats "I learned something new" as a corpus admission event followed
by a versioned training run.

The research map below is not a dependency list. These papers guide design
choices while preserving QuarkLM's rule that no pretrained weights, pretrained
tokenizers, external embeddings, unledgered datasets, or external model outputs
may enter training.

## Executive Finding

The closest published pattern is a closed-loop self-improvement lifecycle:
data acquisition, data selection, model optimization, inference refinement, and
autonomous evaluation. QuarkLM should adopt that lifecycle shape, but every
stage must be constrained by the closed-world boundary:

- acquisition means admitted corpus events, not open web scraping;
- selection means deterministic provenance and quality gates, not model taste;
- optimization means auditable weight updates from admitted curricula;
- inference refinement means retrieval, exact responders, and generated answer
  rails that are labeled separately from learned parametric behavior;
- evaluation means promotion gates that can reject an apparently better run if
  retention, diversity, unknown-policy, or leakage evidence regresses.

The research does not make QuarkLM's thesis impossible. It makes the operating
discipline stricter: accumulated admitted data must never be replaced by
self-generated text, replay and retention gates need to be first-class, and any
future self-judge must prove itself inside the closed world before it can admit
or grade lessons.

## Mechanics Audit Addendum

The v0.66 open-source mechanics audit adds a code-and-systems comparison layer
on top of this paper map. It studies nanoGPT, minGPT, LitGPT, Hugging Face
tokenizers, Avalanche, LLM360, OLMo, OLMo 2, Self-Instruct, STaR, Reflexion,
InsCL, and deep generative replay as design references only.

The audit decision is that QuarkLM's next transformer bottleneck is trainer
mechanics, not another global branch-loss term. v0.67 implements the first
profile-aware replay-plan surface: branch replay can carry profile keys,
coverage deficits and represented-target preservation are computed per profile,
and profile-aware screens emit a replay-plan artifact before training. See
[Open-source mechanics audit](./open-source-mechanics-audit.md).

The v0.69 forward research plan extends that audit with a cross-reference
between papers, public implementation mechanics, and the current QuarkLM
codebase. The v0.70 deep research review then expands the literature,
open-source mechanics, and QuarkLM-codebase gap review before the next
implementation step. The decision is to build the self-improvement operating
system before more objective modes. v0.71 implements the experiment registry;
v0.72 extracts replay planning; v0.73 adds corpus hygiene and training-plan
artifacts; v0.74 adds the research implementation map; v0.75 adds candidate
quarantine; v0.76 adds deterministic verifier checks; v0.77 adds training
recipes and constraint-first promotion gates; v0.78 adds transformer
experiment/artifact surfaces, trainer utilities, and an objective catalog;
v0.79 adds transformer model/config and checkpoint metadata surfaces; v0.80
adds transformer eval/checkpoint-load surfaces; v0.81 adds balanced profile
target-share pressure inside the preserving-deficit direct-answer objective;
v0.82 screens that objective and rejects trained snapshots that collapse branch
diversity. v0.83 adds prompt-specific ownership margins and rejects the screen
because trained snapshots still lose target-token coverage. v0.84 adds baseline
replay anchors and rejects the screen because trained snapshots still preserve
only half of the baseline QA/heldout coverage floor. v0.85 adds a baseline-floor
update guard and rejects the screen because all attempted updates are unsafe
under that floor. v0.86 adds adaptive baseline-floor retries and rejects the
screen because all `200/200` retry attempts remain unsafe under the same floor.
v0.87 adds one bounded baseline-covered repair after each failed retry and
rejects the screen because all `200/200` repaired attempts remain unsafe under
the same floor. v0.88 adds balanced baseline-floor anchors inside the objective
and rejects the screen because all `200/200` objective-shaped attempts remain
unsafe under the same floor. v0.89 removes branch-diversity pressure and trains
only baseline-covered floor anchors, but all `200/200` stabilization-only
attempts remain unsafe under the same floor. Future objective repairs should
use those narrower surfaces and the v0.90 rejection diagnostics before
branch-diversity pressure is added back. v0.90 shows all `200` rejected attempts
are stabilization-shaped, `heldout` violates every attempt, and the worst floor
deficit is `0.25` on `learning`. v0.91 covers the full baseline-covered
profile-target floor surface and still rejects all `200/200` attempts, showing
the repair shape itself needed to change. v0.92 changes the shape to sequential
source-profile floor repair, rejects all `2000` profile-local attempts, and
shows the next repair should isolate floor-preserving weight movement. v0.93
adds calibrated scales below `0.01`, accepts one source-profile update at scale
`0.0025`, and shows the next repair should expand safe calibrated movement.
v0.94 adds profile-scale memory, accepts `8` source-profile updates, and shows
the next repair should make safe movement branch-diverse. v0.95 adds
diversity-aware profile-scale acceptance, accepts `5` score-improving
source-profile updates, rejects `11` floor-preserving score regressions, and
shows the next repair should convert non-regressive movement into full
branch-diversity target coverage. v0.96 adds frontier target anchors, accepts
`9` score-improving source-profile updates, lowers max dominant predicted rate
to `0.9`, and shows the next repair should convert frontier movement into full
branch-diversity target coverage. v0.97 adds coverage-frontier acceptance,
accepts `1` coverage-gaining source-profile update across `68` attempts, and
shows the next repair should keep the audit while isolating missing-target
repairs so later profiles are not starved. v0.98 adds coverage-prep frontier
acceptance, accepts `9` source-profile updates, separates `3` coverage gains
from `6` coverage-preparation moves, and sets up direct missing-target coverage
recovery. v0.99 adds
coverage-recovery frontier retry, accepts `6` source-profile updates, converts
`2` prepared candidates into direct coverage recoveries, keeps `4` preparation
fallbacks, and shows the next repair should stabilize branch diversity after
coverage recovery. v0.100.0 adds branch-stable coverage-recovery acceptance,
keeps the `2` recovery conversions, records `15` branch-stability checks,
rejects `1` retry for branch-score regression, and shows the next repair should
increase branch diversity without weakening the recovery floor. v0.101.0 adds
branch-diversity recovery after safe profile updates, accepts `5` local
branch-score refinements, falls back once, and shows the next repair should
turn local score gains into target-token coverage for the collapsed profiles.
v0.102.0 adds collapsed-profile binding, accepts `1` targeted binding update,
narrows final collapse from `9/9` eval profiles to `3/9`, and shows the next
repair should focus on `learning`, `owner`, and `paraphrases`.
v0.103.0 adds remaining-profile binding, records `21` prioritized attempts,
accepts `6` prioritized updates, improves `learning` coverage from `0.0` to
`0.25`, and shows the next repair should preserve that gain while targeting
`owner` and `paraphrases`.
v0.104.0 adds owner/paraphrase residual binding, records `16` prioritized
attempts, accepts `6` prioritized updates, rejects `24` preservation failures,
and shows the next repair should convert protected owner/paraphrase attempts
from ties into real target-token diversity.
v0.105.0 adds corpus-only retrieval memory, records `497` memory cards and
`219/219` exact retrieval evals with no external model, embeddings, pretrained
retriever, or weight updates, and shows that immediate memory serving should be
separated from slower neural consolidation.
v0.106.0 adds memory-guided consolidation planning, ranks `9` retrieval-served
neural failed profiles, and turns the memory/weight separation into an explicit
target list for the next gated consolidation training screen.
v0.107.0 implements that gated consolidation handoff: it consumes the v0.106.0
plan, targets `owner`, `paraphrases`, and `glossary`, records `26` prioritized
attempts with `8` acceptances and `18` rejections, and still rejects neural
promotion on `branch_diversity_target`.
v0.108.0 consumes the v0.107.0 plan, expands the target window to `owner`,
`paraphrases`, `heldout`, `qa`, and `glossary`, and shows that the next
mechanic needs missing first-token diversity pressure rather than another
target-window expansion.
See
[Forward research plan](./forward-research-plan.md) and
[Deep research review](./deep-research-review.md).

## Paper Map

| Area | Representative work | QuarkLM implication |
| --- | --- | --- |
| Self-improvement lifecycle | [Self-Improvement of Large Language Models](https://arxiv.org/abs/2603.25681) | Model self-improvement should be modeled as a loop with acquisition, selection, optimization, inference refinement, and evaluation layers. QuarkLM already has pieces of this loop; the next work is making the loop explicit and measurable. |
| Continual learning for LLMs | [Continual Learning for Large Language Models: A Survey](https://arxiv.org/abs/2402.01364) and [Continual Learning of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2404.16789) | Treat self-improvement as staged learning: corpus learning, instruction-like answer behavior, and alignment or policy behavior need separate gates. |
| Lifelong pretraining | [Lifelong Pretraining](https://arxiv.org/abs/2110.08534) | Evaluate both plasticity and retention whenever new corpus data is admitted. Every admission batch needs a forgetting audit, not just final accuracy. |
| Catastrophic forgetting | [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796) | Add explicit protection for prior behavior through replay, retention metrics, or weight-importance penalties before larger admission streams. |
| Synaptic consolidation | [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200) | Weight-importance penalties are a possible later stabilizer, but replay and evaluation gates are easier to audit first in a tiny from-scratch codebase. |
| Replay | [Continual Learning with Deep Generative Replay](https://arxiv.org/abs/1705.08690) | Replay should be explicit, stratified, and provenance-bound. Generated replay is only acceptable if it is reconstructable from admitted facts and verified before use. |
| Language lifelong learning | [LAMOL](https://arxiv.org/abs/1909.03329) | Replay can be generated by a model, but QuarkLM must only train on replay that is derived from admitted data and verified against ledgered probes. |
| External memory and retrieval | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | Keep corpus memory explicit and provenance-rich. Retrieval can be used as an auditable responder rail, but it is not the same as weight learning. |
| Self-reflective retrieval | [Self-RAG](https://arxiv.org/abs/2310.11511) | Adaptive retrieve, generate, and critique behavior is relevant, but QuarkLM needs closed-world reflection tokens or verifier records before a self-critique can affect training. |
| Reasoning self-improvement | [STaR](https://arxiv.org/abs/2203.14465) | Self-generated lessons can be candidate training material only after an objective verifier proves they are correct against admitted sources. |
| Self-reward loops | [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | A future QuarkLM judge must be trained and evaluated inside the closed world before it can grade candidate lessons or repairs. |
| Synthetic recursion risk | [The Curse of Recursion](https://arxiv.org/abs/2305.17493) and [Is Model Collapse Inevitable?](https://arxiv.org/abs/2404.01413) | Do not let model-generated material replace the original admitted corpus. Accumulate ledgered originals, label synthetic candidates, and preserve rare records through coverage-aware replay. |
| Agentic skill libraries | [Voyager](https://arxiv.org/abs/2305.16291) and [STOP](https://arxiv.org/abs/2310.02304) | Store improvements as auditable artifacts first. Do not treat scaffold changes, generated code, or self-reports as knowledge unless tests and admission rules accept them. |
| Model editing | [KnowledgeEditor](https://arxiv.org/abs/2104.08164) and [MEMIT](https://arxiv.org/abs/2210.07229) | Direct weight edits are useful reference points, but QuarkLM should prefer retrainable corpus-to-weights paths until edit locality and side effects are measurable. |
| Small-data language learning | [TinyStories](https://arxiv.org/abs/2305.07759) and [BabyLM findings](https://arxiv.org/abs/2504.08165) | Small models can learn useful language behavior from constrained, simple corpora, but QuarkLM should favor human-authored or admitted text over external-model synthetic stories. |
| Data quality and mixtures | [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) and [DoReMi](https://arxiv.org/abs/2305.10429) | Add corpus hygiene, duplicate checks, source balance, and domain/lesson mixture reporting before increasing corpus size. |
| Transformer architecture | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The transformer is the right backbone for contextual token binding, but QuarkLM's implementation must stay from scratch and small enough to audit. |
| Modern transformer mechanics | [RoPE](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), and [GLU variants](https://arxiv.org/abs/2002.05202) | v0.51's rotary, RMSNorm, and gated-MLP options match mainstream mechanics while preserving from-scratch weights and dependency-free training. |
| Tokenization | [Subword Units](https://arxiv.org/abs/1508.07909) and [SentencePiece](https://arxiv.org/abs/1808.06226) | A corpus-derived subword tokenizer is plausible later. Character tokens remain the safer baseline until evidence shows token length is the bottleneck. |
| Evaluation contamination | [Benchmark Data Contamination Survey](https://arxiv.org/abs/2406.04244) and [Data Contamination Quiz](https://arxiv.org/abs/2311.06233) | Keep protected held-out prompts out of training, track generated probe lineage, and treat exact-match success as invalid if the prompt itself leaked into curriculum. |

## Best-Practice Control Matrix

| Control | Why it matters | QuarkLM policy |
| --- | --- | --- |
| Ledgered admission | Continual systems drift when new data is not versioned. | Every learnable fact, lesson, probe, and generated candidate needs provenance, source type, and admission status. |
| Accumulation over replacement | Synthetic-recursion work shows collapse risk when generated data replaces original data. | Original admitted data remains permanent unless explicitly deprecated by a ledger event; generated candidates can supplement only after verification. |
| Replay and retention | Continual-learning work treats forgetting as the central failure mode. | Every weight update should mix new material with representative old material and report backward retention. |
| Coverage-aware sampling | Rare facts and minority targets are easy to lose. | Sampling should report per-profile, per-target, and per-source coverage, not only average loss. |
| Data hygiene | Duplicates and repeated strings inflate memorization and distort evals. | Add duplicate detection and source balance metrics before expanding the corpus materially. |
| Separate memory rails | RAG-style memory improves grounding but does not prove weights learned. | Exact responder, retrieval, classifier, decoder, and transformer metrics stay separate in evidence reports. |
| Closed-world verifier | Self-training works only when incorrect generations are filtered. | A generated lesson cannot become training data until a deterministic or internally validated verifier accepts it. |
| Promotion gates | Continual systems often improve one metric while regressing another. | Promotion must require retention, unknown-policy, leakage, branch diversity, target coverage, and current-task evidence. |

## Current Design Rules

1. Corpus admission and model belief stay separate.
2. Weight updates happen only after admitted data is converted into versioned
   curriculum.
3. Self-generated text can propose lessons, probes, or repairs, but cannot
   become training data until a deterministic verifier accepts it against the
   corpus.
4. Every new training batch must preserve prior accepted behavior through
   forgetting checks, replay, or an explicit retention gate.
5. Evaluation must measure more than final exact match: retention, unknown
   policy, prompt leakage, branch diversity, target coverage, and calibration
   need to be tracked as the model grows.
6. Retrieval and exact responders are grounding rails. They can explain and
   verify answers, but success there is not proof that the transformer weights
   learned the behavior.
7. Direct weight editing remains deferred until QuarkLM can measure locality,
   generalization to paraphrases, and damage to unrelated admitted facts.
8. Model-generated candidates must be labeled as candidates, kept out of the
   permanent curriculum by default, and promoted only through admission checks.
9. Corpus growth should include duplicate, mixture, and rare-record coverage
   reports before the next larger training run.

## Research-Informed Architecture Direction

QuarkLM should keep four lanes visible in reports:

1. **Corpus lane:** admitted originals, generated candidates, rejected
   candidates, duplicate checks, source balance, and curriculum mixtures.
2. **Memory lane:** exact responder and retrieval-style rails that answer from
   explicit corpus artifacts without implying parametric learning.
3. **Weight lane:** tokenizer, neural answer models, decoder, and transformer
   checkpoints trained only from admitted curricula.
4. **Evaluation lane:** retention, leakage, unknown-policy, target coverage,
   diversity, calibration, and verifier-quality evidence.

This framing lets QuarkLM improve continuously without confusing "I stored
something," "I retrieved something," "I generated a plausible candidate," and
"my weights learned something." Those are different claims, and the research
suggests that collapsing them into one score is where many self-improvement
loops become fragile.

## Adopt Next

- Add replay as a first-class training primitive for admitted facts, glossary
  facts, self facts, learning rules, QA lessons, and transformer branch
  targets.
- Add a retention report that tracks old eval performance, new eval
  performance, backward transfer, and any tradeoff introduced by a repair.
- Add corpus hygiene reports: duplicate detection, source mixture counts,
  candidate/original ratios, and rare-record coverage.
- Teach the self-diagnosis layer to emit candidate lessons and candidate probes
  that are verified before admission.
- Continue the transformer repair with anti-collapse preservation inside the
  profile-aware replay plan: v0.68 shows profile-local rank pressure can still
  sacrifice target-token coverage and branch diversity.
- Keep using the replay-plan artifact as a constraint and reject snapshots that
  improve rank while sacrificing per-profile target coverage.
- Keep branch-diversity and target-coverage gates in the transformer path,
  because the current failure is collapse under weight updates, not lack of
  loss movement.
- Keep expanding the closed-world verifier lane. It starts deterministic and
  rule-based in v0.76, and can become a from-scratch model only after its
  judgments can be audited against admitted sources.
- Continue the v0.70 sequence before adding another direct-answer objective
  mode: v0.71 implemented experiment registry and v0.72 extracted replay
  planning; v0.73 added corpus hygiene and training plans; v0.74 added the
  research implementation map; v0.75 added candidate quarantine; v0.76 added
  verifier checks; v0.77 added training recipes and constraint-first promotion;
  v0.78 added transformer experiment/artifact surfaces, trainer utilities, and
  an objective catalog; v0.79 added transformer model/config and checkpoint
  metadata surfaces; v0.80 added transformer eval/checkpoint-load surfaces;
  v0.81 added profile target-share anti-collapse pressure; v0.82 screened it
  and rejected it on branch diversity; v0.83 added prompt-specific ownership
  margins and rejected the screen because target-token coverage still collapses
  during training; v0.84 added baseline replay anchors and rejected the screen
  because trained snapshots preserve only `0.125` QA/heldout coverage against
  the `0.25` floor; v0.85 added a baseline-floor update guard and rejected the
  screen because it preserved the floor by rejecting `50/50` attempted updates;
  v0.86 added adaptive baseline-floor retries and rejected the screen because
  all `200/200` retry attempts remained unsafe; v0.87 added one bounded
  baseline-covered repair after each failed retry and rejected the screen
  because all `200/200` repaired attempts remained unsafe; v0.88 added
  objective-side baseline-floor anchors and rejected the screen because all
  `200/200` objective-shaped attempts remained unsafe; v0.89 added
  stabilization-only floor anchors and rejected the screen because all
  `200/200` stabilization-shaped attempts remained unsafe. The next objective
  repair should use the v0.90 profile-level floor diagnostics to target
  consistently violating profiles before branch-diversity pressure is added
  back; v0.91 shows full profile-target floor coverage alone is still
  insufficient; v0.92 shows sequential source-profile floor repair is still
  insufficient; v0.93 shows calibrated smaller update surfaces can survive the
  floor guard; v0.94 expands those surfaces to eight source profiles, so the
  next repair should turn them into diversity gains.

## Defer

- Self-rewarded training without an independent verifier.
- Model editing or patching individual associations directly in transformer
  weights.
- Replacing the character tokenizer with subwords before context length and
  prompt-binding evidence show that tokenization is the limiting factor.
- Any external-model judge for promotion decisions. External research can guide
  humans, but QuarkLM's own promotion artifacts must stay closed-world.
- Training on model-generated material that replaced, summarized, or diluted
  the original admitted records.
- Letting retrieval success count as transformer learning evidence.

## Novelty Boundary

QuarkLM should not claim that continual learning, self-improvement, or
closed-domain models are new. The research gap it explores is narrower:

- no pretrained model weights;
- no pretrained tokenizer;
- no external embeddings;
- an explicitly admitted corpus as the only training source;
- self-improvement reports that separate corpus changes, generated candidates,
  verifier decisions, weight updates, and forgetting audits;
- a long-term path toward self-improvement that does not require an external
  model to shape the model's future lessons.

That combination is the project thesis. The implementation should keep proving
it one admitted batch and one versioned run at a time.

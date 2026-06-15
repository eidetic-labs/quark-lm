# QuarkLM Forward Research Plan

Last updated: 2026-06-14.

## Purpose

v0.68 showed that QuarkLM can improve target-rank evidence while still erasing
profile coverage and branch diversity. That is not a small hyperparameter
problem. It is a process problem: the trainer can currently optimize a local
objective before the project has enough explicit machinery for experiment
intent, replay planning, verifier acceptance, corpus hygiene, and promotion
constraints.

This plan turns the next phase into a research-backed implementation sequence.
It combines published work, public open-source mechanics, and the current
QuarkLM codebase state. The goal is to keep building toward a closed-world
language model that trains only from QuarkLM's admitted dataset, improves its
weights through versioned runs, and eventually proposes improvements without an
external model shaping the training data.

## Research Method

The review used two kinds of sources:

1. Primary papers for the research claims.
2. Public project repositories and documentation for implementation mechanics.

The open-source projects are design references only. QuarkLM must not copy
their code, import their datasets, reuse pretrained weights, reuse pretrained
tokenizers, or use external model outputs as hidden training data.

## Research Synthesis

### Continual Learning

Continual learning surveys frame the central problem as plasticity without
forgetting. The 2024 LLM continual-learning survey categorizes updates across
continual pretraining, instruction tuning, and alignment, and highlights
benchmarks and evaluation as unresolved challenges
([Wu et al. 2024](https://arxiv.org/abs/2402.01364)). A newer 2026 survey
emphasizes rehearsal, regularization, architecture-based methods, forgetting
rates, and knowledge transfer metrics
([Chen et al. 2026](https://arxiv.org/abs/2603.12658)).

QuarkLM implication: replay and forgetting metrics are not optional side
reports. Every weight update should know which admitted profiles it is allowed
to improve, which prior profiles it must preserve, and which gates can reject
the update.

### Replay And Retention

Experience replay remains one of the most practical continual-learning
controls. Deep Generative Replay separates the solver from a generator of prior
experience ([Shin et al. 2017](https://arxiv.org/abs/1705.08690)).
Avalanche implements continual learning as streams, strategies, plugins,
buffers, and evaluation callbacks. Its docs show replay as a plugin that
changes the training dataloader from current examples alone to current examples
plus memory-buffer examples, with reservoir sampling and evaluation over the
test stream ([Avalanche training docs](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training)).
Reverb makes the same point for RL systems: replay is an explicit data system
with insert and sample policies, not a private detail inside the loss
([Cassirer et al. 2021](https://arxiv.org/abs/2102.04736)).

QuarkLM implication: `direct_answer_replay_plan.json` is the right direction,
but replay planning should be extracted from `transformer_char_model.py` into a
standalone planner that can be tested, reused, and recorded before training.

### Self-Generated Data

Self-Instruct uses model generations to propose new instruction data, then
filters invalid or similar examples before fine-tuning
([Wang et al. 2022](https://arxiv.org/abs/2212.10560)). The public repo's
pipeline generates instructions, classifies task type, generates instances, and
then filters/processes/reformats them
([self-instruct repo](https://github.com/yizhongw/self-instruct)). Its README
also warns that a sampled quality analysis found many generated data points had
problems, which is exactly the risk QuarkLM must avoid.

STaR uses self-generated rationales only when the resulting answer is correct,
then fine-tunes and repeats ([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)).
Self-Refine uses a generator, feedback, and refiner loop at inference time
without weight updates ([Madaan et al. 2023](https://arxiv.org/abs/2303.17651);
[self-refine repo](https://github.com/madaan/self-refine)). Reflexion stores
trial feedback as linguistic memory rather than updating weights
([Shinn et al. 2023](https://arxiv.org/abs/2303.11366)).

QuarkLM implication: generated text must enter a candidate quarantine lane, not
the training corpus. "I learned something new" should mean a candidate was
proposed, verified against admitted sources, admitted with provenance, converted
to curriculum, trained into weights, and evaluated.

### Self-Judging Risk

Self-feedback can work, but it is fragile. A self-bias study found models tend
to favor their own outputs, and self-refinement can amplify that bias
([Xu et al. 2024](https://arxiv.org/abs/2402.11436)). Reward-hacking work shows
that a model evaluator and generator can drift apart from human judgment when
the evaluator is an imperfect proxy
([Pan et al. 2024](https://arxiv.org/abs/2407.04549)). The Self-Feedback
survey separates self-evaluation from self-update and asks when self-feedback
actually works ([Liang et al. 2024](https://arxiv.org/abs/2407.14507)).

QuarkLM implication: a future learned verifier is valuable, but it cannot be
the first gate. Start with deterministic closed-world verification and only
later train a verifier from admitted artifacts, accepted/rejected candidate
history, and run outcomes.

### Verifiable Rewards

Modern post-training increasingly uses verifiable rewards when the task has an
objective checker. Tulu 3 releases data, code, recipes, evaluations, and
decontamination alongside SFT, DPO, and reinforcement learning with verifiable
rewards ([Lambert et al. 2024](https://arxiv.org/abs/2411.15124);
[open-instruct repo](https://github.com/allenai/open-instruct)). DeepSeek-R1
shows that rule-based rewards can elicit reasoning behavior, but also that
multi-stage training and cold-start data were needed to address readability and
language-mixing failures
([DeepSeek-AI 2025](https://arxiv.org/abs/2501.12948)).

QuarkLM implication: our first verifier should be small and objective:
ledger membership, exact answer consistency, unknown-policy compliance, prompt
leakage absence, profile coverage, and branch diversity. We should not jump to
open-ended self-rewarding.

### Model Collapse

The model-collapse literature is directly relevant to a self-improving model.
Training recursively on model-generated data can remove tail information
([Shumailov et al. 2023](https://arxiv.org/abs/2305.17493)). A later study
argues collapse can be avoided when real and synthetic data accumulate instead
of synthetic data replacing real data
([Gerstgrasser et al. 2024](https://arxiv.org/abs/2404.01413)).

QuarkLM implication: original admitted records are permanent evidence. Generated
candidates can supplement after verification, but they must remain labeled by
origin, and replay must protect rare profiles.

### Small And Data-Centric Models

BabyLM findings show that small models can be trained more sample-efficiently,
but many curriculum-learning attempts were weak or only modestly useful
([Warstadt et al. 2025](https://arxiv.org/abs/2504.08165)). TinyStories shows
that constrained language distributions can teach small models coherent English,
but its dataset was generated by external models and therefore is a reference,
not a QuarkLM training source
([Eldan and Li 2023](https://arxiv.org/abs/2305.07759)). SmolLM2 is important
because it is explicitly data-centric: it uses staged mixtures, small-scale
ablations, manual mixture refinements, and releases prepared datasets
([Allal et al. 2025](https://arxiv.org/abs/2502.02737)).

QuarkLM implication: corpus quality and mixture planning matter more than
blindly increasing model complexity. Every training run should have a declared
mixture plan and an acceptance criterion.

### Transparent Open Models

LLM360 argues that open research needs more than final weights: code, data,
checkpoints, intermediate results, and analyses should be released together
([Liu et al. 2023](https://arxiv.org/abs/2312.06550)). OLMo releases training
data, training/evaluation code, model weights, checkpoints, and logs
([Groeneveld et al. 2024](https://arxiv.org/abs/2402.00838)). OLMo 2 goes
further with training recipes, logs, thousands of intermediate checkpoints, and
stage-specific data mixtures ([Team OLMo 2025](https://arxiv.org/abs/2501.00656)).
The OLMo README publishes stage-one and stage-two token counts, configs, seeds,
checkpoint links, and WandB links
([OLMo repo](https://github.com/allenai/OLMo)).

QuarkLM implication: rejected runs, configs, replay plans, and docs are not
overhead. They are the evidence trail required for the project thesis.

### Data Hygiene And Contamination

Deduplication can reduce memorized outputs and train-test overlap while
requiring fewer training steps for similar or better accuracy
([Lee et al. 2021](https://arxiv.org/abs/2107.06499)). Dolma exposes dataset
curation tooling, taggers, and fast deduplication as a first-class part of the
training stack ([Dolma repo](https://github.com/allenai/dolma)). Open-Instruct
ships decontamination scripts that index training prompts, query eval sets, and
produce contamination reports
([open-instruct decontamination](https://github.com/allenai/open-instruct/blob/main/decontamination/README.md)).
LatestEval reinforces that evaluation construction must actively avoid overlap
with training data ([Li et al. 2023](https://arxiv.org/abs/2312.12343)).

QuarkLM implication: corpus hygiene should become a mandatory artifact:
duplicates, near-duplicates, source mixtures, generated-candidate ratios,
protected-prompt overlap, and rare-profile coverage.

### Trainer And Recipe Boundaries

nanoGPT intentionally keeps a small model file and a small training loop; it
records config, optimizer state, best validation loss, and checkpoints in the
training script ([nanoGPT repo](https://github.com/karpathy/nanoGPT)). minGPT
separates a generic `Trainer` from model and dataset concerns
([minGPT repo](https://github.com/karpathy/minGPT)). GPT-NeoX exposes training,
evaluation, and generation entry points through YAML configs
([GPT-NeoX repo](https://github.com/EleutherAI/gpt-neox)). LLM Foundry uses
source modules, data-prep scripts, train/eval/inference scripts, YAML workflows,
and registries for components
([LLM Foundry repo](https://github.com/mosaicml/llm-foundry)). LitGPT presents
pretrain, continued pretrain, finetune, evaluate, deploy, and test as distinct
workflows ([LitGPT repo](https://github.com/Lightning-AI/litgpt)).

QuarkLM implication: `src/closed_world_lm/transformer_char_model.py` is now too
large for the next phase. It has model code, optimizer code, direct-answer
objectives, replay planning, snapshots, CLI parsing, checkpoint writing, and
run reporting in one file. The next implementation should extract recipe,
replay, evaluation, and artifact boundaries before adding more objective modes.

## Current QuarkLM Diagnosis

### Strengths

- The project already preserves the closed-world boundary: no pretrained
  weights, no pretrained tokenizer, no external embeddings, and ledgered
  admissions.
- The self-improvement report already includes prompt leakage, forgetting,
  exact eval, promotion gate, corpus snapshots, corpus diffs, attempts archive,
  and deterministic diagnosis.
- v0.51 added a serious transformer mechanics stack: optimizer state, AdamW,
  scheduling, accumulation, multi-head attention, RMSNorm, gated MLPs, tied
  embeddings, rotary positions, prompt-position projection, checkpoint metadata,
  and eval traces.
- v0.67 added profile-aware replay-plan artifacts and tests.
- v0.68 proved the gate can reject rank improvements that damage coverage and
  diversity.

### Gaps

- `transformer_char_model.py` is a 9,494-line monolith. It is difficult to
  reason about new training modes because model, recipe, replay, eval,
  checkpoint, and CLI behavior are interleaved.
- v0.71 now records hypothesis, acceptance criteria, planned artifacts, and
  promotion decision before training begins.
- v0.72 now extracts replay planning from transformer objective plumbing into a
  reusable artifact module.
- v0.73 now writes corpus hygiene and training-plan artifacts for the current
  self-improvement and transformer run paths.
- v0.75 adds candidate quarantine artifacts and lifecycle states.
- v0.76 adds a deterministic closed-world verifier interface for candidate
  checks and training-plan approval.
- Snapshot scoring is stronger than before, but transformer promotion remains
  less integrated than the self-improvement promotion gate.
- The character tokenizer is still the purity baseline, but any future subword
  tokenizer needs an artifacted manifest, offsets, special-token policy, and
  admitted-corpus-only proof.

## Implementation Blueprint

### 1. Experiment Registry

Add an experiment-intent artifact before any new run:

- version;
- hypothesis;
- allowed data sources;
- planned artifacts;
- training recipe id;
- replay plan id;
- acceptance gates;
- failure criteria;
- result decision.

This turns each run from "try a mode" into an auditable experiment.

### 2. Corpus Governance Layer

Promote corpus state from implicit input to explicit training contract:

- admission ledger summary;
- original versus generated-candidate counts;
- duplicate and near-duplicate report;
- source/profile mixture;
- protected-prompt overlap;
- rare-profile coverage;
- tokenizer vocabulary proof.

### 3. Candidate Quarantine Layer

Generated lessons, generated probes, repair notes, and self-diagnosis proposals
should be written as candidates first. Candidate state should be one of:

- proposed;
- rejected;
- needs human review;
- verifier accepted;
- admitted;
- trained;
- promoted.

No candidate trains weights until it is admitted.

### 4. Closed-World Verifier Lane

Start deterministic:

- verify that candidate facts are entailed by admitted records;
- verify that answers match exact responder evidence;
- verify unknown-policy compliance;
- verify no protected eval prompt leaked into training text;
- verify profile coverage and replay preservation.

Later, train a small verifier from QuarkLM's admitted candidate history and run
outcomes. That learned verifier must remain advisory until it passes its own
held-out verifier evals.

### 5. Replay Planner

Extract replay from the transformer monolith into a reusable planner:

- input: examples, profiles, current snapshot, coverage floors, run objective;
- output: `replay_plan.json`;
- required fields: profile counts, target sets, represented targets, deficits,
  replay ratios, rare-profile floors, synthetic/original labels, and rejection
  constraints.

The trainer should consume the replay plan, not recreate it privately.

### 6. Training Recipe Layer

Add recipe objects that select:

- model family;
- tokenizer artifact;
- curriculum source;
- replay plan;
- objective;
- optimizer;
- snapshot cadence;
- promotion gates.

Recipes should be small and named. A recipe can fail, but it should be
reproducible.

### 7. Evaluation And Promotion Layer

Promotion should be constraint first:

- no prompt leakage;
- no unknown-policy regression;
- no forgetting regression;
- no protected held-out contamination;
- no profile coverage regression;
- no branch diversity regression;
- no rank-only promotion;
- no loss-only promotion.

Only after those constraints pass should rank, top-k, NLL, or loss be used to
compare snapshots.

### 8. Refactor Boundary

Keep the public CLI stable, but carve out narrow modules:

- `transformer_model.py` for model/config/checkpoint format;
- `transformer_training.py` for generic training loops;
- `transformer_replay.py` for replay records and plans;
- `transformer_eval.py` for branch profiles and promotion scoring;
- `experiment_registry.py` for run intent and decision artifacts;
- `candidate_quarantine.py` for candidate quarantine;
- `closed_world_verifier.py` for deterministic verifier checks.

The exact module names can change, but those responsibilities should not keep
living inside one file.

## Cross-Referenced Implementation Matrix

| External pattern | What others do | QuarkLM implementation |
| --- | --- | --- |
| OLMo and LLM360 transparency | Release data, recipes, logs, checkpoints, and intermediate results. | Treat every run artifact, rejected screen, replay plan, and docs update as part of the research claim. |
| Pythia learning dynamics | Same data order and frequent checkpoints make training dynamics analyzable. | Add experiment registry and deterministic data ordering for comparable screens. |
| Avalanche replay | Strategies and plugins modify dataloaders with memory buffers and evaluate after each experience. | Make replay a standalone planner with profile-aware memory, rare-profile floors, and retention gates. |
| Open-Instruct decontamination | Index train data and query eval data for overlap reports. | Add corpus hygiene reports for protected prompts and train/eval overlap. |
| Self-Instruct and STaR | Generate candidates, filter them, and train only accepted examples. | Add candidate quarantine plus deterministic verifier before admission. |
| SCoRe and self-bias work | Self-generated correction can collapse or amplify bias without good rewards. | Do not train on self-judged text until verifier quality is measured. |
| Tulu 3 and DeepSeek-R1 | Verifiable rewards work where correctness can be checked. | Use ledger, exact responder, unknown-policy, leakage, coverage, and diversity as first verifiable rewards. |
| SmolLM2 | Data mixtures are refined with ablations and stage evidence. | Add training-plan mixture reports and small ablation screens before bigger runs. |
| Hugging Face tokenizers | Tokenizers are trained artifacts with normalization, offsets, special tokens, padding, truncation, and serialization. | Keep character tokenizer now; require manifest and admitted-corpus-only proof before subword work. |
| nanoGPT, minGPT, GPT-NeoX, LLM Foundry, LitGPT | Keep model, trainer, config, recipes, eval, and data prep separable. | Refactor transformer recipe/replay/eval/artifact code out of the monolith before adding new modes. |

## Next Versions

### v0.69

Record this forward research plan and docs update. No model-quality claim.

### v0.70

Record the deeper cross-referenced research review in `DEEP_RESEARCH_REVIEW.md`
and the Docusaurus Learn section. No model-quality claim. This supersedes the
earlier v0.70 implementation target because the project needed a deeper
literature, implementation, and codebase gap review before writing more
mechanics.

### v0.71

Implemented experiment registry and run-intent schemas in
`src/closed_world_lm/experiment_registry.py`. Self-improvement answer cycles
and transformer answer-training runs now record hypothesis, allowed data,
planned artifacts, gates, failure criteria, and a result decision. Transformer
screens close through constraint-first promotion reports from v0.77 onward.

### v0.72

Implemented replay planning extraction in `src/closed_world_lm/replay_plan.py`
with focused tests. The existing profile-aware replay plan is behaviorally
preserved while replay records, profile grouping, coverage floors, and missing
target summaries now live outside the transformer monolith.

### v0.73

Implemented corpus hygiene and training-plan artifacts in
`src/closed_world_lm/corpus_hygiene.py`. Self-improvement and transformer
answer-training runs now write source mixture, duplicate, train/eval overlap,
candidate-ratio, rare-profile coverage, allowed-data, planned-artifact, and
replay-plan summary evidence.

### v0.74

Added `RESEARCH_IMPLEMENTATION_MAP.md` and the matching Docusaurus Learn page.
This research-control checkpoint cross-references papers, open-source
mechanics, and QuarkLM's current codebase into a source-to-gap-to-version
implementation map. No model-quality claim.

### v0.75

Implemented candidate quarantine artifacts in
`src/closed_world_lm/candidate_quarantine.py`. Self-improvement and transformer
answer-training runs now write `candidate_quarantine.json`, and training plans
record the quarantine path and summary. Candidates are not training data.

### v0.76

Implemented deterministic closed-world verifier checks in
`src/closed_world_lm/closed_world_verifier.py`. Self-improvement and
transformer answer-training runs now write `closed_world_verifier.json`, and
training plans embed verifier summaries. Verifier evidence is deterministic,
external-model-free, and focused on candidate checks plus training-plan
approval.

### v0.77

Implemented recipe objects and constraint-first promotion gates in
`src/closed_world_lm/training_recipe.py`. Self-improvement and transformer
answer-training runs now write `training_recipe.json` and
`constraint_first_promotion.json`, and transformer decisions cannot promote
from loss, rank, top-k, or NLL movement unless constraints pass first.

### v0.78+

Only after the above should QuarkLM add a new anti-collapse transformer
objective, revisit tokenizer growth, or begin a learned verifier experiment.

## Stop Doing For Now

- Do not add another direct-answer mode as the first move.
- Do not run larger screens without an experiment intent artifact.
- Do not train on self-generated text without quarantine and verification.
- Do not treat retrieval, exact response, or candidate generation as proof of
  learned parametric behavior.
- Do not promote rank or loss improvements that reduce profile coverage,
  branch diversity, unknown-policy behavior, or retention.
- Do not replace admitted originals with summaries.
- Do not move to subword tokenization before the tokenizer artifact policy is
  written and tested.

## Decision

The next implementation phase should build the self-improvement operating
system before more objective tuning. QuarkLM needs experiment registry,
corpus-governance reports, candidate quarantine, verifier checks, replay
extraction, training recipes, and constraint-first promotion gates. That is the
structured path toward a model that can eventually say, truthfully and
audibly: "I learned something new, and now it is part of my training data."

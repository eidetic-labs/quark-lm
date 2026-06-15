# QuarkLM Deep Research Review

Last updated: 2026-06-14.

## Purpose

This review is the deeper cross-reference pass requested after v0.69. The
earlier forward plan identified the right direction, but it was still a
strategy document. v0.70 makes the research obligation explicit: before another
full-stack transformer repair run, QuarkLM must compare the current codebase
against primary research and public implementation patterns, then implement the
missing operating system around learning.

The conclusion is direct: QuarkLM should not keep twisting objective knobs
until experiment intent, replay, corpus hygiene, candidate quarantine,
verification, recipes, and constraint-first promotion are first-class code
paths.

## Review Boundary

Allowed:

- Read papers and official open-source project docs for architecture,
  workflow, artifact, evaluation, and training-system patterns.
- Translate lessons into QuarkLM-native code that preserves the closed-world
  training boundary.
- Use external sources to decide what is missing from QuarkLM's mechanics.

Not allowed:

- Copy outside implementation code into QuarkLM.
- Import pretrained weights, pretrained tokenizers, external embeddings,
  external datasets, or model-generated text into training.
- Use an external model as a hidden teacher, judge, verifier, reward model, or
  repair generator.
- Count retrieval, exact response rails, or generated candidates as parametric
  weight learning.

## Source Tiers

| Tier | Sources | How the tier informs QuarkLM |
| --- | --- | --- |
| Primary research | Continual learning, replay, self-generated data, self-feedback, verifiable rewards, model collapse, data hygiene, tokenization, scaling, and transparent-model papers. | Establishes risks, constraints, and mechanisms that should become QuarkLM artifacts. |
| Official implementation references | nanoGPT, minGPT, GPT-NeoX, LLM Foundry, LitGPT, Avalanche, Hugging Face tokenizers, Open-Instruct, Dolma, OLMo, Pythia, Self-Instruct, and Self-Refine repositories or docs. | Shows how mature systems separate model, trainer, recipe, data prep, eval, replay, logs, and release evidence. |
| QuarkLM codebase | `src/closed_world_lm`, `tests`, `corpus`, `evals`, `runs`, and current docs. | Identifies what QuarkLM has already built and where the next implementation should land. |

## Research Synthesis

### 1. Continual Learning Is The Core Frame

The continual-learning literature treats ongoing updates as a stability versus
plasticity problem, not as a single loss-function problem. The 2024 LLM
continual-learning survey separates continual pretraining, instruction tuning,
and alignment, and contrasts them with retrieval and model editing
([Wu et al. 2024](https://arxiv.org/abs/2402.01364)). The larger 2024 survey
also frames continual learning around dynamic data distributions, task
structures, user preferences, and catastrophic forgetting
([Shi et al. 2024](https://arxiv.org/abs/2404.16789)).

QuarkLM implication: the project needs update identity. Every run should say
what kind of update it is making, what prior behaviors it must preserve, and
what evidence can reject it.

### 2. Replay Must Be A Data System

Deep Generative Replay separates the solver from the system that supplies prior
experience ([Shin et al. 2017](https://arxiv.org/abs/1705.08690)). Avalanche's
training docs operationalize continual learning with streams, strategies,
plugins, buffers, dataloader adaptation, and evaluation callbacks
([Avalanche training docs](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training)).
Reverb reaches the same design lesson from reinforcement learning: replay
requires explicit tables, insertion policies, sampling policies, and rate
control ([Cassirer et al. 2021](https://arxiv.org/abs/2102.04736)).

QuarkLM implication: `direct_answer_replay_plan.json` is the right seed, but
replay planning should become a standalone module that emits a plan before
training and is consumed by the trainer.

### 3. Self-Generated Data Needs Quarantine

Self-Instruct generates instruction candidates and filters them before
fine-tuning ([Wang et al. 2022](https://arxiv.org/abs/2212.10560);
[self-instruct repo](https://github.com/yizhongw/self-instruct)). STaR keeps
generated rationales only when the answer is correct
([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)). Self-Refine and
Reflexion improve behavior through inference-time feedback or linguistic
memory rather than weight updates
([Madaan et al. 2023](https://arxiv.org/abs/2303.17651);
[Shinn et al. 2023](https://arxiv.org/abs/2303.11366)).

QuarkLM implication: "I learned something new" cannot mean the model generated
text and immediately trained on it. It must mean proposed, quarantined,
verified, admitted, trained, evaluated, and promoted.

### 4. Self-Judgment Is Useful But Dangerous

The self-feedback survey separates self-evaluation from self-update and asks
when self-feedback works ([Liang et al. 2024](https://arxiv.org/abs/2407.14507)).
Self-Rewarding Language Models, Constitutional AI, and SCoRe all show that
model-produced feedback can be useful, but they rely on larger pretrained
systems, explicit principles, or reward pipelines that QuarkLM does not yet
have ([Yuan et al. 2024](https://arxiv.org/abs/2401.10020);
[Bai et al. 2022](https://arxiv.org/abs/2212.08073);
[Kumar et al. 2024](https://arxiv.org/abs/2409.12917)). Self-bias and reward
hacking work shows why self-judgment cannot be the first gate
([Xu et al. 2024](https://arxiv.org/abs/2402.11436);
[Panickssery et al. 2024](https://arxiv.org/abs/2404.13076);
[Pan et al. 2024](https://arxiv.org/abs/2402.06627);
[Skalse et al. 2022](https://arxiv.org/abs/2209.13085)).

QuarkLM implication: deterministic verifier checks must come before a learned
verifier. A learned verifier can later be trained from admitted accept/reject
history, but it should remain advisory until it passes held-out verifier evals.

### 5. Verifiable Rewards Are The Right Near-Term Pattern

Tulu 3 exposes recipes, decontamination, evaluation, SFT, DPO, and
reinforcement learning with verifiable rewards
([Lambert et al. 2024](https://arxiv.org/abs/2411.15124);
[Open-Instruct repo](https://github.com/allenai/open-instruct)). DeepSeek-R1
shows rule-based rewards can induce reasoning behavior, while also documenting
readability and language-mixing problems that required extra training stages
([DeepSeek-AI 2025](https://arxiv.org/abs/2501.12948)).

QuarkLM implication: first rewards should be objective and closed-world:
ledger membership, exact answer consistency, unknown-policy compliance, prompt
leakage absence, train/eval separation, profile coverage, and branch diversity.

### 6. Model Collapse Is A Direct Threat

The model-collapse literature is not abstract for QuarkLM. Recursive training
on generated data can remove tail information
([Shumailov et al. 2023](https://arxiv.org/abs/2305.17493)). Accumulating real
and synthetic data rather than replacing originals can avoid collapse in the
studied settings ([Gerstgrasser et al. 2024](https://arxiv.org/abs/2404.01413)).
Other work reports diversity loss when language models recursively train on
synthetic text ([Guo et al. 2023](https://arxiv.org/abs/2311.09807)).

QuarkLM implication: admitted originals are permanent evidence. Generated
candidates must keep origin labels, never replace originals, and never train
weights until accepted by closed-world verification.

### 7. Data-Centric Small Models Require Mixture Discipline

TinyStories shows small models can learn coherent English from constrained
data, but its data is externally generated and cannot be used by QuarkLM
([Eldan and Li 2023](https://arxiv.org/abs/2305.07759)). BabyLM shows
sample-efficient language learning is still an active research problem and
that many curriculum attempts only helped modestly
([Warstadt et al. 2025](https://arxiv.org/abs/2504.08165)). SmolLM2 is a
modern data-centric example: staged mixtures, ablations, manual mixture
refinement, and released prepared datasets
([Allal et al. 2025](https://arxiv.org/abs/2502.02737)).

QuarkLM implication: corpus growth, mixture reports, rare-profile coverage,
and train/eval hygiene matter more right now than scaling the toy model.

### 8. Transparency Is Part Of The Claim

Pythia uses fixed data order and many checkpoints to make training dynamics
analyzable ([Biderman et al. 2023](https://arxiv.org/abs/2304.01373)).
LLM360 argues that open research needs code, data, checkpoints, intermediate
results, and analyses, not only final weights
([Liu et al. 2023](https://arxiv.org/abs/2312.06550)). OLMo and OLMo 2 release
training data, recipes, logs, checkpoints, and evaluation infrastructure
([Groeneveld et al. 2024](https://arxiv.org/abs/2402.00838);
[Team OLMo 2025](https://arxiv.org/abs/2501.00656);
[OLMo repo](https://github.com/allenai/OLMo)). Dolma makes corpus curation and
documentation a first-class research artifact
([Soldaini et al. 2024](https://arxiv.org/abs/2402.00159);
[Dolma repo](https://github.com/allenai/dolma)).

QuarkLM implication: rejected screens, run configs, replay plans, docs, and
failed attempts are research artifacts, not clutter.

### 9. Retrieval And Model Editing Are Boundaries, Not Substitutes

RAG distinguishes parametric memory from non-parametric memory and helps with
knowledge-intensive tasks, but it does not prove weights learned the content
([Lewis et al. 2020](https://arxiv.org/abs/2005.11401)). Model editing aims
to alter local model behavior without broader side effects, but locality and
unintended changes remain core concerns
([Yao et al. 2023](https://arxiv.org/abs/2305.13172)).

QuarkLM implication: exact responders and memory rails are useful scaffolding.
They must remain labeled as rails until a trained checkpoint earns the same
behavior through evaluation.

### 10. Tokenizer Growth Must Be An Artifacted Training Problem

BPE and SentencePiece show why subword tokenization matters for rare words and
raw-text training ([Sennrich et al. 2015](https://arxiv.org/abs/1508.07909);
[Kudo and Richardson 2018](https://arxiv.org/abs/1808.06226)). Hugging Face
tokenizers expose the practical shape of modern tokenization: normalization,
pre-tokenization, special tokens, padding, truncation, decoding, alignment,
training, and serialization
([tokenizers repo](https://github.com/huggingface/tokenizers)).

QuarkLM implication: character tokenization remains the purity baseline. Any
future subword tokenizer must include an admitted-corpus-only vocabulary proof,
manifest, special-token policy, offsets, migration tests, and checkpoint
compatibility.

## Open-Source Mechanics Review

| Reference | Pattern observed | QuarkLM implementation lesson |
| --- | --- | --- |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Small `model.py` and `train.py`, simple checkpoints, validation loss, readable experimentation. | Keep the transformer auditable, but separate model definition from training objectives and artifacts. |
| [minGPT](https://github.com/karpathy/minGPT) | Model, BPE, and GPT-independent trainer are separate files. | Extract trainer/replay/eval responsibilities from the current transformer monolith. |
| [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) | YAML-driven train/eval/generate entry points and explicit data preparation. | Add named recipe objects and stable CLI surfaces before more modes. |
| [LLM Foundry](https://github.com/mosaicml/llm-foundry) | Source modules, data prep, training, inference, evaluation, benchmarks, and platform launch scripts are separate. | Build the same responsibility boundaries in miniature. |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | Validated YAML recipes for pretrain, finetune, and deploy. | Recipes should be data-bearing artifacts, not hidden argparse combinations. |
| [Avalanche](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training) | Continual strategies use plugins, replay buffers, dataloader adaptation, and evaluation streams. | Replay should be explicit and profile-aware before training starts. |
| [Open-Instruct](https://github.com/allenai/open-instruct) | Training configs, eval suite, RLVR scripts, and decontamination tools are visible repo areas. | Add corpus hygiene and protected-prompt overlap reports to every run path. |
| [OLMo](https://github.com/allenai/OLMo) | Recipes, seeds, checkpoints, logs, and evaluation references are public. | Treat run metadata and rejected screens as part of QuarkLM's public evidence. |
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | Tokenizers are trained, serialized, aligned, and policy-bearing components. | Defer subword tokenization until QuarkLM can prove tokenizer provenance. |

## Current QuarkLM Codebase Diagnosis

### Strengths

- `self_improve.py` already creates reports with prompt-leakage, forgetting,
  exact-eval, promotion-gate, corpus-snapshot, corpus-diff, attempt archive,
  and deterministic self-diagnosis evidence.
- `provenance.py`, `curriculum.py`, `admit.py`, `respond.py`, `answer_model.py`,
  and `answer_decoder.py` each own a clear part of the closed-world scaffold.
- The transformer starts from random weights, uses the corpus-trained character
  tokenizer, and has accumulated real mechanics: AdamW-style state, scheduling,
  accumulation, normalization variants, tied embeddings, rotary positions,
  checkpoint metadata, branch profiles, profile-aware replay plans, and
  snapshot scoring.
- The test suite includes focused regression coverage for self-improvement,
  provenance, corpus admission, tokenizer behavior, answer models, and
  transformer branch mechanics.

### Gaps

- `src/closed_world_lm/transformer_char_model.py` is 9,494 lines and owns model
  code, optimizer code, direct-answer objectives, replay planning, snapshot
  scoring, CLI parsing, checkpoint writing, and run reporting.
- `tests/test_transformer_char_model.py` is 5,247 lines, which mirrors the
  implementation monolith and makes targeted regression work harder.
- v0.71 now records hypothesis, allowed data, planned artifacts, gates, failure
  criteria, and result decision before training.
- v0.72 now extracts replay planning from transformer objective plumbing into a
  standalone module.
- v0.73 now writes mandatory corpus hygiene and training-plan artifacts for the
  self-improvement and transformer answer-training paths.
- v0.75 adds a quarantine store with lifecycle states for candidate lessons,
  generated probes, repair proposals, diagnosis notes, and memory proposals.
- v0.76 adds a deterministic verifier interface for candidate checks and
  training-plan approval.
- Transformer promotion is still less integrated than the self-improvement
  promotion gate.

## Required Full Stack Before The Next Major Run

### 1. Experiment Registry

Every run must emit `experiment_intent.json` before training:

- version;
- run id;
- hypothesis;
- component;
- allowed data sources;
- planned artifacts;
- training recipe id;
- replay plan id;
- acceptance gates;
- failure criteria;
- final decision.

Acceptance: transformer screens and self-improvement runs cannot be considered
promotion candidates without this artifact.

### 2. Training Recipe Layer

Add named recipes that bind model family, tokenizer artifact, corpus/curriculum
source, replay plan, objective, optimizer, snapshot cadence, and promotion
gate.

Acceptance: a recipe can be re-run from a small JSON-compatible artifact
without reconstructing intent from an argparse command.

### 3. Corpus Governance

Add a corpus hygiene artifact for every training plan:

- ledger summary;
- original versus candidate counts;
- duplicate and near-duplicate pressure;
- train/eval overlap;
- source/profile mixture;
- rare-profile coverage;
- tokenizer vocabulary proof;
- generated-candidate ratio.

Acceptance: no training run starts without a corpus plan and no promotion
ignores protected-prompt overlap.

### 4. Candidate Quarantine

Generated lessons, generated probes, diagnosis proposals, and repair notes
must enter a candidate store before they can become training data.

Lifecycle states:

- proposed;
- verifier rejected;
- needs human review;
- verifier accepted;
- admitted;
- trained;
- promoted.

Acceptance: training data builders read only admitted records, never proposed
candidates.

### 5. Closed-World Verifier

Start deterministic. The verifier should check ledger membership, exact answer
consistency, unknown-policy behavior, protected prompt leakage, train/eval
overlap, profile coverage, replay preservation, and origin labels.

Acceptance: candidate admission and training-plan approval return structured
pass/fail evidence with reasons.

### 6. Replay Planner

Extract replay planning from `transformer_char_model.py` into a small module.
The trainer should consume a replay plan instead of recreating one privately.

Acceptance: focused tests prove profile isolation, rare-profile floors,
represented-target preservation, deficits, and replay ratios independent of
the transformer training loop.

### 7. Constraint-First Promotion

Promotion should first enforce constraints:

- no prompt leakage;
- no train/eval contamination;
- no unknown-policy regression;
- no forgetting regression;
- no profile coverage regression;
- no branch diversity regression;
- no rank-only promotion;
- no loss-only promotion.

Acceptance: rank, NLL, top-k, and loss are tie-breakers only after constraints
pass.

### 8. Transformer Refactor

Keep the CLI stable, but split responsibilities:

- `transformer_model.py`;
- `transformer_training.py`;
- `transformer_replay.py`;
- `transformer_eval.py`;
- `experiment_registry.py`;
- `candidate_quarantine.py`;
- `closed_world_verifier.py`;
- `training_recipe.py`.

Acceptance: new objectives should become small recipe/objective additions, not
large patches across a 9,494-line file.

### 9. Learned Improvement Policy, Later

Only after deterministic verifier and candidate history exist should QuarkLM
train a learned repair proposer or learned verifier from its own admitted
artifacts.

Acceptance: learned self-improvement remains advisory until it passes held-out
verifier and repair-selection evals.

## Revised Version Sequence

### v0.70

Record this deep cross-referenced research review and shift the next
implementation phase from objective tuning to operating-system mechanics. No
model-quality claim.

### v0.71

Implemented experiment registry and run-intent schemas. Self-improvement runs
and transformer screens now record hypothesis, allowed data, planned artifacts,
acceptance gates, failure criteria, and result decision before their output is
treated as evidence. Transformer screens still do not promote without the later
constraint-first gate.

### v0.72

Implemented replay planning extraction into a standalone module with focused
tests. The existing profile-aware replay behavior is preserved, but replay
plan construction is now inspectable without loading the transformer trainer.

### v0.73

Implemented corpus hygiene and training-plan artifacts for self-improvement
and transformer paths. Runs now record source mixtures, duplicate checks,
train/eval overlap, candidate ratios, rare-profile coverage, allowed data
sources, planned artifacts, and replay-plan summaries where applicable.

### v0.74

Added `RESEARCH_IMPLEMENTATION_MAP.md` and the matching Docusaurus Learn page.
This checkpoint turns the deeper research review into a direct implementation
map: source cluster, external mechanics pattern, QuarkLM gap, required
implementation, and acceptance evidence.

### v0.75

Implemented candidate quarantine artifacts and lifecycle states.
Self-improvement and transformer answer-training runs write
`candidate_quarantine.json`, and training plans link its summary.

### v0.76

Implemented deterministic closed-world verifier checks for candidate acceptance
and training-plan approval. Self-improvement and transformer answer-training
runs now write `closed_world_verifier.json`, and training plans embed verifier
summaries.

### v0.77

Implemented recipe objects and constraint-first promotion gates. Runs now write
`training_recipe.json` and `constraint_first_promotion.json`, and transformer
screens cannot promote from loss, rank, top-k, or NLL movement unless
constraints pass first.

### v0.78

Implemented the first transformer responsibility split. Answer-training now
keeps experiment/artifact contracts, recipe surfaces, promotion decisions,
trainer utilities, and the direct-answer objective catalog outside the
transformer monolith.

### v0.79

Implemented transformer model/config and checkpoint metadata extraction.
`src/closed_world_lm/transformer_model.py` now owns model, optimizer, and
generation configs, validation, checkpoint identity, closed-world dataset
metadata, arg-to-config adapters, and run metadata.

### v0.80

Implemented transformer eval/checkpoint-load extraction.
`src/closed_world_lm/transformer_checkpoint.py` now owns checkpoint payload
loading and identity validation. `src/closed_world_lm/transformer_eval.py` now
owns probe loading, candidate collection, generic scoring, report assembly,
samples JSONL writing, and eval JSON writing.

### v0.81

Implemented the first post-surface anti-collapse objective:
`branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`.
It adds balanced owned target-share pressure across replay targets inside each
profile-aware group, so the preserving-deficit objective no longer protects
only the currently represented target token.

### v0.82

Screened the v0.81 objective mechanic under the existing constraint-first gates
in `runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/`.
The run writes the modern artifact set, fixes the transformer metrics
`external_embeddings` purity field, passes the verifier and branch-context
gate, and preserves target coverage after restore. It is rejected evidence:
trained snapshots still collapse QA and heldout branch diversity before any
rank gain can be trusted.

### v0.83+

The next transformer repair should target prompt-specific branch diversity
directly before tokenizer growth or a learned verifier experiment.

## Decision

Deep research confirms the direction: QuarkLM's novelty is not just training a
tiny transformer from scratch. The valuable research claim is a closed-world
learning lifecycle in which every new belief has provenance, every generated
candidate is quarantined, every weight update is measured, and every promotion
can be rejected when it improves one score by damaging the learner's world.

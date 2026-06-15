# QuarkLM Research Implementation Map

Last updated: 2026-06-14.

## Purpose

This map answers the v0.74 research concern directly: QuarkLM should not move
forward from a shallow literature pass or isolated implementation iteration.
The project needs a cross-referenced implementation map that ties external
research, public open-source mechanics, and the current codebase to the next
versioned work.

v0.74 is therefore a research-control checkpoint, not a model-quality release.
It does not claim better answers. It records what must be implemented before
the next larger transformer repair run and why.

## Boundary

Allowed:

- Use papers, official project docs, and public repositories as design
  references.
- Compare QuarkLM's local implementation against those references.
- Turn the comparison into versioned implementation requirements.

Not allowed:

- Copy implementation code from outside projects.
- Import pretrained weights, pretrained tokenizers, external embeddings, or
  external datasets.
- Use an external model as a teacher, verifier, judge, reward model, repair
  generator, or hidden data shaper.
- Treat retrieval, exact responders, generated candidates, or research notes as
  proof that QuarkLM's model weights learned the behavior.

## Research Clusters

| Cluster | Sources reviewed | Implementation lesson |
| --- | --- | --- |
| Transformer language modeling | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), GPT-style decoder practice, [nanoGPT](https://github.com/karpathy/nanoGPT), [llm.c](https://github.com/karpathy/llm.c) | A transformer is the right backbone for contextual token binding, but QuarkLM must keep model, optimizer, data, eval, checkpoint, and recipe concerns separated enough to audit. |
| Continual learning and forgetting | [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796), [Continual Lifelong Learning with Neural Networks](https://arxiv.org/abs/1802.07569), [Continual Lifelong Learning in NLP](https://arxiv.org/abs/2012.09823), [A Continual Learning Survey](https://arxiv.org/abs/1909.08383) | Self-improvement is a stability-plasticity problem. Every weight update needs a declared preservation target, not only a new-task loss. |
| Small-data language learning | [BabyLM findings](https://arxiv.org/abs/2504.08165), [Second BabyLM findings](https://arxiv.org/abs/2412.05149), [TinyStories](https://arxiv.org/abs/2305.07759) | A small corpus can be scientifically useful, but data quality, held-out design, and curriculum claims must be explicit. External synthetic corpora are references only, not QuarkLM training data. |
| Self-generated data | [Self-Instruct](https://arxiv.org/abs/2212.10560), [STaR](https://arxiv.org/abs/2203.14465), [Self-Refine](https://arxiv.org/abs/2303.17651), [Reflexion](https://arxiv.org/abs/2303.11366) | Generated lessons and repair traces must be candidates first. They cannot train weights until verified, admitted, and evaluated. |
| Self-judgment and alignment | [Constitutional AI](https://arxiv.org/abs/2212.08073), [DPO](https://arxiv.org/abs/2305.18290), self-feedback and self-bias work referenced in `DEEP_RESEARCH_REVIEW.md` | Learned or model-produced judgment can be useful later, but deterministic closed-world verification must come first. |
| Verifiers and process rewards | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), Tulu/Open-Instruct, DeepSeek-R1 | QuarkLM's first rewards should be objective: ledger membership, exact responder agreement, unknown-policy compliance, no leakage, profile retention, and branch diversity. |
| Tokenization | [Subword Units](https://arxiv.org/abs/1508.07909), [SentencePiece](https://arxiv.org/abs/1808.06226), byte-level BPE references | Character tokenization remains the safe purity baseline. Subword tokenization should wait until tokenizer manifests, offsets, special-token policy, and corpus-only vocabulary proof exist. |
| Data curation and contamination | [The Pile](https://arxiv.org/abs/2101.00027), [Dolma](https://arxiv.org/abs/2402.00159), [DataComp-LM](https://arxiv.org/abs/2406.11794), Open-Instruct decontamination | Data is a first-class system. QuarkLM needs source mixtures, duplicate checks, protected prompt overlap, candidate ratios, rare-profile coverage, and artifacted training plans. |
| Transparent open models | [Pythia](https://arxiv.org/abs/2304.01373), [OLMo](https://arxiv.org/abs/2402.00838), OLMo repository, LLM360 | Reproducibility requires recipes, configs, logs, checkpoints, intermediate results, and rejected screens, not only final weights. |
| Open-source training mechanics | nanoGPT, GPT-NeoX, OLMo, llm.c, LitGPT, LLM Foundry, Hugging Face tokenizers, Avalanche | Mature stacks separate data preparation, tokenizer, model definition, trainer, recipes, evaluation, checkpoints, and release evidence. QuarkLM should mirror those boundaries in miniature. |

## QuarkLM Gap Map

| Needed mechanic | External evidence | Current QuarkLM state | Required implementation |
| --- | --- | --- | --- |
| Experiment intent | OLMo/Pythia transparency and reproducible training setup | v0.71 writes `experiment_intent.json` for self-improvement and transformer answer-training paths. | Keep this mandatory for all future training and screening paths. |
| Replay planning | Continual-learning replay systems and Avalanche-style explicit strategies | v0.72 extracts profile-aware replay planning into `replay_plan.py`. | Feed replay plans into verifier, recipe, and promotion gates rather than leaving them as reports only. |
| Corpus hygiene | Dolma, DataComp-LM, Open-Instruct decontamination | v0.73 writes `corpus_hygiene.json` and `training_plan.json`. | Promote hygiene from reporting to pre-training approval once the verifier exists. |
| Candidate quarantine | Self-Instruct filters, STaR correctness filter, model-collapse risk | v0.75 writes `candidate_quarantine.json` with candidate lifecycle states. | Keep non-admitted candidate records excluded from training and feed candidate manifests into verifier and recipe gates. |
| Deterministic verifier | Verifier/process-supervision research and verifiable rewards | v0.76 writes `closed_world_verifier.json` and embeds verifier summaries in training plans. | Use verifier approval as the required pre-training integrity gate before recipes or promotion gates trust a run. |
| Recipe layer | GPT-NeoX/OLMo/LitGPT recipe/config practice | v0.77 writes `training_recipe.json` for self-improvement and transformer answer-training paths. | Use recipe artifacts as the reproducible bridge between intent, training plans, replay, and promotion gates. |
| Constraint-first promotion | Continual-learning forgetting literature and QuarkLM v0.68 rejection evidence | v0.77 writes `constraint_first_promotion.json` and makes transformer decisions depend on it. | Keep loss, NLL, rank, top-k, and exact quality checks advisory until closed-world constraints pass first. |
| Transformer boundaries | nanoGPT keeps model and trainer readable; OLMo/GPT-NeoX separate configs, train/eval, data | v0.78 extracts transformer experiment/artifact contracts, trainer utilities, and the direct-answer objective catalog; v0.79 extracts model/config and checkpoint metadata surfaces; v0.80 extracts eval/checkpoint-load surfaces. | Use the narrower surfaces before another objective-repair screen. |
| Learned verifier or repair policy | Self-reward/self-feedback work is promising but fragile | Current self-diagnosis is deterministic and explicitly `uses_external_model: false`. | Defer learned self-improvement until accepted/rejected candidate history and verifier evals exist. |

## Implementation Ladder

### v0.74

Record this research implementation map in the root docs and Docusaurus Learn
section. Update current state so the project acknowledges that deeper
cross-referenced research is a required control surface, not a side note.

Acceptance:

- `RESEARCH_IMPLEMENTATION_MAP.md` exists.
- Docusaurus exposes the same decision.
- README, STATUS, GOAL, HISTORY, and shared current state point to the map.
- No model-quality claim is made.

### v0.75

Implemented candidate quarantine artifacts.

Acceptance:

- Candidate records have stable ids, types, source, state, evidence, notes, and
  optional admission links.
- Lifecycle transitions are validated.
- Self-improvement and transformer runs write `candidate_quarantine.json` even
  when empty.
- Training plans state that candidates are not training data until admitted.

### v0.76

Implemented deterministic closed-world verifier checks.

Acceptance:

- Candidate admission and training-plan approval return structured pass/fail
  evidence.
- The verifier checks the closed-world data boundary, candidate exclusion,
  candidate quarantine validity, source labels, exact answer consistency when a
  closed-world responder is supplied, and protected train/eval overlap.
- Candidate quarantine can use verifier results without using an external
  model.

### v0.77

Implemented recipe objects and constraint-first promotion gates.

Acceptance:

- A recipe artifact can rerun a screen without reconstructing the run from
  argparse flags.
- Transformer screens cannot promote from loss, NLL, rank, or top-k movement
  unless retention, leakage, contamination, coverage, and diversity constraints
  pass.

### v0.78

Implemented the first transformer responsibility refactor behind the new
recipe and verifier surfaces. Added `transformer_experiment.py`,
`transformer_training.py`, and `transformer_objectives.py`.

Acceptance:

- Artifact contracts, experiment intent, recipe creation, and promotion
  decision logic are separate from the transformer monolith.
- JSONL history writing, shuffled training cursors, and loss averaging are
  separately tested trainer utilities.
- Direct-answer objective names live in a testable objective catalog instead
  of the CLI parser.

### v0.79

Implemented transformer model/config and checkpoint metadata surfaces in
`src/closed_world_lm/transformer_model.py`.

Acceptance:

- Model, optimizer, and generation config dataclasses and validators live
  outside the transformer monolith.
- Checkpoint architecture, checkpoint format, tokenizer identity, dataset
  metadata, and run metadata are centralized and tested.
- `transformer_char_model.py` re-exports the old names for compatibility.

### v0.80

Implemented transformer eval/checkpoint-load surfaces in
`src/closed_world_lm/transformer_checkpoint.py` and
`src/closed_world_lm/transformer_eval.py`.

Acceptance:

- Checkpoint payload loading and identity validation live outside the model
  class.
- Generic transformer eval scoring, probe loading, candidate collection,
  report assembly, and eval artifact writing live outside the monolith.
- The public eval CLI and artifact shapes remain stable.

### v0.81

Implemented a profile target-share anti-collapse objective:
`branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`.
The loss balances owned target-share pressure across replay targets inside
each profile-aware replay group, preserving the closed-world replay plan while
reducing the chance that one represented target dominates a multi-target
profile.

Acceptance:

- Profile-aware replay modes can select the new objective.
- The new mode still emits the replay-plan artifact through the existing
  profile-aware run surfaces.
- Focused tests prove the balanced target-share term lifts a minority replay
  target more than the previous profile-aware replay loss.

### v0.82

Screened the profile target-share objective under the full modern artifact and
constraint-first gates in
`runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/`.
The run passed the deterministic verifier, wrote experiment intent, corpus
hygiene, training plan, candidate quarantine, recipe, replay plan, metrics, and
constraint-first artifacts, and fixed the transformer metrics purity report so
`external_embeddings: false` reaches the promotion gate.

Acceptance:

- Replay plan records `9144` branch/replay records across `21` profiles.
- Branch-context gate passes across `219/219` semantic records.
- Constraint-first promotion sees no pretrained weights, no pretrained
  tokenizer, and no external embeddings.
- Target coverage is preserved after best-snapshot restore.
- Promotion remains rejected because `branch_diversity_target` fails; step `40`
  improves QA rank only by collapsing QA/heldout to one `"c"` token with
  `0.0` target-token coverage.

### v0.83

Implemented and screened prompt-specific branch ownership:
`branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The loss keeps profile target-share pressure, then adds a sibling-target margin
so each replay context is trained to put its own target above other targets
from the same profile.

Acceptance:

- The new mode remains profile-aware and emits `direct_answer_replay_plan.json`.
- Focused tests show prompt-ownership margins lift a context-specific target
  more than the v0.82 target-share pressure.
- The full screen writes the modern artifact set in
  `runs/transformer-answer-v0.83-fullstack-prompt-ownership-smoke-dim4-context80/`.
- Promotion remains rejected because trained snapshots still collapse QA and
  heldout to one `"c"` token with `0.0` target-token coverage, even though QA
  average target rank improves to `8.625`.

### v0.84

Implemented and screened baseline replay anchors:
`branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The loss keeps prompt ownership, target-share balancing, deficit focus, and
coverage preservation, but replay preservation uses the baseline replay
prediction captured before direct-answer training instead of following current
prediction drift.

Acceptance:

- The new mode remains profile-aware and emits `direct_answer_replay_plan.json`.
- Focused tests show profiled replay batches can use baseline prediction
  overrides and that anchored preservation protects a covered target better
  than dynamic prediction preservation.
- The full screen writes the modern artifact set in
  `runs/transformer-answer-v0.84-fullstack-baseline-anchored-prompt-ownership-smoke-dim4-context80/`.
- Replay-plan evidence records `562` active baseline prediction anchors.
- Promotion remains rejected because trained snapshots still collapse QA and
  heldout to one `"i"` token with target-token coverage `0.125`, below the
  baseline `0.25` floor, even though QA average target rank improves to `8.0`.

### v0.85

Implemented and screened baseline-floor update gating:
`branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The mode keeps baseline replay anchors but treats the profile-wise step-0
target-token coverage floor as an update-acceptance rule: an attempted
direct-answer update is rolled back if the branch-profile probe falls below the
floor.

Acceptance:

- The new mode remains profile-aware and emits `direct_answer_replay_plan.json`.
- Focused tests show the mode records active baseline replay anchors and update
  guard accounting.
- The full screen writes the modern artifact set in
  `runs/transformer-answer-v0.85-fullstack-baseline-floor-gated-prompt-ownership-smoke-dim4-context80/`.
- Replay-plan evidence records `562` active baseline prediction anchors.
- The update guard checks `50/50` attempted steps and rejects `50/50`, preserving
  QA/heldout coverage at the baseline `0.25` floor in every recorded snapshot.
- Promotion remains rejected because no update is accepted and
  `branch_diversity_target` still fails across all `9` multi-target profiles.

### v0.86

Implemented and screened adaptive baseline-floor retries:
`branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The mode restores model, optimizer, and RNG state, then retries the same
direct-answer update at learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01`
before rejecting the step.

Acceptance:

- The new mode remains profile-aware and emits `direct_answer_replay_plan.json`.
- Focused tests show the mode records adaptive retry accounting.
- The full screen writes the modern artifact set in
  `runs/transformer-answer-v0.86-fullstack-baseline-floor-adaptive-prompt-ownership-smoke-dim4-context80/`.
- Replay-plan evidence records `562` active baseline prediction anchors and the
  adaptive scale list.
- The update guard checks `50/50` steps, attempts `200` scaled updates, and
  rejects `200/200`, preserving QA/heldout coverage at the baseline `0.25`
  floor.
- Promotion remains rejected because no scaled update is accepted and
  `branch_diversity_target` still fails across all `9` multi-target profiles.

### v0.87

Implemented and screened baseline-floor repair retries:
`branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The mode keeps adaptive scaled retries, then applies one bounded
baseline-covered anchor repair before deciding whether to keep or roll back the
candidate update.

Acceptance:

- The new mode remains profile-aware and emits `direct_answer_replay_plan.json`.
- Focused tests show the mode records repair-anchor and repaired-attempt
  accounting.
- The clean full screen writes the modern artifact set in
  `runs/transformer-answer-v0.87-fullstack-baseline-floor-repaired-prompt-ownership-clean-smoke-dim4-context80/`.
- Replay-plan evidence records `562` active baseline prediction anchors, `227`
  repair anchors, and one repair step per failed retry.
- The update guard checks `50/50` steps, attempts `200` updates, runs `200`
  one-step repairs, and rejects `200/200`, preserving QA/heldout coverage at the
  baseline `0.25` floor.
- Promotion remains rejected because no repaired update is accepted and
  `branch_diversity_target` still fails across all `9` multi-target profiles.

### v0.88

Implemented and screened objective-side baseline-floor anchors:
`branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The mode moves a balanced batch of baseline-covered floor anchors into the same
loss and backward pass as the branch-diversity objective before the optimizer
step.

Acceptance:

- The new mode remains profile-aware and emits `direct_answer_replay_plan.json`.
- Focused tests show the mode records objective-anchor and accepted/rejected
  guard accounting.
- The full screen writes the modern artifact set in
  `runs/transformer-answer-v0.88-fullstack-baseline-floor-objective-prompt-ownership-smoke-dim4-context80/`.
- Replay-plan evidence records `562` active baseline prediction anchors, `227`
  floor anchors, anchor batch size `32`, and anchor weight `10.0`.
- The update guard checks `50/50` steps, attempts `200` updates, runs `200`
  objective anchor batches covering `2400` anchor records, and rejects
  `200/200`, preserving QA/heldout coverage at the baseline `0.25` floor.
- Promotion remains rejected because no objective-shaped update is accepted and
  `branch_diversity_target` still fails across all `9` multi-target profiles.

### v0.89+

Only after these operating surfaces are explicit should QuarkLM add another
branch-diversity repair. The next transformer step should first prove accepted
floor-stabilization updates before reintroducing branch-diversity pressure,
revisiting subword tokenization, or beginning a learned verifier/repair-policy
experiment.

## Decision

Yes: deep cross-referenced research and a review of how comparable systems are
structured are required. The next implementation work should be driven by this
map. QuarkLM's novelty depends on the whole lifecycle: admitted data,
quarantined candidates, deterministic verification, auditable weight updates,
constraint-first promotion, and transparent docs that move with every version.

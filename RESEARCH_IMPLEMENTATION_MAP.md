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
| Candidate quarantine | Self-Instruct filters, STaR correctness filter, model-collapse risk | Candidate ratio is visible, but there is no candidate store or lifecycle state. | Add candidate artifacts with proposed, quarantined, verified, rejected, admitted, trained, and promoted states. Training builders must ignore non-admitted candidates. |
| Deterministic verifier | Verifier/process-supervision research and verifiable rewards | Checks exist across separate audits, but there is no unified verifier interface. | Add `closed_world_verifier.py` that returns structured pass/fail reasons for candidates and training plans. |
| Recipe layer | GPT-NeoX/OLMo/LitGPT recipe/config practice | CLI flags still carry much of the recipe identity, especially in the transformer path. | Add named recipe artifacts that bind model, tokenizer, curriculum, replay plan, objective, optimizer, snapshots, and gates. |
| Constraint-first promotion | Continual-learning forgetting literature and QuarkLM v0.68 rejection evidence | Self-improvement has a stronger promotion gate than transformer screens. | Make transformer promotion impossible from loss, NLL, rank, or top-k alone. Constraints must pass first. |
| Transformer boundaries | nanoGPT keeps model and trainer readable; OLMo/GPT-NeoX separate configs, train/eval, data | `transformer_char_model.py` still owns too many responsibilities in one module. | Split model/config/checkpoint, training loop, direct-answer objective, eval, replay, recipe, and reporting surfaces after verifier/recipe gates are defined. |
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

Implement deterministic closed-world verifier checks.

Acceptance:

- Candidate admission and training-plan approval return structured pass/fail
  evidence.
- The verifier checks ledger membership, exact answer consistency,
  unknown-policy compliance, prompt leakage, source labels, and protected
  train/eval overlap.
- Candidate quarantine can use verifier results without using an external
  model.

### v0.77

Implement recipe objects and constraint-first promotion gates.

Acceptance:

- A recipe artifact can rerun a screen without reconstructing the run from
  argparse flags.
- Transformer screens cannot promote from loss, NLL, rank, or top-k movement
  unless retention, leakage, contamination, coverage, and diversity constraints
  pass.

### v0.78

Refactor transformer responsibilities behind the new recipe and verifier
surfaces.

Acceptance:

- New objectives become small additions rather than broad edits across the
  transformer monolith.
- Tests target model, trainer, replay, verifier, recipe, and eval behavior
  separately.

### v0.79+

Only after the operating system is explicit should QuarkLM add another
anti-collapse transformer objective, revisit subword tokenization, or begin a
learned verifier/repair-policy experiment.

## Decision

Yes: deep cross-referenced research and a review of how comparable systems are
structured are required. The next implementation work should be driven by this
map. QuarkLM's novelty depends on the whole lifecycle: admitted data,
quarantined candidates, deterministic verification, auditable weight updates,
constraint-first promotion, and transparent docs that move with every version.

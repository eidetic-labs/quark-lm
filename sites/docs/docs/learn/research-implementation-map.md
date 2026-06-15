---
title: Research Implementation Map
description: The v0.74 cross-referenced map from papers and open-source mechanics to QuarkLM implementation requirements.
---

# Research Implementation Map

Last reviewed: 2026-06-14.

The full map lives in the repository root at `RESEARCH_IMPLEMENTATION_MAP.md`.

v0.74 is a research-control checkpoint. It does not claim better model
behavior. It records the source-backed implementation map that should guide the
next mechanics before another larger transformer repair run.

## Why It Exists

The project already had a forward plan and a deep research review. The missing
piece was a direct implementation ledger:

- research cluster;
- public implementation pattern;
- QuarkLM gap;
- required mechanic;
- acceptance evidence.

That ledger prevents QuarkLM from drifting into knob turning. Each new version
should now connect to a stated research and implementation reason.

## Source Clusters

The v0.74 map cross-references:

- transformer language modeling from the original Transformer paper, GPT-style
  decoder practice, nanoGPT, llm.c, GPT-NeoX, and OLMo;
- continual-learning and catastrophic-forgetting work, including EWC and
  lifelong-learning surveys;
- small-data language-learning work such as BabyLM and TinyStories;
- self-generated data methods such as Self-Instruct, STaR, Self-Refine, and
  Reflexion;
- verifier and process-supervision work such as GSM8K verifiers and process
  reward models;
- data curation and contamination work from The Pile, Dolma, DataComp-LM, and
  Open-Instruct;
- tokenizer work from BPE, SentencePiece, and byte-level subword systems;
- transparent open-model practice from Pythia, OLMo, OLMo 2, and LLM360.

These are design references only. QuarkLM still forbids pretrained weights,
pretrained tokenizers, external embeddings, external datasets, copied code, and
external-model-shaped training data.

## Implementation Decision

QuarkLM should continue the self-improvement operating system before another
direct-answer objective mode:

1. **v0.75:** candidate quarantine artifacts and lifecycle states. Implemented.
2. **v0.76:** deterministic closed-world verifier checks. Implemented.
3. **v0.77:** recipe objects and constraint-first promotion gates. Implemented.
4. **v0.78:** transformer responsibility surfaces for experiments,
   artifacts, trainer utilities, and objective catalog. Implemented.
5. **v0.79:** transformer model/config and checkpoint metadata surfaces.
   Implemented.
6. **v0.80:** transformer eval/checkpoint-load surfaces. Implemented.
7. **v0.81:** profile target-share anti-collapse objective. Implemented.
8. **v0.82:** profile target-share full-stack screen. Implemented and
   rejected.
9. **v0.83:** prompt-specific branch ownership. Implemented and rejected.
10. **v0.84:** baseline replay anchors. Implemented and rejected.
11. **v0.85:** baseline-floor update gating. Implemented and rejected.
12. **v0.86:** adaptive baseline-floor retries. Implemented and rejected.
13. **v0.87:** baseline-floor repair retries. Implemented and rejected.
14. **v0.88:** objective-side baseline-floor anchors. Implemented and rejected.
15. **v0.89:** stabilization-only baseline-floor updates. Implemented and
   rejected.
16. **v0.90:** baseline-floor rejection diagnostics. Implemented.
17. **v0.91:** profile-targeted floor stabilization. Implemented and rejected.
18. **v0.92:** sequential source-profile floor stabilization. Implemented and
   rejected.
19. **v0.93:** calibrated sequential profile-floor stabilization. Implemented
   with one accepted guarded update; rejected for promotion.
20. **v0.94:** profile-scale calibrated floor stabilization. Implemented with
   eight accepted guarded source-profile updates; rejected for promotion.
21. **v0.95:** diversity-aware profile-scale acceptance. Implemented with
   five accepted score-improving source-profile updates; rejected for promotion.
22. **v0.96:** frontier-driven branch diversity under the full baseline
   target-token floor. Implemented with nine score-improving source-profile
   updates; rejected for promotion.
23. **v0.97:** coverage-frontier audited missing-target repair. Implemented
   with one coverage-gaining update; rejected for promotion.
24. **v0.98:** coverage-prep frontier repair. Implemented with six safe
   preparation moves; rejected for promotion.
25. **v0.99:** coverage-recovery frontier retry. Implemented with two recovery
   conversions; rejected for promotion.
26. **v0.100.0:** branch-stable coverage recovery. Implemented with two
   branch-stable recovery conversions; rejected for promotion.
27. **v0.101.0:** branch-diversity recovery. Implemented with five local
   branch-score refinements; rejected for promotion.
28. **v0.102.0:** collapsed-profile binding. Implemented with one targeted
   binding update and three remaining collapsed eval profiles; rejected for
   promotion.
29. **v0.103.0:** remaining-profile binding. Implemented with six prioritized
   remaining-profile acceptances and a learning coverage gain; rejected for
   promotion.
30. **v0.104.0:** owner/paraphrase residual binding. Implemented with six
   prioritized acceptances and protected-learning rejection evidence; rejected
   for promotion.
31. **v0.105.0:** closed-world retrieval memory. Implemented with a corpus-only
   `retrieval_memory_report.json` artifact, `497` memory cards, and `219/219`
   exact retrieval evals without external embeddings or weight updates.
32. **v0.106.0:** memory-guided consolidation planning. Implemented with
   `memory_consolidation_plan.json`, `9` memory-backed neural failed profiles,
   and top priorities `owner`, `paraphrases`, `glossary`,
   `admission_paraphrases`, and `admissions`.
33. **v0.107.0:** gated memory-consolidation training. Implemented with a
   declared source consolidation plan, consumed targets `owner`, `paraphrases`,
   and `glossary`, and `26` prioritized attempts with `8` acceptances and `18`
   rejections; rejected for promotion on `branch_diversity_target`.
34. **v0.108.0:** expanded memory-consolidation target window. Implemented with
   target-profile-to-source-label mapping and a five-profile source-plan screen
   for `owner`, `paraphrases`, `heldout`, `qa`, and `glossary`; rejected for
   promotion on `branch_diversity_target`.
35. **v0.109.0+:** missing first-token diversity repair, tokenizer growth, or
   learned verifier experiments.

## Current Gap

QuarkLM already has:

- v0.71 experiment intent;
- v0.72 replay planning;
- v0.73 corpus hygiene and training plans;
- v0.75 candidate quarantine artifacts and lifecycle states;
- v0.76 deterministic closed-world verifier checks;
- v0.77 recipe objects and constraint-first promotion gates.
- v0.78 transformer experiment/artifact surfaces, trainer utilities, and
  direct-answer objective catalog.
- v0.79 transformer model/config and checkpoint metadata surfaces.
- v0.80 transformer eval/checkpoint-load surfaces.
- v0.81 profile target-share objective mode:
  `branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`.
- v0.82 full target-share screen evidence:
  `runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/`.
- v0.83 prompt-specific ownership mode:
  `branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
- v0.83 full prompt-ownership screen evidence:
  `runs/transformer-answer-v0.83-fullstack-prompt-ownership-smoke-dim4-context80/`.
- v0.84 baseline-anchored prompt-ownership mode:
  `branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
- v0.84 full baseline-anchor screen evidence:
  `runs/transformer-answer-v0.84-fullstack-baseline-anchored-prompt-ownership-smoke-dim4-context80/`.
- v0.85 baseline-floor update-gated prompt-ownership mode:
  `branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
- v0.85 full baseline-floor update-gate screen evidence:
  `runs/transformer-answer-v0.85-fullstack-baseline-floor-gated-prompt-ownership-smoke-dim4-context80/`.
- v0.86 adaptive baseline-floor prompt-ownership mode:
  `branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
- v0.86 full adaptive baseline-floor screen evidence:
  `runs/transformer-answer-v0.86-fullstack-baseline-floor-adaptive-prompt-ownership-smoke-dim4-context80/`.
- v0.87 baseline-floor repaired prompt-ownership mode:
  `branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
- v0.87 clean full repaired baseline-floor screen evidence:
  `runs/transformer-answer-v0.87-fullstack-baseline-floor-repaired-prompt-ownership-clean-smoke-dim4-context80/`.
- v0.88 baseline-floor objective prompt-ownership mode:
  `branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
- v0.88 full baseline-floor objective screen evidence:
  `runs/transformer-answer-v0.88-fullstack-baseline-floor-objective-prompt-ownership-smoke-dim4-context80/`.
- v0.89 baseline-floor stabilization mode:
  `branch-context-profile-baseline-floor-stabilization-unlikelihood`.
- v0.89 full baseline-floor stabilization screen evidence:
  `runs/transformer-answer-v0.89-fullstack-baseline-floor-stabilization-smoke-dim4-context80/`.
- v0.90 baseline-floor rejection diagnostic evidence:
  `runs/transformer-answer-v0.90-fullstack-baseline-floor-stabilization-diagnostics-smoke-dim4-context80/`.
- v0.91 profile-targeted baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.91-fullstack-baseline-floor-profile-targeted-stabilization-smoke-dim4-context80/`.
- v0.92 sequential source-profile baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.92-fullstack-baseline-floor-sequential-profile-stabilization-smoke-dim4-context80/`.
- v0.93 calibrated sequential baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.93-baseline-floor-calibrated-sequential-profile-stabilization-step1-dim4-context80/`.
- v0.94 profile-scale calibrated baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.94-baseline-floor-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.95 diversity-aware profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.95-baseline-floor-diversity-profile-scale-calibrated-sequential-stabilization-configured-step1-dim4-context80/`.
- v0.96 frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.96-baseline-floor-diversity-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.97 coverage-frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.97-baseline-floor-diversity-coverage-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.98 coverage-prep frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.98-baseline-floor-diversity-coverage-prep-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.99 coverage-recovery frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.99-baseline-floor-diversity-coverage-recovery-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.100.0 branch-stable coverage-recovery frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.100.0-baseline-floor-diversity-branch-stable-coverage-recovery-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.101.0 branch-diversity recovery frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.101.0-baseline-floor-diversity-branch-diversity-recovery-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.102.0 collapsed-profile binding frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.102.0-baseline-floor-diversity-collapsed-profile-binding-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.103.0 remaining-profile binding frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.103.0-baseline-floor-diversity-remaining-profile-binding-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.104.0 owner/paraphrase binding frontier profile-scale baseline-floor stabilization evidence:
  `runs/transformer-answer-v0.104.0-baseline-floor-diversity-owner-paraphrase-binding-frontier-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
- v0.105.0 closed-world retrieval memory evidence:
  `runs/transformer-answer-v0.105.0-retrieval-memory-owner-paraphrase-frontier-profile-scale-step1-dim4-context80/`.
- v0.106.0 memory-guided consolidation planning evidence:
  `runs/transformer-answer-v0.106.0-memory-guided-consolidation-owner-paraphrase-frontier-profile-scale-step1-dim4-context80/`.
- v0.107.0 gated memory-consolidation training evidence:
  `runs/transformer-answer-v0.107.0-gated-memory-consolidation-owner-paraphrase-glossary-frontier-profile-scale-step1-dim4-context80/`.
- v0.108.0 expanded memory-consolidation target-window evidence:
  `runs/transformer-answer-v0.108.0-expanded-memory-consolidation-owner-paraphrase-heldout-qa-glossary-frontier-profile-scale-step1-dim4-context80/`.

It still needs:

- missing first-token diversity repair that uses v0.108.0 expanded
  source-plan-guided consolidation evidence to improve target-token coverage
  without regressing retrieval provenance.

## Operating Rule

Every future mechanics version should answer three questions:

1. Which research or implementation pattern justifies this mechanic?
2. Which closed-world boundary does it protect?
3. Which artifact proves it worked or rejected the run?

That is how QuarkLM keeps the claim clean: the model grows from its admitted
dataset, and the project keeps enough evidence to prove what changed.

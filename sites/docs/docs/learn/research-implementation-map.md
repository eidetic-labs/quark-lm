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
7. **v0.81+:** anti-collapse objective, tokenizer growth, or learned verifier
   experiments.

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

It still needs:

- future objective repairs to use these narrower surfaces rather than broad
  monolith patches.

## Operating Rule

Every future mechanics version should answer three questions:

1. Which research or implementation pattern justifies this mechanic?
2. Which closed-world boundary does it protect?
3. Which artifact proves it worked or rejected the run?

That is how QuarkLM keeps the claim clean: the model grows from its admitted
dataset, and the project keeps enough evidence to prove what changed.

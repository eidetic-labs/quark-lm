---
title: Research Implementation Map
description: The v0.74 cross-referenced map from papers and open-source mechanics to QuarkLM implementation requirements.
---

# Research Implementation Map

Last reviewed: 2026-06-14.

The full map lives in the repository root at `RESEARCH_IMPLEMENTATION_MAP.md`.
This page is the durable summary: what the map is, why it exists, and which work
it has driven since v0.74.

## What v0.74 records

v0.74 is a research-control checkpoint. It makes no claim about model behavior.
It records a cross-referenced ledger that ties external research and public
open-source mechanics to QuarkLM's own gaps and to the next versioned work,
before any larger transformer repair run.

The project already had a forward plan
([Forward research plan](./forward-research-plan.md)) and a deep research review
([Deep research review](./deep-research-review.md)). The missing piece was a
direct implementation ledger with five columns:

| Column | What it answers |
| --- | --- |
| Research cluster | Which published work the mechanic draws on. |
| Public implementation pattern | How comparable open systems structure that piece. |
| QuarkLM gap | What the current codebase is missing. |
| Required mechanic | The versioned change that closes the gap. |
| Acceptance evidence | The artifact that proves it worked or rejected the run. |

That ledger keeps QuarkLM from drifting into knob turning. Each new version now
connects to a stated research and implementation reason rather than to an
unexplained metric move.

## Source clusters

The map cross-references design references only. No source listed below enters
QuarkLM as weights, tokenizers, embeddings, datasets, copied code, or
external-model-shaped training text.

| Cluster | Representative references |
| --- | --- |
| Transformer language modeling | the original Transformer paper, GPT-style decoder practice, nanoGPT, llm.c, GPT-NeoX, OLMo. |
| Continual learning and catastrophic forgetting | EWC and lifelong-learning surveys. |
| Small-data language learning | BabyLM, TinyStories. |
| Self-generated data methods | Self-Instruct, STaR, Self-Refine, Reflexion. |
| Verifiers and process supervision | GSM8K verifiers, process reward models. |
| Data curation and contamination | The Pile, Dolma, DataComp-LM, Open-Instruct. |
| Tokenizers | BPE, SentencePiece, byte-level subword systems. |
| Transparent open-model practice | Pythia, OLMo, OLMo 2, LLM360. |

[Research grounding](./research-grounding.md) holds the per-paper map and the
QuarkLM implication drawn from each cluster.

## How the map is used

Each mechanic in the ladder is constrained by the same boundary, so a research
reference can guide design without crossing into QuarkLM's training data.

- Papers, official project docs, and public repositories are studied as
  structure references.
- The codebase is compared against those references.
- The comparison becomes a versioned implementation requirement with declared
  acceptance evidence.

What the map explicitly forbids: copying outside code; importing pretrained
weights, tokenizers, embeddings, or datasets; using an external model as a
teacher, verifier, judge, reward model, or repair generator; and treating
retrieval, exact responders, generated candidates, or research notes as proof
that the neural weights learned a behavior.

## Implementation ladder

The map directs QuarkLM to build the self-improvement operating system before
adding another direct-answer objective mode. The work falls into three phases.

### Operating-system surfaces (v0.75–v0.80)

These versions install the auditable surfaces that every later screen depends
on. They are implemented and not in dispute.

| Version | Mechanic |
| --- | --- |
| v0.75 | Candidate quarantine artifacts and lifecycle states. See [Candidate quarantine](../operate/candidate-quarantine.md). |
| v0.76 | Deterministic closed-world verifier checks. See [Closed-world verifier](../operate/closed-world-verifier.md). |
| v0.77 | Recipe objects and constraint-first promotion gates. See [Training recipes](../operate/training-recipes.md). |
| v0.78 | Transformer experiment, artifact, trainer-utility, and objective-catalog surfaces. |
| v0.79 | Transformer model/config and checkpoint-metadata surfaces. |
| v0.80 | Transformer eval and checkpoint-load surfaces. |

### Branch-diversity repair attempts (v0.81–v0.104)

With the surfaces in place, the map drove a long sequence of guarded
direct-answer objectives aimed at one problem: multi-target eval profiles
collapse to too few predicted branch tokens. Each attempt is recorded as
versioned diagnostic evidence; most are implemented and rejected for promotion.

The arc moved in a consistent direction. Early objectives added profile
target-share and prompt-ownership pressure (v0.81–v0.83) but trained snapshots
still lost target-token coverage. Baseline-floor work (v0.84–v0.92) then made
coverage preservation an update-acceptance rule; under that floor every
attempted update was rejected, which proved the repair *shape* itself had to
change. Calibrated smaller updates (v0.93–v0.95) produced the first accepted
guarded source-profile movements that preserved the floor, and frontier and
coverage-recovery work (v0.96–v0.104) converted some of that safe movement into
local coverage gains while branch diversity stayed below the gate.

No screen in this phase promoted. Each one tightened what the next attempt was
allowed to claim.

### Memory rail and routing diagnostics (v0.105–v0.115)

v0.105 separated the memory rail from neural consolidation explicitly:
closed-world retrieval memory built a corpus-only `retrieval_memory_report.json`
with `497` memory cards and `219/219` exact retrieval evals, with no external
embeddings and no weight updates. That evidence is `memory-served`, not
`weight-consolidated`; it proves the corpus contains the answers, not that the
transformer learned to route them.

The versions that follow used that separation to guide gated consolidation and
then to diagnose why consolidation keeps failing.

| Version | Mechanic | Outcome |
| --- | --- | --- |
| v0.105 | Closed-world retrieval memory. | `497` cards, `219/219` exact retrieval, no weight updates. |
| v0.106 | Memory-guided consolidation planning. | Ranked `9` memory-served, neural-failed profiles into a target list. |
| v0.107–v0.111 | Gated memory-consolidation training under that plan. | Guarded acceptances on individual source profiles; retrieval stays exact at `219/219`; rejected on `branch_diversity_target`. |
| v0.112 | Branch-diversity root-cause diagnostics. | Critical `target_routing_gap` diagnosis. |
| v0.113 | Branch routing audit. | High output-bias escape risk, low representation separation across `9/9` profiles, `glossary` imbalance. |
| v0.114 | Logit-prior and centroid-separation instrumentation. | Dominant-token wins decompose as hidden-projection pressure across `9/9` profiles. |
| v0.115 | Hidden-projection margin candidate. | Lowers collapsed-token hidden advantage; still rejected on `branch_diversity_target`. |

The diagnosis is now precise: the failure is a routing gap, not missing loss
movement. [Branch diversity research](./branch-diversity-research.md) holds the
per-version evidence, and [Transformer screen history](../build/transformer-screen-history.md)
holds the full run-by-run log and objective names.

## Current gap

The operating-system surfaces are built and the routing failure is
instrumented down to hidden-projection pressure. What remains is a single
required mechanic:

- a broader guarded routing repair that uses the v0.115 hidden-projection
  evidence to improve zero-coverage and buried-target profiles, without
  regressing retrieval provenance and without relaxing the promotion gates.

Until that mechanic clears `branch_diversity_target`, the from-scratch
transformer stays unpromoted, and the docs say so plainly. See
[Transformer](../build/transformer.md) for the current status and evidence
table.

## Operating rule

Every future mechanics version must answer three questions before it is added
to the ladder:

1. Which research or implementation pattern justifies this mechanic?
2. Which closed-world boundary does it protect?
3. Which artifact proves it worked, or recorded the rejected run?

That is how QuarkLM keeps the claim clean. The model grows only from its
admitted, ledgered corpus, and the project keeps enough evidence to show
exactly what each version changed and what it did not.

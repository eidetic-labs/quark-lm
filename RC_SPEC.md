# QuarkLM Release Candidate Spec

Last reviewed: 2026-06-15.

This document defines what "release candidate" means for QuarkLM before the
project resumes versioned model work. It is intentionally stricter than "the
tests pass" and narrower than "a production language model."

## Release Candidate Tracks

QuarkLM has two different RC tracks. Mixing them is how the project drifts into
turning knobs.

| Track | Meaning | Current posture |
| --- | --- | --- |
| Research Prototype RC | The closed-world self-improvement system is reproducible, auditable, documented, and honest about what is and is not learned into weights. | Close, but needs a final readiness pass. |
| Language Model RC | The from-scratch transformer reliably answers from the admitted corpus without candidate crutches and passes neural promotion gates. | Not close yet; branch routing is the blocker. |

The responder/retrieval rails may be release-candidate quality before the
transformer is. A QuarkLM release must name that distinction plainly.

## Non-Negotiable Boundaries

An RC cannot violate the closed-world claim:

- no pretrained weights;
- no pretrained tokenizer;
- no external embeddings;
- no unledgered training text;
- no candidate or retrieval success counted as neural weight learning;
- every promoted behavior must be backed by an artifacted run;
- every rejected behavior must remain evidence rather than being overwritten;
- docs and marketing must distinguish promoted behavior from current research
  screens.

## Research Prototype RC Contract

A Research Prototype RC is ready when all of these are true:

| Area | RC requirement | Proof |
| --- | --- | --- |
| Corpus boundary | Training data is admitted or derived from admitted corpus artifacts only. | `training_plan.json`, `corpus_hygiene.json`, and `closed_world_verifier.json` pass. |
| Reproducibility | Runs declare hypothesis, data, gates, recipe, and decision. | `experiment_intent.json`, `training_recipe.json`, and metrics artifacts are present. |
| Promotion logic | Constraints are evaluated before quality metrics. | `constraint_first_promotion.json` exists and gates promotion. |
| Retrieval honesty | Retrieval memory can answer admitted facts, but is not counted as weight learning. | Retrieval reports and docs state the distinction. |
| Test surface | Python tests and site builds pass from a clean checkout. | `PYTHONPATH=src python3 -m unittest discover -s tests` and `npm run sites:build`. |
| Public docs | README, status, Docusaurus, and marketing align with the current state. | Shared state and rendered builds agree on latest promoted and latest screened evidence. |
| Contributor clarity | A new contributor can understand what is stable, what is experimental, and what not to claim. | `README.md`, `STATUS.md`, `GOAL.md`, and RC docs are current. |

Research Prototype RC does not require the transformer to pass
`branch_diversity_target`, but it must say that the transformer is not yet the
promoted language-model path.

## Language Model RC Contract

A Language Model RC is ready only when the from-scratch neural learner meets
all of these requirements:

| Area | RC requirement | Proof |
| --- | --- | --- |
| Branch routing | Multi-target profiles do not collapse to one dominant first token. | `branch_diversity_target` passes across required profiles. |
| Target coverage | Every multi-target profile covers its target first-token set above the configured floor. | Branch profile snapshots show target-token coverage floors met. |
| Representation separation | Prompt-to-branch states separate by target enough to support routing. | Representation profiles show acceptable centroid distance and margins. |
| Weight learning | Neural checkpoints improve because of admitted-corpus training, not retrieval lookup. | Promotion artifact links the checkpoint, data boundary, metrics, and constraints. |
| Direct answers | Greedy or otherwise approved direct-answer mode answers admitted evals without hidden candidate selection. | Direct-answer eval artifacts pass the selected exactness threshold. |
| Retention | New training does not erase prior admitted knowledge. | Forgetting and coverage audits pass against prior accepted evidence. |
| Unknown policy | The model handles unknowns according to the closed-world policy. | Unknown evals pass without hallucinated corpus claims. |
| Scale sanity | Runtime, context length, and architecture choices are documented and reproducible. | Config, checkpoint metadata, and run recipe are complete. |

The current transformer fails this contract because v0.115 still fails
`branch_diversity_target`.

## RC Naming Guidance

Use precise names:

- "QuarkLM Research Prototype RC" when releasing the audited closed-world
  learning system.
- "QuarkLM Language Model RC" only after the neural learner passes the LM
  contract.
- "Promoted responder evidence" for v0.42-style reliable corpus behavior.
- "Unpromoted transformer screen" for v0.115-style model diagnostics.

## Exit Criteria Before Resuming Versioned Model Work

Before v0.116 begins, the project should have an explicit decision for:

1. Which RC track is being pursued next.
2. Which gap blocks that track.
3. Which experiment bundle is allowed to address the gap.
4. Which metrics can reject the bundle early.
5. Which claims docs and marketing are allowed to make after the run.

Current decision: package the **Research Prototype RC** boundary first. When the
model loop resumes, use the **Profile-Balanced Routing Repair** bundle from
`RC_GAP_AUDIT.md`, with representation-separation acceptance checks.

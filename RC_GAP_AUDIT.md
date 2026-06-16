# QuarkLM RC Gap Audit

Last reviewed: 2026-06-15.

This audit compares the current repository against `RC_SPEC.md`. The goal is to
close the gap to a release candidate by making architectural decisions from
evidence, not by tuning one objective at a time.

## Current Evidence Baseline

| Evidence | Current result |
| --- | --- |
| Latest commit | `711cede Add hidden projection margin` |
| Promoted responder run | `runs/self-improve-v0.42/` |
| Promoted responder behavior | Exact responder, learned answer classifier, and generative answer decoder pass all 10 current eval sets. |
| Admission probes | Direct `48/48`, paraphrase `84/84`, glossary `38/38`. |
| Best no-candidate generator evidence | v0.31 auxiliary generator reached `219/219` exact without candidate selection. |
| Current transformer screen | `runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/` |
| v0.115 transformer result | `10/11` constraints pass; promotion blocked by `branch_diversity_target`. |
| v0.115 retrieval memory | Exact `219/219`; not counted as weight learning. |
| v0.115 hidden-projection signal | Average collapsed-token hidden advantage moved from about `0.0842` to `0.0736`. |
| v0.115 remaining failure | `9/9` multi-target profiles still collapse to `"n"`; `2` profiles have zero target-token coverage. |
| Latest verification | `272` Python tests passed; `npm run sites:build` passed before pause. |

## Track Readiness

| Track | Status | Reason |
| --- | --- | --- |
| Research Prototype RC | Near | The self-improvement operating system exists: corpus boundary, verifier, quarantine, recipes, promotion gates, docs, tests, and rejected evidence are all present. The final gap is packaging the claim honestly and ensuring contributor-facing docs are coherent. |
| Language Model RC | Not ready | The transformer does not yet route prompts to correct answer branches. v0.115 proves hidden projection is relevant but insufficient as a one-step single-batch repair. |

## Gap Register

| Priority | Gap | Evidence | Why it matters | Required proof |
| --- | --- | --- | --- | --- |
| P0 | Transformer branch routing collapse | v0.115 fails `branch_diversity_target`; all `9/9` multi-target profiles collapse to `"n"`. | A language model RC cannot rely on retrieval or candidate selection while the neural learner collapses first-token routing. | A screen where branch diversity passes or fails with a narrower, explained residual gap. |
| P0 | Representation separation is too weak | v0.113-v0.115 report low separation across `9/9` multi-target profiles. | Hidden projection margins cannot hold if prompts do not form separable states by target branch. | Representation profiles show target centroid distances and margins above a declared floor. |
| P0 | Repair objectives are too local | v0.115 lowers hidden advantage but does not change collapse outcome. | A single branch batch can move the measured surface without changing global profile behavior. | A bundle that applies routing pressure across profile-balanced replay, not one sampled batch. |
| P1 | RC claim boundary needs final packaging | Current docs are strong but spread across README, STATUS, GOAL, Learn docs, and research docs. | Contributors need to know what is release-candidate quality and what remains research. | `RC_SPEC.md`, `RC_GAP_AUDIT.md`, README, and docs agree on the RC track and claim boundary. |
| P1 | Transformer success metric hierarchy needs a hard floor | The project records loss, rank, coverage, exactness, and diversity, but the LM RC threshold is not yet declared in one place. | Without a threshold, every partial metric improvement can masquerade as progress. | A named LM RC gate profile with exact thresholds for diversity, coverage, direct answer exactness, and retention. |
| P1 | Architecture capacity decision is still implicit | The current transformer is tiny and character-level; failures may reflect objective design, capacity, tokenizer burden, or curriculum shape. | Continuing objective-only repairs may hide a model-capacity or curriculum bottleneck. | A controlled architecture/curriculum decision screen that holds data boundary and gates constant. |
| P2 | Contributor release mechanics are not audited as an RC package | Site hosting and build scripts exist, but a release checklist is not centralized. | Research Prototype RC should be easy for contributors to reproduce. | A release checklist with commands, expected artifacts, and non-claims. |

## Root-Cause Interpretation

The current failure is not "the model needs a slightly better loss." The best
interpretation is:

1. Retrieval memory can answer the closed world exactly.
2. The transformer can move measured quantities under guard.
3. The transformer's branch states are not yet separable enough for stable
   target routing.
4. Output-side repairs can reduce a dominant token's advantage without changing
   the global collapsed profile behavior.
5. Therefore, the next model step should be a coordinated routing-repair bundle,
   not another isolated direct-answer mode.

## Recommended Experiment Bundles

These are bundles, not knobs. Each bundle should have a hypothesis, allowed
data, planned artifacts, early rejection rules, and a promotion decision before
code changes begin.

### Bundle A: Profile-Balanced Routing Repair

Hypothesis: hidden-projection margin helps only when applied across a
profile-balanced replay surface that includes zero-coverage and buried-target
profiles in the same guarded update.

Components:

- target-balanced branch batches across all failing multi-target profiles;
- hidden-projection margin across branch targets;
- representation-separation pressure for same-target and different-target
  branch states;
- coverage-preserving baseline floor guard;
- branch-diversity score check before accepting the update.

Early rejection:

- any profile drops below baseline target-token coverage;
- dominant predicted rate remains `1.0` across all profiles after the planned
  guarded attempts;
- hidden advantage improves but target coverage stays unchanged.

### Bundle B: Representation-First Routing Screen

Hypothesis: QuarkLM needs separable prompt representations before output-head
repair can work.

Components:

- no output-bias updates;
- representation contrast by target token and profile;
- centroid-margin reporting as an acceptance metric;
- only after representation floors improve, run a small hidden-projection
  margin pass.

Early rejection:

- centroid distance and margin do not improve;
- target rank improves but target-token coverage remains zero for the same
  profiles;
- branch predictions diversify into wrong tokens without covering targets.

### Bundle C: Capacity and Curriculum Decision Screen

Hypothesis: the current tiny character-level transformer may be below the
minimum capacity or curriculum structure needed for stable branch routing.

Components:

- hold corpus and evals constant;
- compare current dim/context settings against one larger but still tiny
  from-scratch config;
- compare current character curriculum against a branch-first curriculum
  ordering;
- no pretrained tokenizer or external embeddings.

Early rejection:

- increased capacity improves loss but not branch diversity;
- branch-first curriculum memorizes current probes but weakens retention;
- runtime cost grows without crossing a declared routing threshold.

## Recommended Path To RC

1. Pursue **Research Prototype RC** first, because it is close and honest.
2. Do not call the transformer an LM RC until `branch_diversity_target` passes.
3. Before v0.116, choose one bundle above and write its experiment intent.
4. Prefer Bundle A if the goal is to continue from v0.115 evidence directly.
5. Prefer Bundle B if the team wants to prove representation separation before
   any more output-head movement.
6. Prefer Bundle C if the team suspects the current architecture is below the
   minimum viable capacity.

## Selected Path

Current decision: package the **Research Prototype RC** boundary first, then
resume the model loop with **Bundle A: Profile-Balanced Routing Repair**. Bundle
A should include the representation-separation acceptance checks from Bundle B.

This means v0.116 should not be a single new objective mode. It should be a
coherent routing repair screen with:

- profile-balanced branch batches across all failing multi-target profiles;
- hidden-projection margin pressure;
- representation-separation pressure or preflight metrics;
- coverage-preserving update guards;
- branch-diversity acceptance gates;
- early rejection when hidden advantage improves but target-token coverage does
  not.

## Immediate Non-Model Work

To close the Research Prototype RC gap without touching model mechanics:

- add an RC checklist that names commands, artifacts, and forbidden claims;
- decide whether `GOAL.md` restart handoffs are local-only or committed
  project history;
- ensure README and Docusaurus expose the RC track distinction;
- run a fresh clean-checkout verification before tagging any RC.

## Decision

QuarkLM is close to a Research Prototype RC and not close to a Language Model
RC. The fastest path is not more knob tuning. It is to lock the RC contract,
ship an honest research prototype when packaging is ready, and then run a
coherent routing-repair bundle for the transformer.

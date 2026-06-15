---
title: Deep Research Review
description: The v0.70 cross-referenced research and implementation-gap review for QuarkLM.
---

# Deep Research Review

Last reviewed: 2026-06-14.

The full review lives in the repository root at `DEEP_RESEARCH_REVIEW.md`.

v0.70 is a research checkpoint. The earlier forward plan identified the right
direction, but the project needed a deeper cross-reference pass before adding
more mechanics. This page summarizes the decision.

## Conclusion

QuarkLM should not keep adding direct-answer objectives until the learning loop
has the operating system around it:

- experiment registry;
- training recipes;
- corpus hygiene;
- candidate quarantine;
- deterministic closed-world verifier;
- extracted replay planner;
- constraint-first promotion;
- transformer module boundaries.

That is the path that keeps "I learned something new" from becoming
self-contamination. The phrase should mean proposed, quarantined, verified,
admitted, trained, evaluated, and promoted.

## Sources Cross-Referenced

The v0.70 review covers:

- continual-learning surveys for staged updates and catastrophic forgetting;
- replay systems such as Deep Generative Replay, Avalanche, and Reverb;
- self-generated-data methods such as Self-Instruct, STaR, Self-Refine, and
  Reflexion;
- self-feedback and self-judgment risks, including self-bias and reward
  hacking;
- verifiable-reward systems such as Tulu 3 and DeepSeek-R1;
- model-collapse studies on recursive synthetic training;
- small-model and data-centric work such as TinyStories, BabyLM, and SmolLM2;
- transparent open-model practice from Pythia, LLM360, OLMo, OLMo 2, and Dolma;
- implementation references from nanoGPT, minGPT, GPT-NeoX, LLM Foundry,
  LitGPT, Avalanche, Open-Instruct, and Hugging Face tokenizers.

All sources are design references only. QuarkLM still forbids pretrained
weights, pretrained tokenizers, external embeddings, external datasets, copied
code, and external-model-shaped training data.

## Current Codebase Gap

The strongest local finding is structural. QuarkLM has serious transformer
mechanics now, but `src/closed_world_lm/transformer_char_model.py` owns too
many responsibilities in one 9,494-line module: model, optimizer,
direct-answer objectives, replay planning, snapshot scoring, CLI parsing,
checkpoint writing, and run reporting.

The self-improvement path is cleaner. It already records prompt leakage,
forgetting, exact eval, promotion gates, corpus snapshots, corpus diffs,
attempt archives, and deterministic self-diagnosis. The transformer path needs
the same discipline before another major repair screen.

## Revised Sequence

1. **v0.71:** experiment registry and run-intent schemas. Implemented.
2. **v0.72:** standalone replay planner. Implemented.
3. **v0.73:** corpus hygiene and training-plan artifacts. Implemented.
4. **v0.74:** research implementation map. Implemented.
5. **v0.75:** candidate quarantine. Implemented.
6. **v0.76:** deterministic closed-world verifier. Implemented.
7. **v0.77:** recipes and constraint-first promotion gates. Implemented.
8. **v0.78:** transformer experiment/artifact surfaces, trainer utilities,
   and objective catalog. Implemented.
9. **v0.79:** transformer model/config and checkpoint metadata surfaces.
   Implemented.
10. **v0.80:** transformer eval/checkpoint-load surfaces. Implemented.
11. **v0.81:** profile target-share anti-collapse objective. Implemented.
12. **v0.82:** profile target-share full-stack screen. Implemented and
    rejected.
13. **v0.83:** prompt-specific branch ownership. Implemented and rejected.
14. **v0.84:** baseline replay anchors. Implemented and rejected.
15. **v0.85:** baseline-floor update gating. Implemented and rejected.
16. **v0.86:** adaptive baseline-floor retries. Implemented and rejected.
17. **v0.87:** baseline-floor repair retries. Implemented and rejected.
18. **v0.88:** objective-side baseline-floor anchors. Implemented and rejected.
19. **v0.89:** stabilization-only baseline-floor updates. Implemented and
    rejected.
20. **v0.90:** baseline-floor rejection diagnostics. Implemented.
21. **v0.91:** profile-targeted floor stabilization. Implemented and rejected.
22. **v0.92:** sequential source-profile floor stabilization. Implemented and
    rejected.
23. **v0.93:** calibrated sequential profile-floor stabilization. Implemented
    with one accepted guarded update; rejected for promotion.
24. **v0.94:** profile-scale calibrated floor stabilization. Implemented with
    eight accepted guarded source-profile updates; rejected for promotion.
25. **v0.95:** diversity-aware profile-scale acceptance. Implemented with
    five accepted score-improving source-profile updates; rejected for promotion.
26. **v0.96:** frontier-driven branch diversity under the full baseline
    target-token floor. Implemented with nine score-improving source-profile
    updates; rejected for promotion.
27. **v0.97:** coverage-frontier audited missing-target repair. Implemented
    with one coverage-gaining update; rejected for promotion.
28. **v0.98:** coverage-prep frontier repair. Implemented with six safe
    preparation moves; rejected for promotion.
29. **v0.99:** coverage-recovery frontier retry. Implemented with two recovery
    conversions; rejected for promotion.
30. **v0.100.0:** branch-stable coverage recovery. Implemented with two
    branch-stable recovery conversions; rejected for promotion.
31. **v0.101.0:** branch-diversity recovery. Implemented with five local
    branch-score refinements; rejected for promotion.
32. **v0.102.0:** collapsed-profile binding. Implemented with one targeted
    binding update and three remaining collapsed eval profiles; rejected for
    promotion.
33. **v0.103.0:** remaining-profile binding. Implemented with six prioritized
    remaining-profile acceptances and a learning coverage gain; rejected for
    promotion.
34. **v0.104.0:** owner/paraphrase residual binding. Implemented with six
    prioritized acceptances and protected-learning rejection evidence; rejected
    for promotion.
35. **v0.105.0:** closed-world retrieval memory. Implemented with a corpus-only
    retrieval report, `497` memory cards, and `219/219` exact retrieval evals
    without external embeddings or weight updates.
36. **v0.106.0:** memory-guided consolidation planning. Implemented with a
    consolidation plan that ranks `9` memory-backed neural failed profiles and
    names `owner`, `paraphrases`, and `glossary` as collapsed memory-backed
    priorities.
37. **v0.107.0:** gated memory-consolidation training. Implemented with a
    source-plan-consuming direct-answer mode that targets `owner`,
    `paraphrases`, and `glossary`, records `26` prioritized attempts, accepts
    `8`, rejects `18`, and still rejects promotion on `branch_diversity_target`.
38. **v0.108.0:** expanded memory-consolidation target window. Implemented with
    explicit source-label mapping for target-only profiles and a five-target
    screen for `owner`, `paraphrases`, `heldout`, `qa`, and `glossary`; still
    rejects promotion on `branch_diversity_target`.
39. **v0.109.0:** missing first-token memory-consolidation pressure.
    Implemented with plan-derived missing first-token target maps, `8`
    candidates, `22` attempts, `1` accepted guarded coverage-gain update, `21`
    rejections, `7` fallback acceptances, and exact `219/219` retrieval; still
    rejects promotion on `branch_diversity_target`.
40. **v0.110.0:** remaining-collapsed missing first-token targeting.
    Implemented with source-plan collapsed-profile enforcement, consumed
    targets `owner`, `paraphrases`, and `learning`, `6` candidates, `16`
    attempts, `1` accepted guarded coverage-gain update, `15` rejections, `5`
    fallback acceptances, and exact `219/219` retrieval; still rejects promotion
    on `branch_diversity_target`.
41. **v0.111.0:** profile-specific remaining-collapsed missing first-token
    pressure. Implemented with source-label-to-target-profile maps,
    `6` candidates, `18` attempts, `0` direct missing-token acceptances, `18`
    rejections, `6` fallbacks, `1` accepted profile-specific update shape, and
    exact `219/219` retrieval; still rejects promotion on
    `branch_diversity_target`.
42. **v0.112.0:** branch-diversity root-cause diagnostics. Implemented with
    external research grounding, root-cause taxonomy under
    `branch_diversity_target.root_cause`, `24` guarded missing-token attempts,
    `0` direct missing-token acceptances, `8` fallbacks, exact `219/219`
    retrieval, and a critical `target_routing_gap` diagnosis; still rejects
    promotion on `branch_diversity_target`.
43. **v0.113.0:** branch routing audit diagnostics. Implemented with
    `branch_routing_audit` under direct-answer snapshots, output-bias escape
    risk, prompt-to-branch representation separation, and target-imbalance
    summaries; still rejects promotion on `branch_diversity_target`.
44. **v0.114.0+:** use the v0.113 audit inside the closed-world lifecycle to
    instrument dominant-token logit priors and representation separation before
    choosing any guarded repair candidate.

## Operating Rule

No larger transformer screen should run without an experiment-intent artifact,
corpus plan, replay plan, verifier checks, and explicit promotion constraints.
Loss, rank, top-k, and NLL are useful metrics, but they are not promotion
criteria unless retention, leakage, unknown-policy, coverage, diversity, and
contamination gates pass first.

v0.71 satisfies the experiment-intent part of that rule for self-improvement
answer cycles and transformer answer-training screens. v0.72 satisfies the
replay-extraction part by moving replay planning to
`src/closed_world_lm/replay_plan.py`. v0.73 satisfies the corpus-hygiene and
training-plan part by writing `corpus_hygiene.json` and `training_plan.json`.
v0.74 adds the [Research implementation map](./research-implementation-map.md)
so the next mechanics are tied to sources, gaps, and acceptance evidence. v0.75
adds candidate quarantine artifacts and lifecycle state. v0.76 adds
deterministic closed-world verifier checks. v0.77 adds recipes and
constraint-first promotion. v0.78 adds transformer experiment/artifact
surfaces, trainer utilities, and an objective catalog. v0.79 adds transformer
model/config and checkpoint metadata surfaces. v0.80 adds transformer
eval/checkpoint-load surfaces. v0.81 adds balanced profile target-share
pressure to the preserving-deficit direct-answer objective. v0.82 screens that
objective, preserves coverage only by restoring step `0`, and rejects trained
snapshots that collapse QA and heldout branch diversity. v0.83 adds
prompt-specific sibling-target ownership margins, but the full screen still
restores step `0` because trained snapshots lose target-token coverage. v0.84
adds baseline replay anchors; trained snapshots avoid the v0.83 zero-coverage
collapse but still restore step `0` because coverage reaches only `0.125`
against the `0.25` floor. v0.85 adds baseline-floor update gating; it preserves
the floor by rejecting `50/50` unsafe attempted updates, so the next repair must
produce accepted updates under the guard. v0.86 adds adaptive baseline-floor
retries across learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01`; it
rejects all `200/200` attempted retry updates, showing step size alone is not
the missing mechanic and the next repair must change update shape under the
same baseline floor. v0.87 adds one bounded baseline-covered anchor repair
after each failed retry; it records `227` repair anchors and rejects all
`200/200` repaired attempts, showing post-update repair is also not the missing
mechanic. v0.88 moves balanced floor anchors into the objective itself; it runs
`200` objective anchor batches covering `2400` anchor records and still rejects
all `200/200` attempts, showing the combined branch-pressure objective is also
not the missing mechanic. v0.89 removes branch-diversity pressure and trains
only baseline-covered floor anchors; it runs `200` stabilization anchor batches
covering `2400` anchor records and still rejects all `200/200` attempts,
showing floor-only updates are also not the missing mechanic under the current
guard. The next repair should diagnose the guard/update interaction before
branch-diversity pressure is added back. v0.90 adds that diagnostic layer: the
screen records `200/200` rejected stabilization-shaped attempts, `50`
rejections at each adaptive scale, `heldout: 200` profile-floor violations, and
a worst rejected floor deficit of `0.25` on `learning`. v0.91 covers all `227`
baseline-covered floor anchors across `12` profile-target groups and still
rejects `200/200` profile-targeted attempts with the same violation pattern.
v0.92 tries sequential source-profile floor repair, rejects all `2000`
profile-local attempts, and records `200` no-effective-update outer attempts.
v0.93 adds calibrated sub-`0.01` scales plus coverage-only guard probes, accepts
one `bridge:owner` source-profile update at scale `0.0025`, and still rejects
promotion on branch diversity. v0.94 adds profile-scale memory, accepts `8`
source-profile updates across `60` profile-scale attempts, and still rejects
promotion on branch diversity. v0.95 adds diversity-aware profile-scale
acceptance, accepts `5` score-improving source-profile updates across `58`
profile-scale attempts, rejects `11` floor-preserving score regressions, and
still rejects promotion on branch diversity. v0.96 adds frontier target anchors,
accepts `9` score-improving source-profile updates across `43` profile-scale
attempts, lowers max dominant predicted rate to `0.9`, and still rejects
promotion on branch diversity. v0.97 adds coverage-frontier acceptance, accepts
`1` coverage-gaining source-profile update across `68` attempts, rejects `15`
coverage ties and `2` coverage regressions, and still rejects promotion on
branch diversity. v0.98 adds coverage-prep frontier acceptance, accepts `9`
source-profile updates across `43` attempts, separates `3` coverage gains from
`6` coverage-preparation moves, and still rejects promotion on branch
diversity. v0.99 adds coverage-recovery frontier retry, accepts `6`
source-profile updates across `54` attempts, converts `2` prepared candidates
into direct coverage recoveries, keeps `4` preparation fallbacks, and still
rejects promotion on branch diversity. v0.100.0 adds branch-stable
coverage-recovery acceptance, keeps the `2` recovery conversions, records `15`
branch-stability checks, rejects `1` retry for branch-score regression, and
still rejects promotion on branch diversity. v0.101.0 adds branch-diversity
recovery after safe profile updates, accepts `5` local branch-score
refinements, falls back once, and still rejects promotion on branch diversity.
v0.102.0 adds collapsed-profile binding after branch-diversity recovery,
accepts `1` targeted binding update, narrows final collapse from `9/9` eval
profiles to `3/9`, and still rejects promotion on branch diversity.
v0.103.0 adds remaining-profile binding after collapsed-profile binding,
records `21` prioritized attempts, accepts `6` prioritized updates, improves
`learning` coverage from `0.0` to `0.25`, preserves target coverage, and still
rejects promotion on branch diversity.
v0.104.0 adds owner/paraphrase residual binding, records `16` prioritized
attempts, accepts `6` prioritized updates, runs `75` learning-preservation
checks, rejects `24` preservation failures, keeps `learning` non-collapsed, and
still rejects promotion on branch diversity.
v0.105.0 adds corpus-only retrieval memory as an explicit non-parametric
evidence rail. The retrieval report builds `497` memory cards from the closed
corpus and answers `219/219` eval probes exactly with no external model,
embeddings, pretrained retriever, or weight updates. This is retrieval success,
not neural promotion; the transformer remains blocked on branch diversity.
v0.106.0 adds memory-guided consolidation planning, writes
`memory_consolidation_plan.json`, ranks `9` retrieval-served neural failures,
and identifies `owner`, `paraphrases`, and `glossary` as collapsed
memory-backed profiles. This creates a target list for gated training without
counting memory retrieval as weight learning.
v0.107.0 consumes that plan in a gated direct-answer mode, targets `owner`,
`paraphrases`, and `glossary`, records `26` prioritized attempts with `8`
acceptances and `18` rejections, keeps retrieval exact at `219/219`, and still
rejects promotion on `branch_diversity_target`.
v0.108.0 expands the consumed target window to `owner`, `paraphrases`,
`heldout`, `qa`, and `glossary`, adds explicit target-to-source label mapping,
keeps retrieval exact at `219/219`, and still rejects promotion on
`branch_diversity_target`.
v0.109.0 consumes the v0.108.0 plan, extracts missing first-token target maps,
runs a guarded missing-token consolidation phase with `8` candidates and `22`
attempts, accepts `1` coverage-gain update, keeps retrieval exact at `219/219`,
and still rejects promotion on `branch_diversity_target`. The next repair should
focus on the remaining collapsed memory-backed profiles: `owner`, `paraphrases`,
and `learning`.
v0.110.0 consumes that remaining-collapsed plan, requires
`collapsed_memory_backed_profiles`, targets only `owner`, `paraphrases`, and
`learning`, records `6` candidates and `16` missing-token attempts, accepts `1`
coverage-gain update, keeps retrieval exact at `219/219`, and still rejects
promotion on `branch_diversity_target`. The next repair should make the
missing-token pressure profile-specific rather than only plan-specific.

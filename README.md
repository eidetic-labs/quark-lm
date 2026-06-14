# QuarkLM

QuarkLM is a tiny research prototype for a model that only learns from an
explicitly admitted corpus. The intended GitHub repository slug is
`quark-lm`. The current Python import path remains `closed_world_lm` until a
dedicated package migration is promoted.

Tagline: Big idea. Tiny package.

The v0 experiment is intentionally modest:

- no pretrained weights
- no pretrained tokenizer
- no external embeddings
- no runtime dependencies outside the Python standard library for the model
  prototype
- character-level vocabulary learned from the admitted corpus
- a tiny neural character model initialized from random weights
- a tiny decoder-only transformer initialized from random weights

This is not yet a useful assistant. It is a first closed-world language model loop:
glossary, simple grammar, tiny stories, question-answer lessons, training, and
closed-world probes.

## Research Grounding

QuarkLM is closest to continual learning, lifelong pretraining, replay, and
self-improvement research, but applies those ideas under a stricter boundary:
no pretrained weights, no pretrained tokenizer, no external embeddings, and no
training text outside the admitted corpus. Self-generated text may propose
lessons, probes, or repairs, but it cannot become training data until it is
verified against admitted sources and included in a versioned curriculum.

The Docusaurus Learn section now includes a paper-backed research grounding page
that records what QuarkLM should adopt next, what should be deferred, and which
claims should stay framed as project goals rather than proven novelty.

## Latest Evidence

Current promoted run: `runs/self-improve-v0.42/`.
Current transformer answer-lesson run:
`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/`.

- v0.42 keeps the admitted corpus unchanged from v0.41 at `12` admitted facts.
- The v0.42 self-improvement cycle passed on archived `attempt-001`, with
  forgetting compared against `runs/self-improve-v0.41/`.
- Admission-probe audit passed: direct probes `48/48`, paraphrase probes
  `84/84`, no missing, extra, or mismatched ids.
- Glossary-probe audit passed: glossary probes `38/38`, no missing, extra, or
  mismatched ids.
- v0.42 keeps sparse prompt-contrast branch repair and widens the from-scratch
  transformer from embedding/feed-forward dimensions `4/8` to `8/16`. This is
  still random initialization with the corpus-trained character tokenizer.
- The current v0.42 direct-transformer run trained `80` target-loss steps plus
  `1000` sparse branch-repair/contrast direct answer steps at context size
  `32`. Direct answer target loss moved `3.4278 -> 2.2708`, transformer answer
  target NLL moved `3.5850 -> 2.4129`, and eval-scoped transformer-only
  candidate accuracy moved `15/219 -> 37/219`.
- Raw direct greedy transformer exact remained `0/219 -> 0/219`; completions
  moved from the repeated `"te"`/`"e"` loop to the short wrong answer `" te."`.
  v0.42 improves the scored target distribution without damaging candidate
  discrimination and reduces runaway looping, but still shows that raw greedy
  answer emission needs stronger prompt-conditioned representation.
- Unpromoted v0.43 experiments added a faster final-position transformer
  forward pass, explicit prompt context-coverage metrics, and an experimental
  hard-negative branch-contrast mode. The hard-negative context-32 run
  (`runs/transformer-answer-v0.43-hard-branch-contrast4-dim8-context32/`) kept
  candidate accuracy at `37/219` but regressed direct loss to `2.4225`, answer
  NLL to `2.5402`, and collapsed greedy output to a repeated `" a"` loop. The
  context-80 run
  (`runs/transformer-answer-v0.43-branch-repair-contrast50-dim8-context80/`)
  achieved full semantic template coverage (`219/219`) and the short `" t."`
  failure, but still trailed v0.42 with direct loss `2.3122` and answer NLL
  `2.4546`. A longer context-80 1500-step run reached `38/219` candidates but
  regressed loss/NLL and greedy output, so v0.42 remains the promoted
  transformer checkpoint. A layer-normalized context-80 screen
  (`runs/transformer-answer-v0.43-layernorm-screen-dim8-context80/`) also kept
  `37/219` candidates with full context coverage, but regressed answer NLL to
  `2.5881` and produced repeated `" y"`/`"e"` loops, so it was not promoted.
  A branch-span screen
  (`runs/transformer-answer-v0.43-branch-span3-screen-dim8-context32/`) tested
  repairing answer positions `1..3`; it preserved `37/219` candidates but
  regressed answer NLL to `2.7426` and produced a long `"neeee"` loop, so it
  also remains unpromoted evidence.
  Multi-layer transformer support was added after that, but the first
  two-layer context-32 screen
  (`runs/transformer-answer-v0.43-two-layer-screen-dim8-context32/`) was
  interrupted before final direct-answer metrics because the full-block scalar
  autograd path was too slow for the regular loop. It produced only partial
  JSONL history and is runtime evidence, not promotion evidence. A follow-up
  optimized the final transformer layer to compute only the last state and
  proved equivalence against full-stack logits, but
  `runs/transformer-answer-v0.43-two-layer-finalopt-screen-dim8-context32/` was
  still interrupted before final metrics; the intermediate full-state layer is
  still too expensive for direct-answer repair updates. A follow-up added
  top-layer-only direct-answer training for stacked models plus an explicit
  `--skip-post-direct-snapshot` screening control. The completed bounded screen
  at
  `runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/`
  saved a two-layer checkpoint after `40` target-loss steps and `80` top-layer
  direct steps, recorded that the post-direct candidate snapshot was skipped,
  moved direct-answer target loss `3.5186 -> 3.2436`, and still failed direct
  greedy exact at `0/219 -> 0/219` with repeated `"a"` output. This is loop
  completion and runtime evidence, not promotion evidence. Direct-answer
  snapshots now also record branch profiles from QuarkLM's own logits. The
  smoke run at
  `runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` shows
  the QA branch-position-1 prediction collapsed from all `"o"` choices at
  baseline to all `"y"` choices after five direct updates, with branch accuracy
  `1/8` and a negative average target margin. That is diagnostic evidence for
  prompt-independent branch collapse, not a promotion candidate. Branch-collapse
  repair then used the dominant branch token as the unlikelihood negative. The
  full-dose smoke
  `runs/transformer-answer-v0.43-branch-collapse-smoke-dim4-context16/`
  regressed loss and moved collapse to all `"a"` predictions; the periodic
  smoke
  `runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/`
  improved direct loss `3.5800 -> 3.5157` but still stayed at QA branch
  accuracy `1/8` and collapsed to all `"n"` predictions. The lesson is that
  dominant-token suppression alone is not enough to create prompt-specific
  branches. Branch-batch contrast then trained several distinct branch targets
  in one update. The full-dose smoke
  `runs/transformer-answer-v0.43-branch-batch-smoke-dim4-context16/` improved
  loss only slightly and collapsed to all `"y"` predictions; the periodic smoke
  `runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/`
  improved direct loss `3.5800 -> 3.5248` but regressed QA branch accuracy
  `1/8 -> 0/8` and collapsed to all `"a"` predictions. The current evidence
  points at weak prompt representation rather than insufficient branch-token
  suppression.
- A representation-side v0.43 experiment added `--use-context-mean`, which
  adds the mean-pooled prompt context to the final transformer hidden state.
  The context-mean branch-batch smoke
  `runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/`
  improved direct loss `3.5805 -> 3.5252`, and the context-mean branch-repair
  smoke
  `runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/`
  improved direct loss `3.5805 -> 3.5310`. Both screens regressed QA branch
  accuracy `1/8 -> 0/8` and collapsed to all `"a"` predictions, so simple
  prompt averaging is rejected representation evidence rather than a promotion
  candidate.
- A follow-up representation experiment added `--use-context-projection`, a
  zero-initialized trainable projection of the mean-pooled context. It starts
  equivalent to the baseline and lets QuarkLM's own training decide whether a
  prompt summary should influence the final hidden state. The branch-repair
  smoke
  `runs/transformer-answer-v0.43-context-projection-branch-repair-smoke-dim4-context16/`
  moved all `20` projection parameters, improved direct loss
  `3.5802 -> 3.5217`, and the branch-batch smoke
  `runs/transformer-answer-v0.43-context-projection-branch-batch-smoke-dim4-context16/`
  improved direct loss `3.5802 -> 3.5252`. Both screens still regressed QA
  branch accuracy `1/8 -> 0/8` and collapsed to all `"a"` predictions, so a
  learned context projection is also rejected representation evidence.
- A stronger representation probe added `--use-prompt-attention-summary`, a
  learned attention-pooled context summary with a zero-initialized output
  projection. The branch-repair smoke
  `runs/transformer-answer-v0.43-prompt-attention-branch-repair-smoke-dim4-context16/`
  moved all `20` zero-initialized output projection parameters and improved
  direct loss `3.5802 -> 3.5217`; the branch-batch smoke
  `runs/transformer-answer-v0.43-prompt-attention-branch-batch-smoke-dim4-context16/`
  improved direct loss `3.5802 -> 3.5252`. Both screens still regressed QA
  branch accuracy `1/8 -> 0/8` and collapsed to all `"a"` predictions, so
  trainable prompt attention is also rejected representation evidence.
- A branch-context coverage diagnostic now records the exact context text,
  semantic feature coverage, context collisions, and target-token ambiguity at
  the direct-answer branch position. The context-16 smoke
  `runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context16/`
  showed QA branch contexts had `0/8` semantic coverage and `4` ambiguous
  branch windows. The context-32 smoke
  `runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context32/`
  removed QA ambiguity but still had `0/8` semantic coverage. The tiny
  context-80 smoke
  `runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context80/`
  reached `219/219` semantic coverage across all eval sets with zero ambiguous
  branch contexts. This points the next transformer work toward efficient
  longer-context branch repair rather than another context-16 objective.
- A branch-context gate now makes that diagnostic actionable for direct-answer
  screens. When `--direct-answer-require-branch-context-gate` is set, training
  is skipped unless branch contexts have complete semantic coverage, no
  ambiguous target-token contexts, and no skipped records. The context-16 gate
  smoke
  `runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context16/`
  requested `5` direct steps but ran `0` because the gate failed. The
  context-80 gate smoke
  `runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context80/`
  passed the gate and ran the requested `1` direct step.
- Direct-answer snapshots now support an explicit `branch-only` mode for
  bounded longer-context screens. It skips greedy completion evals in the JSONL
  snapshots while retaining branch profiles, branch-context coverage, and the
  branch-context gate result. The gated context-80 screen
  `runs/transformer-answer-v0.43-branch-context-gated-branchonly-smoke-dim4-context80/`
  passed the required gate across all `219/219` semantic records, ran all `5`
  requested branch-repair direct steps, and recorded `evals_skipped: true`.
  This is screening efficiency evidence, not promoted model quality evidence.
  Two dim8 context-80 follow-up screens used that mode to test the best prior
  sparse repair/contrast policy and branch-batch contrast without paying for
  full greedy snapshots. The periodic repair/contrast screen
  `runs/transformer-answer-v0.43-branchonly-periodic-repair-contrast50-dim8-context80/`
  ran `100/100` direct steps and reduced interval train loss
  `6.7890 -> 6.4326`, but final QA branch prediction collapsed to all `"a"`.
  The branch-batch screen
  `runs/transformer-answer-v0.43-branchonly-branch-batch-dim8-context80/`
  ran `50/50` direct steps and reduced interval train loss `3.4614 -> 3.1976`,
  but also collapsed QA branch prediction to all `"a"`. Neither screen earns a
  full promotion snapshot; the next transformer step needs a stronger
  prompt-specific branch signal, not another loss-only branch objective.
- Branch profiles now include a structured `diversity` summary and snapshots
  include a `branch_diversity_target` across multi-target eval profiles. The
  context-80 smoke
  `runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/`
  passed the branch-context gate but failed branch diversity across all `9`
  multi-target profiles. QA ended with `target_unique: 8`,
  `predicted_unique: 1`, dominant predicted token `"r"` at rate `1.0`, and
  target-token coverage `0.125`. This makes prompt-specific branch diversity an
  explicit screen target before any full greedy-eval promotion snapshot.
- A new `branch-diversity-unlikelihood` mode now trains each distinct branch
  target while also penalizing the model's current wrong prediction for that
  branch context. The first context-80 smoke
  `runs/transformer-answer-v0.43-branch-diversity-train-smoke-dim4-context80/`
  passed the branch-context gate and ran `10/10` direct steps, but still failed
  the diversity target across all `9` multi-target profiles. QA moved from all
  `"x"` predictions to all `"b"` predictions, with target-token coverage
  `0.0 -> 0.125` and `predicted_unique` still `1/8`, so it is rejected
  training-mode evidence rather than a promotion candidate.
- Direct-answer training can now freeze the transformer output bias with
  `--direct-answer-freeze-output-bias`, preventing a branch screen from solving
  loss by moving one global token bias. The context-80 freeze-bias smoke
  `runs/transformer-answer-v0.43-branch-diversity-freezebias-smoke-dim4-context80/`
  passed the branch-context gate and ran `50/50` direct steps, but still failed
  branch diversity across all `9` multi-target profiles. QA moved from all
  `"x"` predictions to all `"w"` predictions, final target-token coverage was
  `0.0`, and `predicted_unique` stayed `1/8`. This rejects output-bias-only
  collapse as the full explanation.
- `branch-target-softmax-unlikelihood` adds a restricted softmax over the
  distinct branch targets in each batch, forcing the correct target to compete
  directly against the other observed target tokens. The frozen-output-bias
  context-80 smoke
  `runs/transformer-answer-v0.43-branch-target-softmax-freezebias-smoke-dim4-context80/`
  ran `50/50` direct steps and moved composite train loss
  `5.6671 -> 5.5820`, but final branch diversity still failed across all `9`
  multi-target profiles. QA briefly reached `predicted_unique: 2` at step `20`,
  then collapsed back to all `"w"` by step `50`, so target-set competition is
  rejected as a standalone repair.
- `--direct-answer-restore-best-branch-snapshot` now preserves the best
  measured branch-diversity checkpoint before final metric writing. The
  target-softmax restore-best smoke
  `runs/transformer-answer-v0.43-branch-target-softmax-restorebest-smoke-dim4-context80/`
  restored the final checkpoint from step `40` after `50/50` direct steps. It
  improved final QA target-token coverage from the prior all-`"w"` final
  coverage `0.0` to all-`"u"` coverage `0.125`, but `predicted_unique` stayed
  `1/8` and the diversity target still failed across all `9` multi-target
  profiles.
- `--use-prompt-prefix-projection` adds a zero-initialized trainable projection
  of non-padding prompt-prefix positions before the final answer token. The
  context-80 target-softmax restore-best screen
  `runs/transformer-answer-v0.43-prompt-prefix-target-softmax-restorebest-smoke-dim4-context80/`
  moved all `20` prompt-prefix projection parameters and lowered composite
  loss `5.6649 -> 5.5679`, but it restored from step `40` to the same all-`"u"`
  QA collapse with target-token coverage `0.125` and `predicted_unique` still
  `1/8`.
- `--use-prompt-position-projection` strengthens that idea with a separate
  trainable projection for each context position before the final answer token.
  The matching context-80 screen
  `runs/transformer-answer-v0.43-prompt-position-target-softmax-restorebest-smoke-dim4-context80/`
  moved `1108/1284` prompt-position projection parameters and again lowered
  composite loss `5.6649 -> 5.5679`, but the final restored branch profile
  stayed all `"u"` with target-token coverage `0.125` and `predicted_unique`
  `1/8`.
- `branch-target-margin-unlikelihood` adds a smooth pairwise target-margin loss
  over each batch's distinct branch targets. The prompt-position context-80
  screen
  `runs/transformer-answer-v0.43-branch-target-margin-prompt-position-smoke-dim4-context80/`
  ran `50/50` direct steps, moved train loss `4.8973 -> 4.7784`, and moved
  `1108/1284` prompt-position projection parameters, but the restored final
  profile stayed all `"u"` with target-token coverage `0.125`,
  `predicted_unique` `1/8`, and diversity failure across all `9` multi-target
  profiles.
- Direct-answer snapshots now include `branch_representation_profiles`, which
  measure hidden-state distance between branch contexts before the output head.
  The high-weight representation-contrast screen
  `runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim4-context80/`
  used `branch-representation-contrast-unlikelihood` with
  `--direct-answer-contrast-weight 50.0`; QA different-target hidden distance
  moved only about `0.00097 -> 0.00107` at the restored checkpoint, and the
  final branch profile still collapsed to all `"u"` with `predicted_unique`
  `1/8`.
- A dim-8 capacity screen
  `runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim8-context80-steps40/`
  completed `40/40` direct steps after the matching 50-step dim-8 screen proved
  too slow for the regular loop. It doubled the restored QA different-target
  hidden distance to about `0.00209`, but final QA still restored to all `"u"`
  with target-token coverage `0.125`, `predicted_unique` `1/8`, and diversity
  failure across all `9` multi-target profiles.
- `--prompt-position-projection-scale` now allows bounded screens to amplify
  the prompt-position projection residual without changing the closed-world
  data boundary. The scale-32 representation-contrast smoke
  `runs/transformer-answer-v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/`
  moved `1108/1284` prompt-position projection parameters and briefly pushed
  QA different-target hidden distance as high as about `0.4115` before
  restore. The best restored checkpoint came from step `40`, with QA hidden
  distance about `0.01235`, but final QA still collapsed to all `"u"` with
  target-token coverage `0.125`, `predicted_unique` `1/8`, and diversity
  failure across all `9` multi-target profiles. This rejects "the prompt signal
  is merely too quiet" as a complete explanation.
- The next transformer work should pause objective churn for an open-source
  structure audit. `STRUCTURE_AUDIT.md` records the allowed boundary: study
  model/trainer/tokenizer/checkpoint patterns from projects such as minGPT,
  nanoGPT, LitGPT, Hugging Face tokenizers, and LLM360, but never import their
  pretrained weights, tokenizers, embeddings, datasets, or text. The completed
  comparison table points the next implementation target at an opt-in
  pre-layer-norm transformer block path with final normalization before another
  branch-loss objective is promoted.
- `--use-pre-layer-norm` now adds that opt-in GPT-style block path and applies
  final normalization before the output head. The context-80 smoke
  `runs/transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
  passed focused tests, ran `50/50` direct steps, moved `1108/1284`
  prompt-position projection parameters plus all `8` final-norm parameters,
  and lowered interval train loss to `43.8918` at step `50`. It still failed
  the branch-diversity target across all `9` multi-target profiles, and QA
  stayed collapsed to all `"y"` with target-token coverage `0.125`. Unlike the
  prior all-global-token screens, `7/9` profiles were no longer fully
  collapsed, so the structural path is useful partial evidence but not a
  promotion.
- `branch-balanced-representation-contrast-unlikelihood` now tests a
  target-balanced branch batch sampler so repeated first-answer tokens cannot
  crowd rare branch targets out of a repair batch. The matching pre-layer-norm
  screen
  `runs/transformer-answer-v0.44-target-balanced-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
  ran `50/50` direct steps, but every trained snapshot scored worse than the
  baseline, so best-snapshot restoration returned to step `0`. The restored
  final state collapsed all `9/9` multi-target profiles to `"n"` and QA stayed
  `predicted_unique` `1/8`. Target-balanced sampling alone is rejected.
- Direct-answer branch profiles now include target-rank diagnostics: average
  target rank, top-3/top-5 target coverage, and the top predicted alternatives
  for failed branch records. The v0.45 branch-rank diagnostic smoke at
  `runs/transformer-answer-v0.45-branch-rank-diagnostic-smoke-dim4-context80/`
  used the pre-layer-norm prompt-position path and recorded QA and heldout both
  collapsed to `"n"` with average target rank `14.25` and top-3/top-5 coverage
  `0.125`. This shows the correct target is usually buried, not merely a
  near-miss behind the dominant token, so the next repair should target
  prompt-to-answer binding through the output head.
- `branch-output-binding-unlikelihood` combines branch target softmax with
  representation contrast in one direct-answer update, and best branch snapshot
  scoring now uses target-rank/top-k evidence before generic wrong-token
  diversity. The bounded v0.46 screen
  `runs/transformer-answer-v0.46-output-binding-rankscore-smoke-dim4-context80/`
  ran `20/20` direct steps with output bias frozen. It improved QA average
  target rank from `17.375` to `14.125` and raised QA/heldout top-5 coverage to
  `0.25`, but target-token coverage stayed `0.0`, top-3 coverage ended `0.0`,
  and both profiles still collapsed to wrong `"l"`/`"j"` predictions. This is
  rejected repair evidence, with rank-aware restore kept as a guardrail.
- `branch-rank-margin-unlikelihood` now trains each branch target against the
  model's own top wrong tokens, using `--direct-answer-hard-negatives` as the
  hard-negative count. The longer bounded v0.47 screen
  `runs/transformer-answer-v0.47-rank-margin-steps50-smoke-dim4-context80/`
  ran `50/50` direct steps, restored the rank-aware best snapshot from step
  `40`, lowered train loss `7.3649 -> 6.1629`, and improved QA average target
  rank `17.375 -> 9.0`. QA target-token coverage rose to `0.125`, top-3
  coverage rose to `0.25`, and top-5 coverage rose to `0.5`. This is the
  strongest branch-rank movement so far, but it is still rejected for promotion
  because predicted diversity stayed `1/8` and the branch remained collapsed to
  wrong `"n"`.
- `branch-balanced-rank-margin-unlikelihood` combines target-balanced branch
  batches with the same top-wrong-token rank margin. The v0.48 screen
  `runs/transformer-answer-v0.48-balanced-rank-margin-smoke-dim4-context80/`
  ran `50/50` direct steps and reached QA predicted diversity `2/8`, QA
  target-token coverage `0.125`, average target rank `9.375`, top-3 coverage
  `0.375`, and top-5 coverage `0.5`. It improved prompt-specific wrong-token
  diversity versus v0.47, but it is still rejected because QA and heldout remain
  wrong top-1 branch choices.
- The v0.49 top-one hard-negative screen
  `runs/transformer-answer-v0.49-balanced-rank-margin-top1-smoke-dim4-context80/`
  kept the balanced rank-margin path but reduced `--direct-answer-hard-negatives`
  from `5` to `1`, concentrating all margin pressure on the current top wrong
  token. It restored the best snapshot from step `10`; QA target-token coverage
  stayed `0.125`, but average target rank regressed to `12.5`, top-3 coverage
  fell to `0.125`, and top-5 coverage fell to `0.25`. This rejects "just focus
  harder on one wrong token" as the next top-1 repair.
- `branch-balanced-topk-softmax-unlikelihood` adds a restricted softmax over
  each branch target plus the model's current top wrong tokens, using
  `--direct-answer-hard-negatives` as the candidate count and
  `--direct-answer-contrast-weight` as the candidate-softmax weight. The v0.50
  screen
  `runs/transformer-answer-v0.50-balanced-topk-softmax-w5-smoke-dim4-context80/`
  restored the best branch snapshot from step `40`. QA average target rank
  improved from `17.375` to `8.75`, target-token coverage stayed `0.125`,
  top-3 coverage reached `0.375`, and top-5 coverage reached `0.5`. This is
  stronger than the v0.49 top-one screen and roughly matches v0.48 top-k
  coverage with a tighter target margin, but prediction diversity still stayed
  `1/8` with wrong `"u"` top-1 branch choices, so it remains rejected repair
  evidence.
- v0.51 implements a full transformer foundation stack before the next
  direct-answer repair run: dependency-free AdamW/SGD optimizer state,
  gradient accumulation, warmup/decay scheduling, checkpoint resume validation,
  v2 checkpoint metadata, multi-head attention, RMSNorm, gated MLPs, tied output
  embeddings, rotary-position support, cache-aware generation metadata,
  top-k/top-p/temperature/repetition controls, token-level generation traces,
  and replayable eval sample JSONL. The tiny all-switch smoke
  `runs/transformer-v0.51-foundation-stack-smoke/` ran `2/2` language-model
  steps with AdamW and wrote `optimizer_state.json`, a
  `quarklm-transformer-v2` checkpoint, `eval.json`, and
  `eval_samples.jsonl`. This is mechanics evidence, not model-quality
  promotion evidence.
- The v0.52 full-stack top-k screen
  `runs/transformer-answer-v0.52-fullstack-topk-softmax-smoke-dim4-context80/`
  reran `branch-balanced-topk-softmax-unlikelihood` with AdamW, gradient
  accumulation, two attention heads, RMSNorm, gated MLPs, tied output
  embeddings, rotary positions, cache-aware metadata, and prompt-position
  projection. The full-stack baseline had better wrong-token diversity than
  v0.50, with QA and heldout predicted diversity `3/8` and target-token
  coverage `0.25`, but direct training collapsed to one wrong `"a"` token and
  best-snapshot restore returned to step `0`. This rejects reusing top-k
  pressure unchanged under the new stack and points the next repair toward
  bidirectional prompt-to-token binding.
- v0.53 adds `branch-balanced-bidirectional-binding-unlikelihood`, which trains
  branch targets in two directions: each prompt context should choose its own
  target, and each target token should concentrate probability mass on its own
  prompt contexts. The focused transformer test suite now covers that
  context-ownership signal. The full-stack screen
  `runs/transformer-answer-v0.53-fullstack-bidir-binding-smoke-dim4-context80/`
  completed `50/50` direct steps and restored the best branch snapshot from
  step `40`. QA average target rank improved to `7.875` and top-5 coverage
  reached `0.5`, but target-token coverage ended at `0.125` and the diversity
  target still failed `0/9` multi-target profiles. This is useful rank-pressure
  evidence, not promotion evidence; the next repair should preserve target
  coverage while converting rank lift into top-1 branch choices.
- v0.54 adds `branch-balanced-coverage-binding-unlikelihood`, which makes each
  branch target compete against both sibling branch targets and the current hard
  wrong tokens while also training target-set mass as a coverage guard. The
  focused transformer test suite covers that hard-wrong-token coverage signal.
  The full-stack screen
  `runs/transformer-answer-v0.54-fullstack-coverage-binding-smoke-dim4-context80/`
  completed `50/50` direct steps, but best-snapshot scoring restored step `0`.
  Training snapshots improved QA rank as far as `8.125`, but target-token
  coverage collapsed to `0.0` with one wrong `"a"` top-1 branch token. This
  rejects the bundled coverage-binding loss under the full stack and points the
  next repair toward stronger coverage preservation before exact-target
  sharpening.
- v0.55 isolates that idea with
  `branch-balanced-target-set-coverage-unlikelihood`: it trains target-set mass
  against hard wrong tokens without exact-target row loss or cross-context
  ownership, and the screen turns off positive target CE. The focused
  transformer test suite covers the target-set-only signal. The full-stack
  screen
  `runs/transformer-answer-v0.55-fullstack-target-set-coverage-smoke-dim4-context80/`
  again completed `50/50` direct steps and restored step `0`. Training improved
  QA average target rank to `10.0`, but target-token coverage still collapsed
  to `0.0` with the same wrong `"a"` top-1 branch token. This rejects
  batch-local target-set mass as a sufficient coverage objective; the next
  repair should add explicit anti-collapse pressure over predicted target
  tokens.
- The v0.31 no-candidate auxiliary generator remains the best exact
  no-candidate answer evidence: it trained for `80000` weighted steps at
  learning rate `0.035` and moved exact generation from `0/219 -> 219/219` with
  `uses_answer_candidates: false`.
- The transformer uses QuarkLM's existing corpus-trained character tokenizer,
  learned token and position embeddings, one causal self-attention block, a
  feed-forward block, and a next-character language-model head.
- v0.23 introduced attempt archives. A failed gate remains preserved at
  `runs/self-improve-v0.23/attempts/attempt-001/`, while the repaired passing
  attempt remains preserved at `runs/self-improve-v0.23/attempts/attempt-002/`.
- Direct and paraphrase admission probes are generated from
  `corpus/admissions.jsonl` and audited in the run report.
- Self and learning evals expanded to `7/7` and `4/4`, including questions about
  self-diagnosis source, external model shaping, and repair-action selection.
- Forgetting audit passed against `runs/self-improve-v0.41/`.
- Protected prompt leakage audit passed.
- Learned eval summaries now retain `failed_records` so failed cycles are
  diagnosable from report artifacts.
- Self-improvement runs now include an exact eval audit and a promotion gate;
  the command exits non-zero unless generated probes, prompt leakage, forgetting,
  and exact eval audits all pass.
- Self-improvement reports now include a rule-based `self_diagnosis` section
  that uses no external model and recommends the next action from the report.
- Responder, learned answer classifier, and generative answer decoder reached
  100% exact match across QA, unknowns, held-out, paraphrases, ownership, self,
  learning, admissions, admission-paraphrase, and glossary evals.

## Quick Start

Run from this directory:

```bash
PYTHONPATH=src python3 -m closed_world_lm.curriculum --output build
PYTHONPATH=src python3 -m closed_world_lm.train --steps 300 --run runs/smoke
PYTHONPATH=src python3 -m closed_world_lm.evaluate --checkpoint runs/smoke/checkpoint.json
PYTHONPATH=src python3 -m closed_world_lm.respond --eval --json runs/smoke/respond.json
PYTHONPATH=src python3 -m closed_world_lm.answer_model train --run runs/answer-smoke
PYTHONPATH=src python3 -m closed_world_lm.answer_model eval \
  --checkpoint runs/answer-smoke/answer_model.json \
  --json runs/answer-smoke/answer_eval.json
PYTHONPATH=src python3 -m closed_world_lm.answer_decoder train --run runs/decoder-smoke
PYTHONPATH=src python3 -m closed_world_lm.answer_decoder eval \
  --checkpoint runs/decoder-smoke/answer_decoder.json \
  --json runs/decoder-smoke/decoder_eval.json
PYTHONPATH=src python3 -m closed_world_lm.self_improve answer-cycle \
  --run runs/self-improve-smoke
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 20 \
  --context-size 8
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model answer-train \
  --run runs/transformer-answer-smoke \
  --steps 100 \
  --eval-every 0 \
  --candidate-scope eval \
  --selector-steps 200 \
  --selector-eval-every 0 \
  --selector-emit-completions \
  --generator-steps 400 \
  --generator-eval-every 0 \
  --direct-answer-steps 100 \
  --direct-answer-eval-every 0 \
  --direct-answer-mode periodic-branch-repair-contrast-unlikelihood \
  --direct-answer-negative-weight 1.0 \
  --direct-answer-positive-weight 1.0 \
  --direct-answer-contrast-weight 1.0 \
  --direct-answer-branch-position 1 \
  --direct-answer-rollout-interval 50 \
  --embedding-dim 8 \
  --feedforward-dim 16
```

The `closed_world_lm` module name is still the stable command path during this
phase. Installed console scripts also include `quark-lm-*` aliases while the
older `closed-world-*` aliases remain available.

The short training run is a smoke test, not a quality target. Increase
`--steps` once you want to watch the toy model absorb the nursery corpus.
Each training run writes `metrics.jsonl`, including step `0` as the untrained
baseline and periodic probe losses for known, unknown, and held-out questions.
Evaluation reports both free-form exact match and closed-world candidate
accuracy. Candidate accuracy asks whether the model assigns the lowest loss to
the correct admitted answer among known corpus answers.

`closed_world_lm.respond` is a reliable corpus-only response model. It parses
the admitted `fact:` lines from `build/train.txt` and answers known, unknown,
and held-out probes exactly. This gives the project a grounded teacher while the
neural character model improves.

`closed_world_lm.answer_model` is a learned closed-world response model. It uses
randomly initialized softmax weights and trains on question/answer lessons
derived from admitted corpus facts. It records step-0 baseline metrics,
generated lessons, a checkpoint, and evaluation reports.

`closed_world_lm.answer_decoder` is a generative learned response model. It
updates random weights to emit answer characters one at a time from the prompt
and prior answer prefix, using only corpus-derived lessons. Its trainer uses a
weighted training pool so longer self-improvement lessons get enough updates
without changing the admitted lesson corpus.

`closed_world_lm.transformer_char_model` is an experimental decoder-only
transformer language model. It is built in the Python standard library with a
tiny scalar autodiff engine, starts from random weights, uses the corpus-trained
character tokenizer, and writes its own checkpoint and metrics. The
`answer-train` subcommand trains on corpus-derived `AnswerExample` lessons. It
can also train a closed-world answer candidate selector with random weights
using `--selector-steps`. Add `--selector-emit-completions` to record the
selector's chosen candidate as the emitted completion for exact-match evidence.
Add `--generator-steps` to train the transformer-guided character answer
generator without answer candidates. Selector-assisted emission,
generator-emitted answers, transformer-only NLL, and raw transformer free-form
generation stay separate in metrics. Add `--direct-answer-steps` to continue
updating the transformer weights for strict greedy answer completion with an
admitted terminator token; `--direct-answer-mode first-error` targets the first
current greedy mismatch, `first-error-unlikelihood` also penalizes the wrong
self-predicted token, `rollout-unlikelihood` trains on generated-prefix
contexts, `hybrid-unlikelihood` alternates first-error and rollout updates,
`staged-unlikelihood` runs them in phases, and `random-char` samples answer
characters. `periodic-rollout-unlikelihood` keeps most updates on first-error
repair while injecting rollout repair every N steps. `early-stop-unlikelihood`
targets premature terminator predictions, and
`periodic-early-stop-unlikelihood` injects that repair every N steps.
`repeat-loop-unlikelihood` targets repeated suffix patterns in the model's own
greedy rollout, and `periodic-repeat-loop-unlikelihood` injects that repair
every N steps. `balanced-repair-unlikelihood` pairs a self-generated repair
with a teacher-forced admitted continuation, and
`periodic-balanced-repair-unlikelihood` injects that balanced repair every N
steps. `generated-prefix-recovery-unlikelihood` trains after the first bad
generated prefix, while `sequence-repair-unlikelihood` samples greedy mistakes
across the correct admitted target prefix. The `periodic-*` sequence and
generated-prefix modes inject those repairs every N steps.
`loop-escape-unlikelihood` pairs repeated-loop penalties with admitted
continuations, and `branch-repair-unlikelihood` targets an answer branch
position such as the first content character after the leading answer space.
`branch-contrast-unlikelihood` contrasts that branch against another admitted
prompt with a different branch target; `periodic-branch-repair-contrast-*`
keeps branch repair as the base update and injects contrast every N steps.
Add `--use-context-mean` to either transformer training command to test a
mean-pooled prompt-context residual in the final transformer representation.
Add `--use-context-projection` to test a zero-initialized trainable projection
of that prompt-context summary.
Add `--use-prompt-attention-summary` to test a trainable attention-pooled
summary of the current context through a zero-initialized output projection.
Add `--direct-answer-require-branch-context-gate` to require complete,
unambiguous branch contexts before direct-answer training runs.
The direct transformer path is not yet part of the promotion gate for reliable
answers.

`closed_world_lm.admit` appends a new structured memory to
`corpus/admissions.jsonl`. That is the operational meaning of "I learned
something new": the fact is admitted first, then `self_improve answer-cycle`
regenerates lessons and updates versioned weights. It can admit one fact from
CLI fields or a JSONL batch. When writing to the default project admissions
file, it also syncs generated admission probes to `evals/admissions.jsonl`.
It syncs admission paraphrase probes to `evals/admission_paraphrases.jsonl` at
the same time.

`closed_world_lm.admission_probes` regenerates or checks direct and paraphrase
admission probes from `corpus/admissions.jsonl`. This keeps admitted-memory
eval sets derived from the admitted-memory source instead of hand-maintained
beside it.

`closed_world_lm.glossary_probes` regenerates or checks glossary definition
probes from the `probe_words` listed in `corpus/glossary.json`. Glossary probes
are evaluation-only ledger sources; glossary definitions become trainable
through the admitted glossary itself.

`closed_world_lm.self_improve answer-cycle` runs a repeatable improvement cycle:
regenerate the admitted curriculum, train the learned answer model, evaluate the
reliable responder, learned answer model, and generative answer decoder, then
write `self_improvement_report.json`. The report records the admission source,
direct/paraphrase admission-probe audit, glossary-probe audit, weight updates,
prompt-leakage audit, exact eval audit, promotion gate, rule-based
self-diagnosis, and exact evals for QA, unknowns, held-out, paraphrases,
ownership, self, learning, admissions, admission paraphrases, and glossary
definitions. Learned-component summaries include failed record details when an
eval is not exact.
Pass `--compare-report runs/<prior>/self_improvement_report.json` to add a
forgetting audit against a prior cycle. Each run archives every attempt under
`runs/<name>/attempts/attempt-###/`, then updates the top-level report as the
latest pointer. Each run also writes `corpus_snapshot.json` and
`corpus_diff.json` so source-level corpus changes can be audited alongside
weights and evals.

`closed_world_lm.self_diagnose` reads a `self_improvement_report.json` and emits
a deterministic repair recommendation from report evidence. It is intentionally
rule-based and sets `uses_external_model` to `false`; this is the current
non-external-model bridge toward future self-improvement guidance learned from
QuarkLM's own admitted artifacts.

## Admitting New Knowledge

To teach the model a new closed-world fact, admit it first:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admit \
  --id learned-child-book \
  --person child \
  --object book \
  --color blue \
  --relation on \
  --container table
```

Use a fresh `--id` for each new admission. Duplicate ids are rejected. After
admission, run `closed_world_lm.self_improve answer-cycle` so the curriculum is
regenerated and the learned weights are updated from the new admitted data.
When using the default project paths, admission probes are regenerated
automatically. Use `--no-sync-probes` only when deliberately staging corpus and
eval changes separately.

To admit a batch, create a JSONL file with one object per fact using the same
fields, then run:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admit \
  --batch path/to/new_admissions.jsonl
```

Batches are checked before writing: duplicate ids already in the corpus, or
duplicate ids inside the batch, reject the whole batch.

To check probe sync without writing:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admission_probes --check
```

## Forgetting Checks

When adding new admissions, compare the new run to the last promoted report:

```bash
PYTHONPATH=src python3 -m closed_world_lm.self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.42/self_improvement_report.json
```

The forgetting audit compares responder, answer-classifier, and answer-decoder
final evals. For every shared eval set, the new run must keep at least the prior
probe count, exact matches, and exact rate.

## Engineering Quality

Project quality principles live in `QUALITY.md`. In short: keep module
responsibilities narrow, prefer pure functions for audits and corpus transforms,
add focused tests for behavior changes, and treat provenance artifacts as part
of the promotion gate. README, STATUS, GOAL, QUALITY, Docusaurus docs, and the
standalone marketing page should be updated with every promoted version so docs
do not drift from the current state. If they reference current product state,
release evidence, evals, or commands, they must be updated with the release.

## Purity Boundary

The corpus ledger in `corpus/ledger.json` is the admission gate. v0 code should
only train on generated curriculum derived from files listed there. Evaluation
probes are checked into the project, but they are source probes rather than a
claim of out-of-distribution generalization.

## Shape Of The Experiment

```text
corpus/glossary.json
corpus/grammar.json
corpus/admissions.jsonl
        |
        v
closed_world_lm.curriculum
        |
        v
build/train.txt + build/valid.txt
        |
        v
closed_world_lm.train
        |
        v
runs/<name>/checkpoint.json
        |
        v
closed_world_lm.evaluate
```

## Next Milestones

1. Complete the Python package/import migration from `closed_world_lm` to the
   QuarkLM naming plan without breaking existing run artifacts.
2. Expand the Docusaurus docs and standalone marketing page as the model,
   corpus, and release evidence grow.
3. Deepen self-diagnosis from explicit rules toward admitted-corpus-trained
   repair proposal and selection, without external model shaping.
4. Add larger continual-learning batches using generated probes and forgetting
   checks.
5. Strengthen prompt-conditioned representation so the direct transformer emits
   target-specific answers instead of a short global wrong answer, while
   preserving the `37/219` candidate discrimination and v0.42 target-loss gains.
6. Consider a from-scratch corpus-derived subword tokenizer only after the
   character-token transformer evidence shows tokenizer length is the bottleneck.
7. Fold the reliable decoder behavior back into the broader free-form character
   language model.

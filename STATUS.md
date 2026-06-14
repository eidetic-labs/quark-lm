# QuarkLM - Status

**Status:** Experimental research scaffold
**Active version:** v0.42 wider sparse branch-contrast transformer training
**Last updated:** 2026-06-14
**Buildable:** yes, with Python standard library only

QuarkLM explores bounded epistemic growth: a model starts from random weights
and only learns from an explicitly admitted corpus. The intended GitHub
repository slug is `quark-lm`. The first milestone is a dependency-free toy
learner, not a production model.

Working tagline: Big idea. Tiny package.

## Current Scope

- Human-authored seed glossary, grammar, and admitted-memory log.
- Deterministic curriculum generation.
- Character tokenizer trained only on generated admitted text.
- Tiny neural character MLP trained from random initialization.
- Tiny decoder-only transformer trained from random initialization with a
  dependency-free scalar autodiff engine.
- Reliable corpus responder for exact closed-world answers.
- Learned answer classifier trained from random weights.
- Generative answer decoder trained from random weights.
- Operational self facts: dataset boundary, pretrained-weight policy, unknown
  policy, and improvement method.
- Admission rule: "I learned something new" means a fact was admitted into the
  ledgered corpus before training and weight updates.
- Batch admission support for appending multiple structured memories with
  duplicate-id rejection before writing.
- Forgetting audit support for comparing a new self-improvement cycle against a
  prior report.
- Corpus provenance snapshots and corpus diffs for self-improvement reports.
- Generated direct and paraphrase admission probes from
  `corpus/admissions.jsonl`, with probe-sync audits in self-improvement
  reports.
- Generated glossary probes from `corpus/glossary.json` probe words, with
  glossary-probe audits in self-improvement reports.
- Admitted memories and story facts now produce generated bridge lessons to
  preserve held-out transfer without leaking protected held-out prompts.
- Learned-component eval summaries include failed records when any probe misses.
- Self-improvement reports include an exact eval audit and a promotion gate; the
  command returns failure unless all audits and evals are promotion-ready.
- Self-improvement reports include rule-based self-diagnosis that recommends
  the next action from report evidence without using an external model.
- Self-improvement attempts are archived under `attempts/attempt-###/` before
  the top-level latest report is updated.
- Package metadata now uses `quark-lm`, with `quark-lm-*` script aliases.
- Public surfaces: Docusaurus docs at `docs.quark-lm.eidetic-labs.com` and a
  standalone static marketing page at `quark-lm.eidetic-labs.com`, with
  GitHub Actions deployment scaffolds. The marketing site is not Docusaurus.
- SOLID-aligned quality guidance in `QUALITY.md`.
- Paper-grounded research guidance for continual learning, replay,
  self-generated candidate lessons, retrieval rails, model editing boundaries,
  transformer architecture, and tokenizer timing.
- Source probes for known, unknown, held-out, paraphrase, ownership, self,
  learning, admission, admission-paraphrase, and glossary answers.

## Research Grounding

QuarkLM's current research posture is documented in
`sites/docs/docs/learn/research-grounding.md`. The 2026-06-14 research pass
maps QuarkLM to self-improvement lifecycle work, continual learning, replay,
synthetic-recursion risk, small-data language learning, data hygiene, retrieval
rails, model editing boundaries, transformer mechanics, and evaluation
contamination. The project is adjacent to those areas, but its stricter claim
is that model weights and tokenizer state should be trained only from admitted,
ledgered data.

The practical near-term guidance is:

- make replay a first-class training primitive;
- keep corpus admission separate from model belief;
- accumulate original admitted records instead of replacing them with
  model-generated summaries;
- add corpus hygiene reports for duplicates, source mixtures, synthetic
  candidate ratios, and rare-record coverage;
- verify self-generated lessons before admission or training;
- promote only through retention, forgetting, leakage, unknown-policy, and
  branch-diversity gates;
- train the transformer from coverage deficits, not only from already-covered
  branch targets;
- defer model editing and self-rewarded grading until locality, side effects,
  and verifier quality are measurable inside the closed world.

## Latest Evidence

`runs/self-improve-v0.42/` passes protected prompt leakage, forgetting against
`runs/self-improve-v0.41/`, exact eval audit, promotion gate, and reaches 100%
exact match for the responder, learned answer classifier, and generative answer
decoder across all 10 current eval sets. Admission probes now pass `48/48`
direct and `84/84` paraphrase records; glossary probes pass `38/38`. The
passing attempt is archived at
`runs/self-improve-v0.42/attempts/attempt-001/`. The report diagnosis records
zero blockers with `uses_external_model: false`.

`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/` is the current
from-scratch transformer answer evidence. It uses the corpus-trained character
tokenizer, no pretrained weights, no pretrained tokenizer, and no external
embeddings. v0.42 keeps the v0.41 sparse branch-repair/contrast objective and
widens the from-scratch transformer from embedding/feed-forward dimensions
`4/8` to `8/16`. The run trained `80` target-loss steps plus `1000` sparse
branch repair/contrast direct answer steps with context size `32`; answer
target NLL moved `3.5850 -> 2.4129`, direct answer target loss moved
`3.4278 -> 2.2708`, and transformer-only eval-scoped candidate accuracy moved
`15/219 -> 37/219`. Raw direct greedy exact remained `0/219 -> 0/219`; the
failure changed from a repeated `"te"`/`"e"` loop to the short wrong answer
`" te."`, so prompt-conditioned greedy branching is still the current
bottleneck.

The latest unpromoted transformer diagnostic is
`runs/transformer-answer-v0.45-branch-rank-diagnostic-smoke-dim4-context80/`.
Branch profiles now record target rank and top predicted alternatives. In the
pre-layer-norm prompt-position path, QA and heldout both still collapse to
`"n"` with average target rank `14.25` and top-3/top-5 coverage `0.125`, so the
next repair should improve prompt-to-answer output binding rather than only
balance branch sampling.

`runs/transformer-answer-v0.46-output-binding-rankscore-smoke-dim4-context80/`
tests that repair direction with `branch-output-binding-unlikelihood` and
rank-aware best-snapshot scoring. The run completed `20/20` direct steps with
output bias frozen. QA average target rank improved from `17.375` to `14.125`
and QA/heldout top-5 coverage reached `0.25`, but target-token coverage stayed
`0.0`, top-3 coverage ended `0.0`, and both profiles still collapsed to wrong
branch tokens. The repair is rejected for promotion, while rank-aware restore
remains a useful guardrail.

`runs/transformer-answer-v0.47-rank-margin-steps50-smoke-dim4-context80/`
adds `branch-rank-margin-unlikelihood`, which pushes each branch target above
the model's own top wrong tokens. The run completed `50/50` direct steps and
restored the rank-aware best snapshot from step `40`. QA average target rank
improved from `17.375` to `9.0`, target-token coverage rose to `0.125`, top-3
coverage rose to `0.25`, and top-5 coverage rose to `0.5`. This is the
strongest rank-lift evidence so far, but prediction diversity stayed `1/8` and
QA/heldout remained collapsed to wrong `"n"`, so it is not promotion evidence.

`runs/transformer-answer-v0.48-balanced-rank-margin-smoke-dim4-context80/`
combines target-balanced branch batches with rank-margin repair. It completed
`50/50` direct steps and reached QA predicted diversity `2/8`, average target
rank `9.375`, target-token coverage `0.125`, top-3 coverage `0.375`, and top-5
coverage `0.5`. It improves wrong-token diversity and top-3 coverage versus
v0.47, but QA and heldout still choose wrong top-1 branch tokens, so it remains
rejected evidence.

`runs/transformer-answer-v0.49-balanced-rank-margin-top1-smoke-dim4-context80/`
tests whether concentrating rank-margin pressure on only the top wrong token
converts rank lift into top-1 branch choices. It restored the best branch
snapshot from step `10`; QA target-token coverage stayed `0.125`, but average
target rank regressed to `12.5`, top-3 coverage fell to `0.125`, and top-5
coverage fell to `0.25`. The screen is rejected and points away from
single-wrong-token pressure.

`runs/transformer-answer-v0.50-balanced-topk-softmax-w5-smoke-dim4-context80/`
tests `branch-balanced-topk-softmax-unlikelihood`, a target-balanced branch
batch objective that makes the correct branch target compete in a restricted
softmax against the model's current top wrong tokens. It restored the best
branch snapshot from step `40`; QA target-token coverage stayed `0.125`, but
average target rank improved to `8.75`, top-3 coverage reached `0.375`, and
top-5 coverage reached `0.5`. The screen is stronger than v0.49 and comparable
to v0.48 on top-k coverage, but it is still rejected because QA and heldout
remain collapsed to one wrong top-1 branch token.

`runs/transformer-v0.51-foundation-stack-smoke/` is the latest transformer
foundation-stack evidence. v0.51 adds optimizer state and scheduling
mechanics, gradient accumulation, checkpoint-resume validation, v2 checkpoint
metadata, multi-head attention, RMSNorm, gated MLPs, tied output embeddings,
rotary-position support, cache-aware generation metadata, sampling controls,
token-level traces, and replayable eval sample JSONL before the next
direct-answer repair run. The all-switch smoke completed `2/2` language-model
steps with AdamW, wrote `optimizer_state.json`, saved a
`quarklm-transformer-v2` checkpoint, and produced traced eval artifacts. This
is mechanics-readiness evidence, not a promoted responder checkpoint.

`runs/transformer-answer-v0.52-fullstack-topk-softmax-smoke-dim4-context80/`
uses the full v0.51 stack for the first post-foundation direct-answer screen.
It reruns `branch-balanced-topk-softmax-unlikelihood` with AdamW, gradient
accumulation, two attention heads, RMSNorm, gated MLPs, tied output embeddings,
rotary positions, cache-aware metadata, and prompt-position projection. The
run completed `50/50` direct steps and restored the best branch snapshot from
step `0`. The baseline/final restored state had QA predicted diversity `3/8`,
target-token coverage `0.25`, average target rank `13.25`, top-3 coverage
`0.25`, and top-5 coverage `0.375`; heldout was similar with average rank
`13.375`. Training improved rank at later snapshots but collapsed target
coverage and diversity to one wrong token, so the screen rejects unchanged
top-k pressure under the full stack and points next toward bidirectional
prompt-to-token binding.

`runs/transformer-answer-v0.53-fullstack-bidir-binding-smoke-dim4-context80/`
adds `branch-balanced-bidirectional-binding-unlikelihood`, which binds prompt
contexts to target tokens in both directions. It completed `50/50` direct steps
under the same full stack and restored the best branch snapshot from step `40`.
QA average target rank improved to `7.875` with top-5 coverage `0.5`; heldout
average rank improved to `9.0` with top-5 coverage `0.375`. The screen is
still rejected for promotion because target-token coverage ended at `0.125`,
top-1 remained wrong, and the diversity target still failed `0/9` multi-target
profiles. This is partial rank-pressure progress and makes coverage
preservation the next repair target.

`runs/transformer-answer-v0.54-fullstack-coverage-binding-smoke-dim4-context80/`
adds `branch-balanced-coverage-binding-unlikelihood`, which combines
bidirectional binding with hard-wrong-token competition and an explicit
target-set mass coverage guard. The focused tests pass, but the full-stack
screen rejects the mechanism: it completed `50/50` direct steps and
best-snapshot scoring restored step `0`. Training snapshots improved QA rank as
far as `8.125`, but target-token coverage fell to `0.0` and top-1 collapsed to
wrong `"a"`. The guard prevented promotion of a worse checkpoint; the next
repair should preserve target-set coverage as its own objective before
sharpening exact target selection.

`runs/transformer-answer-v0.55-fullstack-target-set-coverage-smoke-dim4-context80/`
isolates target-set coverage with
`branch-balanced-target-set-coverage-unlikelihood`: no exact-target row loss,
no cross-context ownership term, and positive target CE disabled in the screen.
The focused tests pass, but the full-stack screen still restores step `0`.
Training snapshots improved QA average target rank to `10.0`, yet
target-token coverage again fell to `0.0` and top-1 collapsed to wrong `"a"`.
The failure is now sharper: batch-local target-set mass is not enough to
preserve eval target-token coverage, so the next repair should add explicit
anti-collapse pressure over predicted target tokens.

`runs/transformer-answer-v0.57-fullstack-target-diversity-smoke-dim4-context80/`
adds that pressure with `branch-balanced-target-diversity-unlikelihood`, which
combines target-set mass with a target-share diversity term. The focused tests
pass, including a regression that lifts both restricted target-set mass and the
weakest target's average share in a small branch batch. The full-stack screen
still rejects the mechanism: it completed `50/50` direct steps and
best-snapshot scoring restored step `0`. Training snapshots improved QA
average target rank to `10.0`, but target-token coverage again fell to `0.0`
with wrong `"a"` top-1 collapse. The next repair should preserve eval-wide
target-token coverage directly, likely through replay or diversity scoring tied
to heldout branch profiles rather than batch-local target sharing alone.

`runs/transformer-answer-v0.58-fullstack-target-replay-coverage-smoke-dim4-context80/`
implements that replay-shaped repair with
`branch-balanced-target-replay-coverage-unlikelihood`: the sampled branch batch
still trains unlikelihood, but target-set mass and target-share balance use the
broader admitted branch training pool at the same branch position. The focused
tests pass, including a regression where the sampled batch omits some pool
targets and training raises both replay target-set mass and the weakest missing
target share. The full-stack screen still rejects the mechanism: it completed
`50/50` direct steps and best-snapshot scoring restored step `0`. Training
snapshots improved QA average target rank as far as `6.875` and QA top-5
coverage to `0.5`, while admissions top-5 reached `0.5417`; however,
target-token coverage still hit `0.0` during training and QA/heldout top-1
predictions collapsed to one wrong branch token by step `50`. The next repair
should make replay context-owned, not merely pool-owned, so broad target mass
cannot be satisfied by the same target distribution on every branch context.

`runs/transformer-answer-v0.59-fullstack-context-replay-coverage-smoke-dim4-context80/`
implements context-owned replay with
`branch-balanced-context-replay-coverage-unlikelihood`: the repair batch still
trains wrong-token unlikelihood, while a broader target-balanced replay sample
trains each admitted branch context to own its own target within the replay
target set. The focused tests pass, including a regression that raises replay
target-set mass and the weakest owned-target share on fixed replay contexts.
The full-stack screen still rejects the mechanism: it completed `50/50` direct
steps and best-snapshot scoring restored step `0`. Training snapshots improved
QA average target rank as far as `7.375`, QA top-3 to `0.375`, QA top-5 to
`0.5`, and admissions top-5 to `0.5208` by step `50`; however, the diversity
target still failed `0/9`, target-token coverage hit `0.0` during training,
and no trained snapshot beat the step-0 baseline under branch snapshot scoring.
The next repair should either strengthen target-preserving ownership directly
or make snapshot scoring reject rank/top-k gains whenever target-token coverage
regresses.

`runs/transformer-answer-v0.60-fullstack-context-replay-coverage-floor-metadata-smoke-dim4-context80/`
implements the scoring side of that repair. Best branch snapshot selection now
has a profile-wise target-token coverage floor: every multi-target profile must
preserve its baseline coverage before rank/top-k improvements can promote a
trained snapshot. Direct-answer JSONL snapshots also record
`branch_target_coverage_by_profile`, making the floor auditable in run
artifacts. The focused tests pass, including a regression that rejects a
rank-lifted candidate when QA target-token coverage drops below baseline. The
full-stack screen completed `50/50` direct steps, wrote `7` clean direct-answer
JSONL rows, and restored step `0`. The baseline coverage floor was preserved
in the final row (`qa` `0.25`, `heldout` `0.25`, `admissions` `0.1429`,
minimum profile `0.0714`), while step `40` still improved QA average target
rank to `7.375` and top-5 to `0.5` only by regressing profile coverage. This
accepts the self-improvement gate repair but rejects the trained model
behavior; the next repair should make the objective preserve target-token
coverage under training.

`runs/transformer-answer-v0.61-fullstack-context-coverage-anchor-smoke-dim4-context80/`
moves coverage preservation into the objective with
`branch-balanced-context-coverage-anchor-unlikelihood`: replay branches whose
target is already the top-1 prediction receive an additional covered-target
anchor against the replay target set and hard wrong tokens. The focused tests
pass, including a regression where the anchor protects a covered branch better
than identical replay training without the anchor. The full-stack screen still
rejects the mechanism: it completed `50/50` direct steps, wrote `7`
direct-answer JSONL rows, and restored step `0` under the v0.60 coverage
floor. Training snapshots over-anchored the already-covered wrong `"i"` token:
QA/heldout predicted diversity fell to `1/8`, target-token coverage fell to
`0.125`, and average target rank regressed above `21`. The next repair should
make preservation target-balanced or profile-aware, so an anchor can protect
baseline coverage without turning one covered token into a global attractor.

`runs/transformer-answer-v0.62-fullstack-target-balanced-anchor-smoke-dim4-context80/`
makes the covered-target anchor target-balanced with
`branch-balanced-context-target-balanced-anchor-unlikelihood`: anchor losses are
averaged by covered target and skipped when a replay batch contains only one
covered target. The focused tests pass, including a regression where the
singleton guard skips the v0.61 over-anchor while the old global anchor still
raises that single token. The full-stack screen completed `50/50` direct
steps, wrote `7` direct-answer JSONL rows, and restored step `0` under the
v0.60 coverage floor. It avoided the v0.61 global `"i"` attractor, but
QA/heldout target-token coverage still collapsed to `0.0` during training and
trained snapshots remained ineligible for restore. The next repair should use
profile-level coverage deficits as the training signal, not only anchors from
already-covered replay branches.

`runs/transformer-answer-v0.64-fullstack-coverage-deficit-smoke-dim4-context80/`
adds that first deficit-driven training mode with
`branch-balanced-context-coverage-deficit-unlikelihood`: replay target tokens
that are absent from current replay predictions receive extra target-vs-hard
candidate pressure. The focused tests pass, including a regression that lifts a
missing replay target above the old context replay objective. The full-stack
screen completed `50/50` direct steps, wrote `7` direct-answer JSONL rows, and
restored step `0` under the v0.60 coverage floor. Step `50` reached QA branch
accuracy `1/8`, QA predicted diversity `4/8`, and QA average target rank
`10.0`, but QA and heldout target-token coverage regressed to `0.125`, so the
trained snapshots remained ineligible. The next repair should combine deficit
pressure with an explicit coverage-preserving constraint.

Unpromoted v0.43 work added three pieces of transformer-loop evidence without
changing the promoted checkpoint. The forward pass now computes only the final
position consumed by the language-model head, cutting the transformer unit-test
runtime from roughly `13.9s` to `6.2s` on this machine. Transformer answer runs
now record prompt context-coverage metrics, showing that context size `80`
covers all current semantic eval templates (`219/219`) while context size `32`
does not. The hard-negative branch-contrast pilot at
`runs/transformer-answer-v0.43-hard-branch-contrast4-dim8-context32/` preserved
`37/219` candidates but regressed direct loss to `2.4225`, answer NLL to
`2.5402`, and greedy output to a repeated `" a"` loop. The full-context pilot at
`runs/transformer-answer-v0.43-branch-repair-contrast50-dim8-context80/`
preserved `37/219` candidates with `219/219` coverage but still trailed v0.42
on direct loss (`2.3122`) and answer NLL (`2.4546`). A 1500-step context-80 run
reached `38/219` candidates but regressed loss, NLL, and greedy output, so it
was not promoted. A layer-normalized context-80 screen at
`runs/transformer-answer-v0.43-layernorm-screen-dim8-context80/` preserved full
coverage and `37/219` candidates but regressed answer NLL to `2.5881` and
collapsed greedy output into repeated `" y"`/`"e"` loops, so it also remains
unpromoted evidence. A branch-span screen at
`runs/transformer-answer-v0.43-branch-span3-screen-dim8-context32/` broadened
branch repair to answer positions `1..3`; it preserved `37/219` candidates but
regressed answer NLL to `2.7426` and produced a long `"neeee"` loop, so it was
not promoted. Multi-layer transformer support was added as an explicit
architecture option, but the first two-layer context-32 screen at
`runs/transformer-answer-v0.43-two-layer-screen-dim8-context32/` was interrupted
before final direct-answer metrics because the full-block scalar autograd path
was too slow for the regular loop. The partial JSONL history remains runtime
evidence only. A follow-up optimized the final layer of stacked transformers to
compute only the final state and added equivalence coverage against full-stack
logits, but
`runs/transformer-answer-v0.43-two-layer-finalopt-screen-dim8-context32/` still
interrupted before final metrics because the intermediate full-state layer and
positive-context repair update remain too expensive. A follow-up added
top-layer-only direct-answer updates for stacked transformers and the explicit
`--skip-post-direct-snapshot` screening flag. The completed bounded screen at
`runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/`
saved a two-layer checkpoint after `40` target-loss steps and `80` top-layer
direct steps, recorded that the post-direct candidate snapshot was skipped,
improved direct-answer target loss `3.5186 -> 3.2436`, but kept direct greedy
exact at `0/219 -> 0/219` with repeated `"a"` output. It is runtime and
training-loop evidence only; v0.42 remains the promoted transformer checkpoint.
Direct-answer snapshots now include branch profiles computed from QuarkLM's own
logits. The smoke run at
`runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` recorded
QA branch-position-1 accuracy at `1/8` before and after five tiny direct updates;
the dominant prediction moved from all `"o"` to all `"y"`, and the average
target margin stayed negative. This gives the next repair loop a measurable
prompt-independent branch-collapse signal. Branch-collapse repair now uses the
dominant sampled branch prediction as the unlikelihood negative. The full-dose
smoke at `runs/transformer-answer-v0.43-branch-collapse-smoke-dim4-context16/`
regressed direct loss and moved the QA branch collapse to all `"a"` predictions.
The periodic smoke at
`runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/`
improved direct loss `3.5800 -> 3.5157`, but QA branch accuracy stayed
`1/8 -> 1/8` and the dominant branch prediction moved from all `"o"` to all
`"n"`. The repair is recorded as rejected evidence: dominant-token suppression
helps loss under sparse dosage, but does not create prompt-specific branches.
Branch-batch contrast now trains multiple distinct branch targets in a single
update. The full-dose smoke at
`runs/transformer-answer-v0.43-branch-batch-smoke-dim4-context16/` improved loss
only slightly and moved the QA branch collapse to all `"y"` predictions. The
periodic smoke at
`runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5800 -> 3.5248`, but QA branch accuracy regressed
`1/8 -> 0/8` and the dominant branch prediction moved to all `"a"`. That makes
branch-batch contrast rejected repair evidence too: the objective can move loss
without forcing prompt-conditioned branch separation.

A representation-side screen added `--use-context-mean`, which adds the
mean-pooled prompt context to the final transformer hidden state. The
branch-batch comparison run at
`runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5805 -> 3.5252`; the selected branch-repair comparison
at
`runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/`
improved direct loss `3.5805 -> 3.5310`. Both regressed QA branch accuracy
`1/8 -> 0/8` and collapsed to all `"a"` predictions, so context averaging is
rejected representation evidence rather than a promoted transformer step.

A follow-up representation screen added `--use-context-projection`, a
zero-initialized trainable projection of the mean-pooled context. The
branch-repair run at
`runs/transformer-answer-v0.43-context-projection-branch-repair-smoke-dim4-context16/`
moved all `20` projection parameters, improved direct loss
`3.5802 -> 3.5217`, and the branch-batch comparison at
`runs/transformer-answer-v0.43-context-projection-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5802 -> 3.5252`. Both regressed QA branch accuracy
`1/8 -> 0/8` and collapsed to all `"a"` predictions, so learned prompt-summary
projection is rejected representation evidence too.

A stronger representation screen added `--use-prompt-attention-summary`, a
trainable attention-pooled context summary with a zero-initialized output
projection. The branch-repair run at
`runs/transformer-answer-v0.43-prompt-attention-branch-repair-smoke-dim4-context16/`
moved all `20` zero-initialized output projection parameters and improved
direct loss `3.5802 -> 3.5217`; the branch-batch comparison at
`runs/transformer-answer-v0.43-prompt-attention-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5802 -> 3.5252`. Both regressed QA branch accuracy
`1/8 -> 0/8` and collapsed to all `"a"` predictions, so prompt attention is
also rejected representation evidence.

Direct-answer snapshots now include branch-context coverage diagnostics: the
visible branch context text, semantic feature coverage, context collisions, and
target-token ambiguity. The context-16 screen at
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context16/`
showed QA branch contexts had `0/8` semantic coverage and `4` ambiguous branch
windows. The context-32 screen at
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context32/`
removed QA ambiguity but still had `0/8` semantic coverage. The tiny
context-80 screen at
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context80/`
reached complete branch-context coverage across all eval sets (`219/219`) with
zero ambiguous branch contexts. This makes efficient longer-context branch
repair the next structured transformer target.

The branch-context coverage diagnostic is now actionable through
`--direct-answer-require-branch-context-gate`. The context-16 gated screen at
`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context16/`
required the gate, failed it, and recorded `actual_steps: 0` for `5` requested
direct steps. The context-80 gated screen at
`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context80/`
passed the gate and recorded `actual_steps: 1` for `1` requested direct step.
This is a training-loop guardrail, not promoted model quality evidence.

Direct-answer snapshots now have a `branch-only` mode for bounded longer-context
screens. The mode skips greedy completion evals in JSONL snapshots while still
recording branch profiles, branch-context coverage, and the gate result. The
context-80 gated screen at
`runs/transformer-answer-v0.43-branch-context-gated-branchonly-smoke-dim4-context80/`
passed the required gate across all `219/219` semantic records, recorded
`actual_steps: 5` for `5` requested direct branch-repair steps, and marked
`direct_answer_evals_skipped: true`. This makes longer-context branch-repair
screening cheaper and auditable, but it is not promoted quality evidence because
greedy completion evals were intentionally skipped.

Two dim8 context-80 branch-only follow-up screens used that cheaper evidence
path to test whether the best prior sparse repair/contrast policy or
branch-batch contrast could create prompt-specific branches once the visible
context was complete. The periodic repair/contrast screen at
`runs/transformer-answer-v0.43-branchonly-periodic-repair-contrast50-dim8-context80/`
passed the required gate, ran `100/100` direct steps, and lowered interval train
loss `6.7890 -> 6.4326`, but final QA branch prediction collapsed from all
space to all `"a"` with final QA branch accuracy `0/8`. The branch-batch screen
at `runs/transformer-answer-v0.43-branchonly-branch-batch-dim8-context80/`
passed the same gate, ran `50/50` direct steps, and lowered interval train loss
`3.4614 -> 3.1976`, but QA branch prediction still collapsed to all `"a"` with
final QA branch accuracy `0/8`. These are rejected screening results: complete
context and lower branch loss are still insufficient without a prompt-specific
branch signal.

Branch diversity is now an explicit direct-answer snapshot target. Each branch
profile records target-token diversity, predicted-token diversity, target-token
coverage, dominant predicted token/rate, collapse status, and missing target
tokens. Each direct-answer snapshot also records a `branch_diversity_target`
summary across multi-target eval profiles. The context-80 smoke at
`runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/`
passed the branch-context gate, ran `5/5` direct steps, and then failed the
branch-diversity target across all `9` multi-target profiles. The final QA
profile had `target_unique: 8`, `predicted_unique: 1`, dominant token `"r"` at
rate `1.0`, and target-token coverage `0.125`. This turns branch diversity into
a required screen signal before a full greedy-eval promotion snapshot is worth
running.

The first diversity-aware training mode is now implemented as
`branch-diversity-unlikelihood`. It trains distinct branch targets together,
penalizes the model's current wrong prediction for each branch context, and
retains the existing branch-target contrast penalty. Unit coverage verifies that
the objective suppresses a global wrong token while raising target probability
on a small branch batch. The context-80 corpus smoke at
`runs/transformer-answer-v0.43-branch-diversity-train-smoke-dim4-context80/`
passed the branch-context gate and ran `10/10` direct steps, but the final
branch-diversity target still failed across all `9` multi-target profiles. QA
prediction moved from all `"x"` to all `"b"`, target-token coverage improved
`0.0 -> 0.125`, and `predicted_unique` stayed `1/8`. This is rejected
training-mode evidence: the objective moves the collapse token but does not yet
create prompt-specific branch diversity.

Direct-answer training can now exclude the transformer output bias from updates
with `--direct-answer-freeze-output-bias`. This tests whether branch-diversity
loss was escaping through one global output-bias move instead of learning
prompt-specific weights. Unit coverage verifies that branch-diversity training
can keep `bout` unchanged while still updating output weights. The context-80
corpus smoke at
`runs/transformer-answer-v0.43-branch-diversity-freezebias-smoke-dim4-context80/`
passed the branch-context gate and ran `50/50` direct steps with the output bias
frozen. Interval loss moved `3.6149 -> 3.5016`, but the final
branch-diversity target still failed across all `9` multi-target profiles. QA
prediction moved from all `"x"` to all `"w"`, final target-token coverage was
`0.0`, and `predicted_unique` stayed `1/8`. This is rejected stabilizer
evidence: output bias can be guarded, but the current direct-answer path still
collapses through deeper prompt-independent weights.

`branch-target-softmax-unlikelihood` now adds a restricted softmax over the
distinct branch targets in each batch, so each context's correct branch token
must beat the other observed target tokens directly. Unit coverage verifies
that the objective improves restricted target probability on a small branch
batch. The context-80 corpus smoke at
`runs/transformer-answer-v0.43-branch-target-softmax-freezebias-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, and ran `50/50` direct
steps. Composite train loss moved `5.6671 -> 5.5820`, but the final
branch-diversity target still failed across all `9` multi-target profiles. QA
briefly reached `predicted_unique: 2` at step `20`, then collapsed back to all
`"w"` by step `50` with final target-token coverage `0.0`. This is rejected
target-set evidence: the restricted competition can crack collapse transiently,
but it does not yet stabilize prompt-specific branches.

Direct-answer runs can now opt into
`--direct-answer-restore-best-branch-snapshot`, which scores every branch
snapshot and restores the best measured branch-diversity checkpoint before
final metrics and checkpoint writing. The restore-best target-softmax smoke at
`runs/transformer-answer-v0.43-branch-target-softmax-restorebest-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
and restored the final checkpoint from step `40`. The final diversity target
still failed across all `9` multi-target profiles, but final QA target-token
coverage improved from the previous all-`"w"` final coverage `0.0` to all-`"u"`
coverage `0.125`. This is rejected guardrail evidence: best-snapshot
restoration preserves the best measured branch state, but it does not itself
create prompt-specific branch choices.

A prompt-focused representation screen added `--use-prompt-prefix-projection`,
a zero-initialized trainable projection over non-padding prompt-prefix positions
before the final answer token. The context-80 target-softmax restore-best smoke
at
`runs/transformer-answer-v0.43-prompt-prefix-target-softmax-restorebest-smoke-dim4-context80/`
moved all `20` prompt-prefix projection parameters and improved composite train
loss `5.6649 -> 5.5679`, but the final branch-diversity target still failed
across all `9` multi-target profiles. The final checkpoint restored from step
`40`; QA stayed collapsed to all `"u"` with target-token coverage `0.125` and
`predicted_unique` still `1/8`. This is rejected representation evidence:
targeted prompt-prefix access is active, but still not enough to separate
prompt-specific branches.

The next representation screen added `--use-prompt-position-projection`, which
keeps a separate zero-initialized trainable projection for each non-padding
prompt-prefix context position. The context-80 target-softmax restore-best
smoke at
`runs/transformer-answer-v0.43-prompt-position-target-softmax-restorebest-smoke-dim4-context80/`
moved `1108/1284` prompt-position projection parameters and improved composite
train loss `5.6649 -> 5.5679`, but the final branch-diversity target still
failed across all `9` multi-target profiles. The final checkpoint restored from
step `40`; QA stayed collapsed to all `"u"` with target-token coverage `0.125`
and `predicted_unique` still `1/8`. This is rejected representation evidence:
position-specific prompt access also moves, but still does not produce
prompt-specific branch choices.

A follow-up target-margin screen added `branch-target-margin-unlikelihood`, a
smooth pairwise margin loss over each batch's distinct branch targets. The
prompt-position context-80 smoke at
`runs/transformer-answer-v0.43-branch-target-margin-prompt-position-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
moved train loss `4.8973 -> 4.7784`, and moved `1108/1284` prompt-position
projection parameters. The final checkpoint restored from step `40`; QA stayed
collapsed to all `"u"` with target-token coverage `0.125`, `predicted_unique`
still `1/8`, and the branch-diversity target failed across all `9`
multi-target profiles. This is rejected target-margin evidence: pairwise target
separation improves the bounded loss but still does not stabilize
prompt-specific branch choices.

Direct-answer snapshots now include `branch_representation_profiles`, which
measure pairwise hidden-state distance between branch contexts before the
output head. A representation-contrast follow-up added
`branch-representation-contrast-unlikelihood`, which penalizes nearly identical
hidden states for different branch targets. The high-weight prompt-position
context-80 smoke at
`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim4-context80/`
used `--direct-answer-contrast-weight 50.0`, passed the branch-context gate,
froze output bias, ran `50/50` direct steps, and moved train loss
`53.5827 -> 53.4342`. The final checkpoint restored from step `40`; QA stayed
collapsed to all `"u"` with target-token coverage `0.125`, `predicted_unique`
still `1/8`, and average different-target hidden distance only about
`0.00107`. This is rejected representation-contrast evidence: the current
hidden states remain nearly indistinguishable at the answer branch.

A dim-8 capacity follow-up tested whether the representation-contrast path was
too narrow at embedding/feed-forward dimensions `4/8`. The matching 50-step
dim-8 screen reached step `40` but was too slow for the regular loop, so the
completed evidence run is
`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim8-context80-steps40/`.
It used embedding/feed-forward dimensions `8/16`, passed the branch-context
gate, froze output bias, ran `40/40` direct steps, and restored the final
checkpoint from step `10`. The restored QA different-target hidden distance
rose to about `0.00209`, but QA still collapsed to all `"u"` with target-token
coverage `0.125`, `predicted_unique` still `1/8`, and the branch-diversity
target failed across all `9` multi-target profiles. This is rejected capacity
evidence: more width gives more hidden distance, but not prompt-specific branch
choices.

A prompt-signal scale follow-up added `--prompt-position-projection-scale` so
bounded screens can amplify the prompt-position projection residual without
changing corpus data, tokenizer training, or initialization policy. The
scale-32 context-80 smoke at
`runs/transformer-answer-v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
moved `1108/1284` prompt-position projection parameters, and restored the final
checkpoint from step `40`. The raw step-50 snapshot briefly pushed QA
different-target hidden distance to about `0.4115`, and the restored checkpoint
kept it above the prior dim-4 screen at about `0.01235`. QA still collapsed to
all `"u"` with target-token coverage `0.125`, `predicted_unique` still `1/8`,
and the branch-diversity target failed across all `9` multi-target profiles.
This rejects prompt-signal volume as the whole fix: the model can separate
hidden states more than before, but the output path still turns them into one
global branch token.

The next transformer checkpoint should include an open-source structure audit
before another repair objective is added. `STRUCTURE_AUDIT.md` records the
boundary: QuarkLM may study model/trainer/tokenizer/checkpoint patterns from
open-source language-model projects, but must not import pretrained weights,
pretrained tokenizer vocabularies, external embeddings, unledgered datasets, or
training text. The audit now includes a comparison table against common GPT
structure. It identifies the next implementation target as an opt-in
pre-layer-norm transformer block path with final normalization, preserving the
existing default path for checkpoint compatibility before the next
branch-diversity repair.

The opt-in pre-layer-norm path is now implemented with
`--use-pre-layer-norm`. It uses GPT-style pre-normalization for attention and
MLP sublayers and applies final layer normalization before the output head,
while leaving the legacy default path unchanged for older checkpoints and prior
evidence. Focused tests cover config round trip, older-checkpoint defaults,
scalar/float forward parity, parameter inclusion, and parser wiring. The
context-80 prompt-position representation-contrast smoke at
`runs/transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
moved `1108/1284` prompt-position projection parameters and all `8` final-norm
parameters, and did not need best-snapshot restoration because step `50` was
the best measured branch snapshot. The final branch-diversity target still
failed across all `9` multi-target profiles; QA and heldout stayed fully
collapsed, with QA all `"y"` and target-token coverage `0.125`. The useful
change is that `7/9` profiles were no longer fully collapsed, including
admission paraphrases at `predicted_unique: 4/14` and admissions at
`2/14`. This is partial structural evidence, not promotion evidence: the next
repair should stabilize and extend that diversity to QA and heldout rather
than add another unrelated branch objective.

A target-balanced branch-batch follow-up tested whether repeated first-answer
tokens in the weighted direct-answer pool were crowding rare branch targets out
of representation-contrast updates. The new
`branch-balanced-representation-contrast-unlikelihood` mode builds each branch
batch from distinct target buckets while preserving the admitted-corpus-only
training boundary. The matching pre-layer-norm context-80 smoke at
`runs/transformer-answer-v0.44-target-balanced-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
passed the branch-context gate and ran `50/50` direct steps, but every trained
snapshot scored worse than the baseline branch-diversity snapshot. Best-snapshot
restoration returned to step `0`; the restored final state collapsed all `9/9`
multi-target profiles to `"n"`, and QA stayed at `predicted_unique: 1/8` with
target-token coverage `0.125`. This rejects target-balanced branch sampling as
a standalone repair. The next repair should keep the pre-layer-norm path but
target the prompt-to-answer binding that keeps QA and heldout moving together.

The v0.31 no-candidate auxiliary generator remains the best no-candidate exact
answer evidence: `runs/transformer-answer-v0.31-generator-weighted-lr035-80k/`
trained the generator for `80000` weighted steps at learning rate `0.035` and
moved exact generation from `0/219 -> 219/219` with
`uses_answer_candidates: false`.

## Out Of Scope

- Pretrained foundation models.
- Pretrained tokenizers or embedding models.
- Web retrieval.
- Production-grade model quality.

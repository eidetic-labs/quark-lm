---
title: Transformer
description: Train the from-scratch QuarkLM transformer prototype.
---

# Transformer

v0.24 introduced a tiny decoder-only transformer in
`closed_world_lm.transformer_char_model`.

It is intentionally small:

- corpus-trained character tokenizer
- learned token and position embeddings
- one causal self-attention block
- one feed-forward block
- next-character language-model head
- dependency-free scalar autodiff
- random initialization only

Train a smoke checkpoint:

```bash
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 40 \
  --context-size 8 \
  --embedding-dim 6 \
  --feedforward-dim 12
```

Evaluate answer probes:

```bash
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model eval \
  --checkpoint runs/transformer-smoke/transformer.json \
  --json runs/transformer-smoke/transformer_eval.json
```

Train on corpus-derived answer lessons:

```bash
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
  --direct-answer-mode periodic-balanced-repair-unlikelihood \
  --direct-answer-negative-weight 1.0 \
  --direct-answer-positive-weight 1.0 \
  --direct-answer-rollout-interval 50
```

Add `--use-context-mean` to either `train` or `answer-train` to test the
experimental mean-pooled context residual in the final transformer
representation. It is diagnostic architecture evidence only until it improves
prompt-conditioned branch profiles and complete answer metrics.

Current language-model evidence from `runs/transformer-v0.25/`:

| Signal | Value |
| --- | --- |
| Steps | `40` |
| Validation NLL | `3.5885 -> 3.4382` |
| Answer exact eval | `0/28` |
| Pretrained weights | `false` |
| Pretrained tokenizer | `false` |

Current promoted answer-lesson evidence from
`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/`:

| Signal | Value |
| --- | --- |
| Steps | `80` |
| Context size | `32` |
| Embedding / feed-forward dimensions | `8 / 16` |
| Candidate scope | `eval` |
| Direct answer steps | `1000` |
| Direct answer mode | `periodic-branch-repair-contrast-unlikelihood` |
| Direct answer negative weight | `1.0` |
| Direct answer positive weight | `1.0` |
| Direct answer contrast weight | `1.0` |
| Direct answer branch position | `1` |
| Direct answer rollout interval | `50` |
| Direct answer training examples | `9144` |
| Direct answer exact | `0/219 -> 0/219` |
| Direct answer target loss | `3.4278 -> 2.2708` |
| Direct answer uses candidates | `false` |
| Direct answer auxiliary weights | `false` |
| Answer target NLL | `3.5850 -> 2.4129` |
| Transformer-only candidate accuracy | `15/219 -> 37/219` |
| Selector-emitted exact answers | `18/219 -> 219/219` |
| Selector candidate accuracy | `18/219 -> 219/219` |
| v0.31 generator exact without candidates | `0/219 -> 219/219` |
| v0.31 generator target loss | `3.3160 -> 0.0029` |
| Pretrained weights | `false` |
| Pretrained tokenizer | `false` |
| External embeddings | `false` |
| v0.31 generator uses answer candidates | `false` |

Latest bounded stacked-transformer screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/` |
| Layers | `2` |
| Steps | `40` target-loss + `80` direct-answer |
| Direct-answer update scope | top layer and language-model head only |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| Pre-direct candidate accuracy | `15/219 -> 15/219` |
| Pre-direct answer target NLL | `3.5855 -> 3.4796` |
| Direct answer target loss | `3.5186 -> 3.2436` |
| Direct answer exact | `0/219 -> 0/219` |
| Failure pattern | repeated `"a"` greedy completion |
| Promotion status | screening evidence only |

Latest direct-answer diagnostic smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` |
| Diagnostic | branch profiles from model logits |
| Branch position | `1` |
| Smoke steps | `5` target-loss + `5` direct-answer |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| QA branch accuracy | `1/8 -> 1/8` |
| Dominant QA branch prediction | all `"o"` -> all `"y"` |
| Final QA target margin | negative, about `-0.0048` |
| Promotion status | diagnostic smoke only |

Latest branch repair smoke:

| Signal | Value |
| --- | --- |
| Selected comparison run | `runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/` |
| Prior rejected repair | `runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/` |
| Mode | `periodic-branch-batch-contrast-unlikelihood` |
| Branch batch size | `4` |
| Rollout interval | `5` |
| Steps | `5` target-loss + `20` direct-answer |
| Direct answer loss | `3.5800 -> 3.5248` |
| QA branch accuracy | `1/8 -> 0/8` |
| Dominant QA branch prediction | all `"o"` -> all `"a"` |
| Promotion status | rejected repair evidence |

Latest representation-side smoke:

| Signal | Value |
| --- | --- |
| Selected run | `runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/` |
| Comparison run | `runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/` |
| Representation option | `--use-context-mean` |
| Selected mode | `periodic-branch-repair-unlikelihood` |
| Comparison mode | `periodic-branch-batch-contrast-unlikelihood` |
| Steps | `5` target-loss + `20` direct-answer |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| Selected direct answer loss | `3.5805 -> 3.5310` |
| Comparison direct answer loss | `3.5805 -> 3.5252` |
| Selected QA branch accuracy | `1/8 -> 0/8` |
| Comparison QA branch accuracy | `1/8 -> 0/8` |
| Dominant QA branch prediction | all `"o"` -> all `"a"` in both screens |
| Promotion status | rejected representation evidence |

The transformer is not yet promoted as a reliable responder. It is architecture
evidence: a from-scratch attention model can update weights on the admitted
corpus and leave a checkpoint plus metrics. v0.42 preserves the `37/219`
transformer-only candidate result while improving answer-target NLL versus
v0.41, but raw greedy completion still fails exact answers with the short wrong
completion `" te."`. The latest v0.43 stacked screen proves that two-layer
top-layer-only direct-answer training can complete and write a checkpoint when
the expensive post-direct candidate snapshot is explicitly skipped, but its
repeated `"a"` output is still a failed direct decoder. v0.31's no-candidate
transformer-guided generator remains useful comparison evidence, but it is not
raw transformer decoding. The branch-profile smoke adds a sharper diagnosis:
at the configured branch position, the model is selecting one global token
across prompts instead of separating target-specific answer branches. The
branch-collapse repair uses that diagnosis by penalizing the sampled dominant
branch token, but the evidence shows it only moves the collapse to a new global
token. Branch-batch contrast then trains several distinct target branches in
one update; it lowers loss under sparse dosage, but the branch profile still
collapses globally and even loses the one initially correct QA branch.
`--use-context-mean` then adds a mean-pooled context residual to the final
hidden representation, but the bounded screens still collapse the QA branch to
one wrong global token. The next repair needs a stronger prompt-conditioned
representation signal than simple prompt averaging, not just suppression or
batching of branch tokens.

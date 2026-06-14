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

Current language-model evidence from `runs/transformer-v0.25/`:

| Signal | Value |
| --- | --- |
| Steps | `40` |
| Validation NLL | `3.5885 -> 3.4382` |
| Answer exact eval | `0/28` |
| Pretrained weights | `false` |
| Pretrained tokenizer | `false` |

Current answer-lesson evidence from `runs/transformer-answer-v0.38-periodic-balanced50-context32/`:

| Signal | Value |
| --- | --- |
| Steps | `80` |
| Context size | `32` |
| Candidate scope | `eval` |
| Selector steps | `1600` |
| Selector labels | `55` |
| Selector features | `3163` |
| Direct answer steps | `1000` |
| Direct answer mode | `periodic-balanced-repair-unlikelihood` |
| Direct answer negative weight | `1.0` |
| Direct answer positive weight | `1.0` |
| Direct answer rollout interval | `50` |
| Direct answer training examples | `9144` |
| Direct answer exact | `0/219 -> 0/219` |
| Direct answer target loss | `3.3496 -> 3.0399` |
| Direct answer uses candidates | `false` |
| Direct answer auxiliary weights | `false` |
| Answer target NLL | `3.5828 -> 2.8552` |
| Transformer-only candidate accuracy | `15/219 -> 37/219` |
| Selector-emitted exact answers | `18/219 -> 219/219` |
| Selector candidate accuracy | `18/219 -> 219/219` |
| v0.31 generator exact without candidates | `0/219 -> 219/219` |
| v0.31 generator target loss | `3.3160 -> 0.0029` |
| Pretrained weights | `false` |
| Pretrained tokenizer | `false` |
| External embeddings | `false` |
| v0.31 generator uses answer candidates | `false` |

The transformer is not yet promoted as a reliable responder. It is architecture
evidence: a from-scratch attention model can update weights on the admitted
corpus and leave a checkpoint plus metrics. v0.38 adds periodic balanced
repair training: most updates preserve first-error discrimination while every
fiftieth update pairs a self-generated repair target with a teacher-forced
admitted continuation. That preserves the `37/219` transformer-only candidate
result and improves answer-target NLL versus v0.37, but raw greedy completion
still fails exact answers and loops on `" t"`. v0.31's no-candidate
transformer-guided generator remains useful comparison evidence, but it is not
raw transformer decoding.

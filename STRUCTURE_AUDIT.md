# Open-Source Structure Audit

Last updated: 2026-06-14.

This audit records architectural patterns QuarkLM can study from open-source
language-model projects without importing their weights, tokenizers,
embeddings, datasets, or training text.

## Boundary

Allowed:

- Study code organization, training-loop shape, checkpoint discipline,
  tokenizer interfaces, config layout, and evaluation artifacts.
- Reimplement ideas in QuarkLM's own dependency-free code when they advance the
  closed-world goal.
- Record every adopted structure in docs, tests, and run evidence.

Not allowed:

- Pretrained model weights.
- Pretrained tokenizer vocabularies or merge tables.
- External embeddings.
- Unledgered text in training, validation, tokenizer fitting, or repair data.
- Copying an open-source implementation into the project instead of building a
  QuarkLM-native version.

## References

- [minGPT](https://github.com/karpathy/minGPT): a small educational GPT split
  into model, tokenizer, and trainer concerns. Its README calls out that most
  GPT complexity is batching across examples and sequence length.
- [nanoGPT](https://github.com/karpathy/nanoGPT): a compact
  train/model/sample/config structure for from-scratch GPT experiments. Its
  README now points readers toward newer work, so QuarkLM uses it only as a
  structural reference.
- [LitGPT](https://github.com/Lightning-AI/litgpt): a larger recipe-oriented
  layout for pretraining, finetuning, and deployment with explicit configs,
  tests, and workflow separation.
- [Hugging Face tokenizers](https://github.com/huggingface/tokenizers): a
  reference for tokenizer pipeline stages,
  vocabulary training, special tokens, padding, truncation, and alignment
  metadata. QuarkLM may study these interfaces, but any tokenizer must train
  only on admitted corpus text.
- [LLM360](https://arxiv.org/abs/2312.06550): a transparency reference for
  publishing training code, data, checkpoints, intermediate results, and
  analyses together.

## Current QuarkLM Gap Analysis

QuarkLM already has strong provenance, corpus admission, self-improvement
reports, and evaluation gates. The current transformer evidence shows that
objective iteration is no longer enough by itself: prompt-position projection
scaling and representation contrast can increase hidden-state separation while
branch predictions still collapse to one global token.

Before adding another direct-answer objective, inspect the transformer against
standard GPT structure:

- model boundary: config, embeddings, attention block, residual path,
  normalization, MLP, output head, generation
- trainer boundary: batching, optimizer, loss computation, checkpoint
  selection, eval cadence, resume behavior
- tokenizer boundary: corpus-only vocabulary training, special tokens,
  padding, truncation, manifesting, future subword path
- evidence boundary: baseline/final metrics, failed-record retention, branch
  diversity, hidden-state diagnostics, run provenance
- runtime boundary: scalar-autodiff clarity versus the minimum batching needed
  to make structural experiments practical

## Architecture Comparison

| Area | Open-source GPT pattern | QuarkLM current state | Next decision |
| --- | --- | --- | --- |
| Module boundary | minGPT separates `model.py`, `trainer.py`, and tokenizer concerns; nanoGPT keeps compact `model.py`, `train.py`, and `sample.py` boundaries. | `transformer_char_model.py` is currently `6219` lines and owns the transformer, answer repair modes, selector, generator, CLI, training loops, and metrics. | Split structure before adding many more repair modes: move reusable transformer model helpers and direct-answer training helpers behind narrower interfaces. |
| Block shape | minGPT and nanoGPT use a GPT block with layer norm before attention and before MLP: residual attention, then residual MLP. | QuarkLM applies attention on unnormalized states, then optional normalization inside `_feed_forward_*`, then an optional final post-MLP normalization. This is not the common pre-layer-norm GPT path. | Add an opt-in pre-layer-norm GPT block path and compare it against the existing optional layer norm before another branch objective is promoted. |
| Final normalization | GPT implementations commonly apply a final layer norm before the language-model head. | QuarkLM has per-block optional norms, but no dedicated final normalization before `wout`. | Add or screen final layer norm with the pre-layer-norm block, and record whether it changes branch diversity. |
| Attention | minGPT and nanoGPT use multi-head causal attention and batch/sequence tensorization for efficient training. | QuarkLM uses one scalar-audited attention head and single-context updates, with a final-position optimization for some paths. | Keep single-head scalar clarity for now, but document multi-head and batched updates as scaling targets rather than immediate repairs. |
| MLP activation | GPT blocks commonly use GELU-family activations with feed-forward width near `4 * n_embd`. | QuarkLM uses `tanh` and a configurable feed-forward dimension; current promoted transformer uses `8/16`, not `4x`. | Add a corpus-only GELU option after pre-layer norm, then screen activation separately from branch objectives. |
| Output head | nanoGPT ties token embeddings and the language-model head; minGPT uses an untied head in its older form. | QuarkLM uses an untied `wout`/`bout` head. Tying is possible because token embeddings are `[vocab][dim]` and the head is `[dim][vocab]`, but the output bias would remain separate. | Treat tied embeddings as a later stabilizer screen after pre-layer norm, because it changes output capacity directly. |
| Optimizer | nanoGPT configures AdamW groups, applies weight decay only to matrix parameters, and records optimizer choices. | QuarkLM uses plain gradient descent with gradient clipping in each train step. | Add a small dependency-free AdamW-style optimizer helper before longer structural screens, or explicitly reject it with evidence. |
| Tokenizer | Hugging Face tokenizers separates normalization, pre-tokenization, model, trainer, special tokens, padding, and truncation. | QuarkLM has a clean corpus-only character tokenizer with `<pad>` and strict out-of-vocabulary errors. | Keep the character tokenizer as the current purity baseline; design any future subword tokenizer as a separate admitted-corpus-trained artifact. |
| Evidence | LLM360 emphasizes publishing training code, data, checkpoints, intermediate results, and analyses together. | QuarkLM already records corpus provenance, run artifacts, checkpoints, metrics, failed attempts, and docs evidence. | Keep the evidence discipline; add structural comparison fields to transformer screens that claim architectural progress. |

## Decision

The latest prompt-position scale run shows QuarkLM can increase hidden-state
separation without producing prompt-specific branch tokens. That shifts the
next model change away from another branch-loss objective and toward a standard
GPT structural screen.

Next implementation target:

1. Add an opt-in pre-layer-norm transformer block path.
2. Include a final normalization before the output head when that path is
   enabled.
3. Preserve the existing default path so prior checkpoints still load and
   previous evidence remains comparable.
4. Add focused tests proving baseline-equivalent behavior at initialization
   where zero-initialized additions are expected, forward parity between scalar
   and float paths, config round trip, and distinct behavior when the new path
   is enabled.
5. Run a bounded context-80 branch-only screen only after those structural tests
   pass.

## Next Structural Targets

1. Implement and test an opt-in pre-layer-norm transformer path with final
   normalization.
2. Decide whether QuarkLM should add additional standard stabilizers such as
   residual-dropout placeholders, tied embeddings, better initialization, or a
   more conventional optimizer schedule.
3. Keep branch-diversity repair gated by context coverage, but stop treating
   new loss terms as the default next move until the structural audit explains
   why the base model should be able to use them.
4. Preserve the current dependency-free path unless a new dependency is
   deliberately admitted as engineering infrastructure and does not affect the
   model's closed-world training data.

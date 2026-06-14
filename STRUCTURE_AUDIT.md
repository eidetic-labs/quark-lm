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

- minGPT: a small educational GPT split into model, tokenizer, and trainer
  concerns. Its README calls out that most GPT complexity is batching across
  examples and sequence length.
- nanoGPT: a compact train/model/sample/config structure for from-scratch GPT
  experiments. Its README now points readers toward newer work, so QuarkLM uses
  it only as a structural reference.
- LitGPT: a larger recipe-oriented layout for pretraining, finetuning, and
  deployment with explicit configs, tests, and workflow separation.
- Hugging Face tokenizers: a reference for tokenizer pipeline stages,
  vocabulary training, special tokens, padding, truncation, and alignment
  metadata. QuarkLM may study these interfaces, but any tokenizer must train
  only on admitted corpus text.
- LLM360: a transparency reference for publishing training code, data,
  checkpoints, intermediate results, and analyses together.

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

## Next Structural Targets

1. Produce a transformer-architecture comparison table before the next repair
   objective is promoted.
2. Decide whether QuarkLM's transformer should add standard stabilizers such as
   pre-layer normalization, residual-dropout placeholders, tied embeddings,
   better initialization, or a more conventional optimizer schedule.
3. Keep branch-diversity repair gated by context coverage, but stop treating
   new loss terms as the default next move until the structural audit explains
   why the base model should be able to use them.
4. Preserve the current dependency-free path unless a new dependency is
   deliberately admitted as engineering infrastructure and does not affect the
   model's closed-world training data.

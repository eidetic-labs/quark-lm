---
title: Tokenizer
description: How QuarkLM tokenizers stay closed-world while improving compression.
---

# Tokenizer

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- Why `CharTokenizer` remains the reference baseline.
- How the append-only closed-world subword tokenizer proposes new tokens.
- Which artifacts prove the tokenizer used admitted text only.
- How tokenizer diagnostics separate compression evidence from model promotion.

</div>

QuarkLM treats tokenization as part of the model boundary. A tokenizer can make
long answers shorter, faster, and less drift-prone, but it can also smuggle in
outside vocabulary if it is not governed. The rule is simple: tokenizer state is
trained only from admitted corpus text, never from a pretrained vocabulary.

## Current baseline

`tokenizer.CharTokenizer` is still the reference tokenizer. It learns a sorted
character vocabulary from admitted text, reserves `<pad>` as token `0`, rejects
characters outside that vocabulary, and round-trips text exactly. It is simple,
auditable, and intentionally conservative.

That baseline is not performance-optimal. Long answers produce many generation
steps, and each step is a chance for autoregressive drift. The baseline stays in
place because it is the truth surface for comparison.

## Closed-world subword path

`closed_world_subword_tokenizer.ClosedWorldSubwordTokenizer` adds an append-only
subword path. It starts from admitted characters, then replays accepted merge
rules in order. Existing token IDs are never remapped.

<div className="qlm-grid">
<div><h4>Corpus-only</h4><p>Candidate tokens are scored from admitted text supplied to the run. No pretrained vocabulary is accepted.</p></div>
<div><h4>Append-only</h4><p>New tokens are added after existing IDs so older checkpoints remain interpretable.</p></div>
<div><h4>Composable</h4><p>New token embeddings can be initialized from their constituent token embeddings instead of random noise.</p></div>
<div><h4>Guarded</h4><p>Full-answer tokens, newline-crossing tokens, excessive-length tokens, and whole rare words are rejected or penalized.</p></div>
</div>

The scorer is WordPiece-like rather than raw BPE frequency alone. It rewards
compression, pair informativeness, reuse across sources, and branch-diverse
contexts. It penalizes single-context chunks and rejects candidates that would
turn the tokenizer into an answer table.

## Interface contract

`TokenizerProtocol` is the shared interface for tokenizer implementations. A
tokenizer must expose `encode`, `decode`, `extend`, `extends`, `to_dict`,
`from_dict`, `pad_id`, `vocab_size`, `tokenizer_type`, and its append-only
`tokens` list. This keeps the transformer, checkpoint, tokenizer-manifest, and
diagnostic paths working against the same boundary whether a screen uses the
character baseline or closed-world subword tokens.

When a resumed checkpoint receives append-only tokenizer growth, QuarkLM runs a
vocabulary-expansion parity audit before accepting the resized model. The audit
samples admitted training contexts and compares old-vocabulary logits before and
after expansion; any old-logit drift blocks the resume.

## Train with a subword tokenizer

The default remains character tokenization. Use the subword path explicitly:

```bash title="Subword tokenizer screen"
PYTHONPATH=src python3 -m transformer_char_model train \
  --run runs/transformer-subword-screen \
  --tokenizer closed-world-subword \
  --tokenizer-max-token-chars 4 \
  --tokenizer-max-new-tokens 16 \
  --steps 80
```

Transformer answer-training screens use the same governed tokenizer path:

```bash title="Answer-training subword screen"
PYTHONPATH=src python3 -m transformer_char_model answer-train \
  --run runs/transformer-answer-subword-screen \
  --tokenizer closed-world-subword \
  --tokenizer-max-token-chars 4 \
  --tokenizer-max-new-tokens 16 \
  --steps 0
```

The run writes `tokenizer_manifest.json` and `tokenizer_report.json` next to the
checkpoint unless explicit paths are supplied. Answer-training metrics and
`sweep_plan.json` record the tokenizer type and manifest hash so char and
subword screens can be compared without hidden tokenizer drift.

The self-improvement answer cycle also writes those artifacts automatically.
In that path they are candidate evidence, not an active-tokenizer change. The
cycle records `tokenizer_candidate_guard` and blocks promotion if the proposal
uses pretrained state, imports outside vocabulary, fails round-trip, or creates
protected full-answer tokens.

## Compare tokenizer behavior

`transformer_tokenizer_benchmark` compares the character baseline with the
closed-world subword path on short answers and longer answers with repeated
fragments. It records what the tokenizer naturally selects from the scored
corpus instead of requiring a particular segmentation.

The benchmark is intentionally narrow. It proves compression and round-trip
behavior; it does not prove the transformer weights have learned the answers.

## What is not promoted

The subword tokenizer does not promote the transformer by itself. A tokenizer
screen can pass while the neural model still fails answer generation,
retention, branch diversity, or unknown-policy gates. Promotion still belongs
to the full model evidence path.

## What is next

<div className="qlm-next">

<a href="../transformer/"><strong>Read next</strong><span>Transformer</span><small>How tokenizer changes feed the weight-consolidation path.</small></a>

<a href="../../operate/tokenizer-manifests/"><strong>Operate</strong><span>Tokenizer manifests</span><small>The artifacts that prove corpus-only vocabulary growth.</small></a>

<a href="../../secure/purity-boundary/"><strong>Secure</strong><span>Purity boundary</span><small>Why pretrained tokenizers remain out of bounds.</small></a>

</div>

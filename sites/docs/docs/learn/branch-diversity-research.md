---
title: Branch Diversity Research
description: External research and v0.112 root-cause evidence for QuarkLM branch-diversity failures.
---

# Branch Diversity Research

Last reviewed: 2026-06-15.

QuarkLM's branch-diversity problem is the current transformer bottleneck.
Retrieval memory can serve the admitted corpus exactly, and guarded weight
updates can be accepted locally, but the transformer still predicts too few
branch tokens across multi-target profiles.

## v0.112 Evidence

Diagnostic run:
`runs/transformer-answer-v0.112.0-branch-diversity-root-cause-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.111 memory-consolidation plan, targets `owner`,
`paraphrases`, and `glossary`, keeps retrieval exact at `219/219`, records
`24` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `8` fallbacks, and remains rejected on
`branch_diversity_target`.

The new root-cause diagnostic classifies the final failure as
`target_routing_gap` with `critical` severity. It records `9/9` failed
profiles, `3` collapsed profiles, `1` zero-coverage profile, `6` buried-target
profiles, and reused dominant tokens: `"n"` across `5` profiles and `"a"`
across `4` profiles. The worst profile is `paraphrases`: `0.0` target-token
coverage, `predicted_unique: 1`, and average target rank `22.5`.

## What External Work Suggests

| Source | What others do | QuarkLM implication |
| --- | --- | --- |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | Shows that common decoding choices can produce bland or repetitive language even from strong likelihood-trained models. | Decoding diversity is not enough. QuarkLM needs branch diversity in the learned distribution before promotion. |
| [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) | Penalizes undesirable tokens or sequences during training. | QuarkLM's unlikelihood variants can move the collapse token, but v0.112 says routing remains broken. |
| [Hugging Face generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies) | Exposes greedy, sampling, beam, and custom generation methods; sampling can reduce repetition at inference. | Sampling may become an inference rail, but it cannot prove closed-world weight consolidation. |
| [Diverse Beam Search](https://arxiv.org/abs/1610.02424) | Adds diversity to beam decoding to avoid near-duplicate candidate outputs. | Useful later for candidate exploration; not a substitute for target-token coverage. |
| [Class-Balanced Loss](https://arxiv.org/abs/1901.05555) | Reweights long-tailed classes using effective sample counts. | Audit profile/target imbalance before changing another objective. |
| [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Separates representations by label. | Measure whether prompt states separate by branch target before adding output-head pressure. |
| [OLMo](https://arxiv.org/abs/2402.00838) and [LLM360](https://arxiv.org/abs/2312.06550) | Release training code, data, checkpoints, evaluations, and intermediate artifacts. | Keep root-cause diagnostics and promotion decisions artifacted. |
| [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/train.py) and [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) | Use clean GPT mechanics, cross-entropy training, validation loss, checkpointing, and sampling. | Structure references only; they do not directly solve QuarkLM's tiny closed-world branch gate. |

## Taxonomy

v0.112 adds `branch_diversity_target.root_cause`:

| Hypothesis | Meaning |
| --- | --- |
| `global_output_prior_collapse` | Multi-target profiles collapse to one shared dominant token. |
| `profile_local_prediction_collapse` | Profiles collapse, but not to one shared token. |
| `target_routing_gap` | At least one profile has zero target-token coverage. |
| `target_rank_burial` | Correct targets are usually outside the top-k set. |
| `wrong_diversity_not_target_coverage` | Predictions are diverse but miss the target tokens. |
| `mixed_branch_diversity_gap` | Multiple weaker failure modes appear together. |

## Decision

The next repair should audit the route from prompt evidence to branch target
before adding another branch objective:

1. Measure global logit priors and output-bias escape paths.
2. Compare prompt-to-branch hidden-state separation for failed profiles.
3. Separate zero-coverage profiles from buried-target profiles.
4. Add candidate construction and sampling diagnostics for profile/target
   imbalance.
5. Require the root-cause report to improve without relaxing
   `branch_diversity_target`.

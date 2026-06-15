---
title: Branch Diversity Research
description: External research and v0.113 routing-audit evidence for QuarkLM branch-diversity failures.
---

# Branch Diversity Research

Last reviewed: 2026-06-15.

QuarkLM's branch-diversity problem is the current transformer bottleneck.
Retrieval memory can serve the admitted corpus exactly, and guarded weight
updates can be accepted locally, but the transformer still predicts too few
branch tokens across multi-target profiles.

## v0.113 Evidence

Diagnostic run:
`runs/transformer-answer-v0.113.0-branch-routing-audit-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.112 memory-consolidation plan, targets `owner`,
`paraphrases`, and `learning`, keeps retrieval exact at `219/219`, records
`18` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `6` fallbacks, and remains rejected on
`branch_diversity_target`.

The root-cause diagnosis remains a critical `target_routing_gap`: `9/9`
profiles fail, `3` remain collapsed, `1` has zero target-token coverage, and
`6` have buried targets. The new `branch_routing_audit` narrows the next
mechanics target:

- `audit_hypothesis`: `routing_gap_requires_representation_and_logit_audit`
- output-bias escape risk: `high`, with `"n"` at bias rank `2`
- prompt-to-branch representation separation: low across `9/9` multi-target
  profiles
- minimum different-target hidden distance: about `0.00077`
- target imbalance hotspot: `glossary`, with top target share `0.6316`

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
| [Hugging Face generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation) | Exposes repetition penalties, no-repeat n-gram controls, diversity penalties, sampling, beam settings, and other generation-time controls. | Sampling or penalties may become inference rails, but they cannot prove closed-world weight consolidation. |
| [Hugging Face generation utilities](https://huggingface.co/docs/transformers/en/internal/generation_utils) | Exposes logits processors, processed score tensors, and optional hidden-state outputs for instrumentation. | v0.113 follows this diagnostic pattern by inspecting output-bias ranks and prompt-to-branch hidden-state separation. |
| [Diverse Beam Search](https://arxiv.org/abs/1610.02424) | Adds diversity to beam decoding to avoid near-duplicate candidate outputs. | Useful later for candidate exploration; not a substitute for target-token coverage. |
| [fairseq search mechanics](https://github.com/facebookresearch/fairseq/blob/main/fairseq/search.py) | Implements search variants, including diversity-aware beam scoring. | Mature stacks keep search diversity separate from model learning, so QuarkLM should keep decoding diversity out of promotion claims. |
| [Class-Balanced Loss](https://arxiv.org/abs/1901.05555) | Reweights long-tailed classes using effective sample counts. | Audit profile/target imbalance before changing another objective. |
| [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Separates representations by label. | Measure whether prompt states separate by branch target before adding output-head pressure. |
| [OLMo](https://arxiv.org/abs/2402.00838) and [LLM360](https://arxiv.org/abs/2312.06550) | Release training code, data, checkpoints, evaluations, and intermediate artifacts. | Keep root-cause diagnostics and promotion decisions artifacted. |
| [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) and [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) | Use clean GPT mechanics, cross-entropy training, logits, validation loss, checkpointing, and sampling. | Structure references only; they do not directly solve QuarkLM's tiny closed-world branch gate. |

## Taxonomy

v0.112 adds `branch_diversity_target.root_cause`, and v0.113 adds
`branch_routing_audit`:

| Hypothesis | Meaning |
| --- | --- |
| `global_output_prior_collapse` | Multi-target profiles collapse to one shared dominant token. |
| `profile_local_prediction_collapse` | Profiles collapse, but not to one shared token. |
| `target_routing_gap` | At least one profile has zero target-token coverage. |
| `target_rank_burial` | Correct targets are usually outside the top-k set. |
| `wrong_diversity_not_target_coverage` | Predictions are diverse but miss the target tokens. |
| `mixed_branch_diversity_gap` | Multiple weaker failure modes appear together. |

## Decision

The next repair should instrument the route from prompt evidence to branch
target before adding another branch objective:

1. Measure global logit priors and output-bias escape paths.
2. Compare prompt-to-branch hidden-state separation for failed profiles.
3. Separate zero-coverage profiles from buried-target profiles.
4. Add candidate construction and sampling diagnostics for profile/target
   imbalance.
5. Require both `branch_diversity_target.root_cause` and `branch_routing_audit`
   to improve without relaxing `branch_diversity_target`.

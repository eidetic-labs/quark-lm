# Branch Diversity Research

Last reviewed: 2026-06-15.

## Question

QuarkLM keeps failing `branch_diversity_target`: retrieval memory is exact, some
guarded updates are accepted, but the transformer still predicts too few branch
tokens for multi-target profiles. The v0.112 decision is to stop adding repair
objectives until the failure is classified.

## Current v0.112 Evidence

Diagnostic run:
`runs/transformer-answer-v0.112.0-branch-diversity-root-cause-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.111 memory-consolidation plan, targets `owner`,
`paraphrases`, and `glossary`, keeps retrieval exact at `219/219`, records
`24` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `8` fallbacks, and remains rejected on
`branch_diversity_target`.

The new root-cause diagnostic classifies the final failure as:

- `root_cause_hypothesis`: `target_routing_gap`
- `severity`: `critical`
- `failed_profiles`: `9/9`
- `collapsed_profile_count`: `3`
- `zero_coverage_profile_count`: `1`
- `buried_target_profile_count`: `6`
- reused dominant tokens: `"n"` across `5` profiles and `"a"` across `4`
  profiles
- worst profile: `paraphrases`, with `0.0` target-token coverage,
  `predicted_unique: 1`, and average target rank `22.5`

This means the immediate bottleneck is not just output diversity. The model is
not routing the prompt to the right target set, and in several profiles the
correct targets are still buried.

## External Findings

| Source | What others do | QuarkLM implication |
| --- | --- | --- |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | Shows that likelihood-trained language models can produce bland or repetitive output under common decoding choices, and motivates nucleus sampling as an inference-time diversity control. | Do not treat a decoding trick as weight learning. QuarkLM needs branch diversity in the learned distribution before promotion. |
| [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) | Adds losses that reduce probability assigned to undesirable tokens or sequences. | QuarkLM already uses many unlikelihood variants; v0.112 evidence says suppressing wrong tokens can move the collapse token without fixing routing. |
| [Hugging Face generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies) | Production libraries expose greedy, sampling, beam, and custom generation methods; sampling can reduce repetition at inference. | Keep generation strategy separate from promotion. Sampling may be useful later, but it cannot prove closed-world weight consolidation. |
| [Diverse Beam Search](https://arxiv.org/abs/1610.02424) | Adds diversity directly to beam decoding so multiple output candidates do not collapse to near-duplicates. | Useful for future inference rails, but branch diversity in QuarkLM must be measured at the profile target-token level before decoding diversity is trusted. |
| [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) | Reweights long-tailed classes using effective sample counts rather than raw counts. | Branch targets should be audited for profile/target imbalance before adding another objective; repeated source patterns can hide rare target failures. |
| [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Pulls representations with the same label together and pushes different labels apart. | Branch repair needs representation diagnostics: if prompt states are not separated by target branch, output-head pressure alone is likely insufficient. |
| [OLMo](https://arxiv.org/abs/2402.00838) and [LLM360](https://arxiv.org/abs/2312.06550) | Open LLM projects emphasize training code, data, checkpoints, evaluations, and transparent intermediate artifacts. | QuarkLM should keep the failure taxonomy, source plan, retrieval report, update guard, and promotion decision as first-class artifacts. |
| [nanoGPT training script](https://github.com/karpathy/nanoGPT/blob/master/train.py) and [minGPT model](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) | Small GPT projects focus on clean transformer mechanics, cross-entropy training, validation loss, checkpointing, and sampling. | These are structure references, not direct solutions. They do not solve QuarkLM's closed-world branch gate because they are not optimizing a tiny per-profile answer manifold. |

## Root-Cause Taxonomy

v0.112 adds a data-only root-cause report under
`branch_diversity_target.root_cause`:

- `global_output_prior_collapse`: all multi-target profiles collapse to the
  same dominant token.
- `profile_local_prediction_collapse`: every profile collapses, but not to one
  shared token.
- `target_routing_gap`: at least one multi-target profile has zero target-token
  coverage.
- `target_rank_burial`: correct targets are usually outside top-k even when
  predictions are not fully collapsed.
- `wrong_diversity_not_target_coverage`: predictions are diverse, but they do
  not cover the target tokens.
- `mixed_branch_diversity_gap`: multiple weaker failure modes appear together.

## Decision

The next repair should not be another generic branch-loss knob. It should first
audit the route from prompt evidence to branch target:

1. Measure global logit priors and output-bias escape paths.
2. Compare prompt-to-branch hidden-state separation for failed profiles.
3. Separate zero-coverage profiles from buried-target profiles.
4. Add candidate construction and sampling diagnostics for profile/target
   imbalance.
5. Only then choose a repair objective, and require the root-cause report to
   improve without relaxing `branch_diversity_target`.

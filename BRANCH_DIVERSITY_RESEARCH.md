# Branch Diversity Research

Last reviewed: 2026-06-15.

## Question

QuarkLM keeps failing `branch_diversity_target`: retrieval memory is exact, some
guarded updates are accepted, but the transformer still predicts too few branch
tokens for multi-target profiles. The v0.112 decision is to stop adding repair
objectives until the failure is classified.

## Current v0.115 Evidence

Candidate run:
`runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`.

v0.115 adds `branch-hidden-projection-margin-unlikelihood`, a repair candidate
that compares target-token `hidden * output_weight` contributions directly and
runs with output bias frozen. The one-step screen reduces average
collapsed-token hidden advantage from about `0.0842` to `0.0736`, supporting
hidden projection as a relevant repair surface.

The candidate remains rejected for neural promotion. Constraint-first promotion
passes `10/11` constraints and fails `branch_diversity_target`; all `9/9`
multi-target profiles still collapse to `"n"`, `2` profiles keep zero
target-token coverage, and hidden-projection pressure remains primary across
`9/9` profiles. The next repair must scale beyond one branch batch while
preserving coverage, representation separation, and branch-diversity gates.

## v0.114 Evidence

Diagnostic run:
`runs/transformer-answer-v0.114.0-logit-prior-representation-instrumentation-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.113 memory-consolidation plan, targets `owner`,
`paraphrases`, and `glossary`, keeps retrieval exact at `219/219`, records
`24` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `8` fallbacks, and remains rejected on
`branch_diversity_target`.

The root-cause diagnosis remains a critical `target_routing_gap`. The audit
still flags high output-bias escape risk (`"n"` bias rank `1`, `"a"` bias rank
`3`), but the new `branch_logit_prior_profiles` decompose the failed
dominant-token wins as hidden-projection pressure across `9/9` multi-target
profiles. Centroid separation remains poor: all `9/9` multi-target profiles
have low representation separation, and the sampled centroid margins for
`owner`, `paraphrases`, `learning`, and `glossary` are poorly separated.

## v0.113 Evidence

Diagnostic run:
`runs/transformer-answer-v0.113.0-branch-routing-audit-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.112 memory-consolidation plan, targets `owner`,
`paraphrases`, and `learning`, keeps retrieval exact at `219/219`, records
`18` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `6` fallbacks, and remains rejected on
`branch_diversity_target`.

The root-cause diagnosis remains a critical `target_routing_gap`. The new
`branch_routing_audit` classifies the next mechanics risk as
`routing_gap_requires_representation_and_logit_audit`:

- output-bias escape risk: `high`, with `"n"` at bias rank `2`
- low representation separation across `9/9` multi-target profiles
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
| [Hugging Face generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation) | Production libraries expose repetition penalties, no-repeat n-gram controls, diversity penalties, sampling, beam settings, and other generation-time controls. | Keep generation strategy separate from promotion. Sampling or penalties may be useful later, but they cannot prove closed-world weight consolidation. |
| [Hugging Face generation utilities](https://huggingface.co/docs/transformers/en/internal/generation_utils) | Generation internals expose logits processors, processed score tensors, and optional hidden-state outputs for instrumentation. | v0.114 follows the diagnostic pattern: inspect logits, output-bias ranks, hidden projections, and hidden-state separation before changing another training objective. v0.115 turns that evidence into a guarded hidden-projection candidate. |
| [Diverse Beam Search](https://arxiv.org/abs/1610.02424) | Adds diversity directly to beam decoding so multiple output candidates do not collapse to near-duplicates. | Useful for future inference rails, but branch diversity in QuarkLM must be measured at the profile target-token level before decoding diversity is trusted. |
| [fairseq search mechanics](https://github.com/facebookresearch/fairseq/blob/main/fairseq/search.py) | Implements multiple search strategies, including diversity-aware beam variants that rewrite candidate scores during decoding. | Mature systems isolate search diversity from model learning; QuarkLM should keep inference diversity separate from guarded weight updates. |
| [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) | Reweights long-tailed classes using effective sample counts rather than raw counts. | Branch targets should be audited for profile/target imbalance before adding another objective; repeated source patterns can hide rare target failures. |
| [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Pulls representations with the same label together and pushes different labels apart. | Branch repair needs representation diagnostics: if prompt states are not separated by target branch, output-head pressure alone is likely insufficient. |
| [OLMo](https://arxiv.org/abs/2402.00838) and [LLM360](https://arxiv.org/abs/2312.06550) | Open LLM projects emphasize training code, data, checkpoints, evaluations, and transparent intermediate artifacts. | QuarkLM should keep the failure taxonomy, source plan, retrieval report, update guard, and promotion decision as first-class artifacts. |
| [nanoGPT model](https://github.com/karpathy/nanoGPT/blob/master/model.py) and [minGPT model](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) | Small GPT projects focus on clean transformer mechanics, cross-entropy training, validation loss, checkpointing, logits, and sampling. | These are structure references, not direct solutions. They do not solve QuarkLM's closed-world branch gate because they are not optimizing a tiny per-profile answer manifold. |

## Root-Cause Taxonomy

v0.112 adds a data-only root-cause report under
`branch_diversity_target.root_cause`, v0.113 adds `branch_routing_audit`, v0.114
adds `branch_logit_prior_profiles`, and v0.115 adds a hidden-projection margin
candidate:

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

The next repair should not be another generic branch-loss knob. v0.115 shows
that hidden projection is a relevant repair surface, but one branch batch is not
enough. The next repair should scale the route from prompt evidence to branch
target:

1. Target hidden-projection contributions that make dominant tokens beat
   missing target tokens.
2. Compare prompt-to-branch hidden-state separation for failed profiles.
3. Separate zero-coverage profiles from buried-target profiles.
4. Add candidate construction and sampling diagnostics for profile/target
   imbalance.
5. Only then choose a repair objective, and require the root-cause report,
   routing audit, logit-prior profile, and representation separation evidence
   to improve without relaxing `branch_diversity_target`.

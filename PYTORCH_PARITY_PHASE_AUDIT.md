# PyTorch Parity Phase Audit

Last reviewed: 2026-06-20.

This audit scopes the current `/goal` phase: experimental PyTorch backend
parity for the scalar transformer reference. Its purpose is to prevent the loop
from turning into endless validator tightening. The phase is about proving a
safe evidence contract for optional PyTorch parity attempts, not promoting
PyTorch as QuarkLM's trainer.

## Current Evidence Baseline

| Evidence | Current result |
| --- | --- |
| Branch | `feature/model-loop-audit` |
| Latest implementation checkpoint entering this audit | `a362d27 Validate training candidate schema` |
| Latest closeout checkpoint | `c82d918 Audit training exports` plus `PYTORCH_PARITY_PHASE_EXIT_CHECKLIST.md`. |
| Scalar role | Canonical dependency-free reference implementation. |
| PyTorch role | Optional runtime library for parity evidence only. |
| Default dependency status | PyTorch remains optional and is not required by the scalar install. |
| Training evidence status | Tiny CPU `float64` real-runtime replay parity is fixture-level evidence only. |
| Promotion status | PyTorch is unpromoted and is not a general training backend. |
| Latest broad gate before this audit | `806` Python tests passed, `1` skipped; docs and marketing build passed. |

## Phase Scope

This phase is in scope:

- optional PyTorch runtime detection and runtime-report validation;
- scalar fixture contracts for forward and training parity;
- PyTorch candidate artifacts that embed runtime evidence;
- trainable-parameter manifests and state summaries;
- training readiness, initial loss, backward, accumulation, update, final
  evaluation, and checkpoint compatibility probes;
- aggregate replay parity gates that distinguish matched fixture evidence from
  promotion;
- training parity reports rebuilt from fixture and candidate payloads;
- compact attempt summaries, artifact hashes, persisted artifact maps, and
  written-attempt reload validation;
- compact audit results for loop automation;
- public optional-backend exports for contributors to inspect the audit
  contract without private module paths.

This phase is out of scope:

- promoting PyTorch as QuarkLM's trainer;
- counting PyTorch runs as model-quality evidence;
- importing pretrained weights, pretrained tokenizers, external embeddings, or
  unledgered training text;
- replacing the scalar implementation as the canonical reference;
- solving the transformer branch-diversity blocker;
- completing the tokenizer and full transformer optimization roadmap.

## Completed Phase Work

| Area | Status | Evidence |
| --- | --- | --- |
| Runtime preflight | Done | Runtime reports validate schema, status, check catalog, summary counts, closed-world boundary, and eligibility flags. |
| Candidate artifacts | Done | Candidates embed runtime reports and validate top-level schema, runtime report, readiness, replay gate, training case, backend metadata, and route consistency. |
| Replay evidence | Done for tiny fixture | Replay control, gradient comparison, buffer comparison, update comparison, final evaluation, and checkpoint compatibility are gated and scoped. |
| Aggregate replay gate | Done | Gate requires runtime readiness, initial loss, backward coverage, optimizer control, replay gradients, buffers, updates, final evaluation, and checkpoint compatibility. |
| Promotion boundary | Done | Backend-promotion gate remains intentionally unpassed and names future gates. |
| Attempt artifacts | Done | Attempts validate status, pass flag, evidence scope, closed-world boundary, promotion gate, next requirements, compact summaries, hashes, and artifact map. |
| Persistence | Done | Writer records payload hashes and artifact paths, then reloads through the same validation contract. |
| CLI audit | Done | `--verify-existing` audits written attempts without rebuilding artifacts. |
| Compact audit | Done | Valid and invalid audit results have standalone validation for status, errors, routing, promotion, artifact file map, artifact hashes, evidence hashes, and exact payload keys. |
| Public surface | Done | Optional backend exports include the current audit, requirements, runtime, candidate, replay-gate, promotion-gate, and compact audit schema helpers/catalogs needed by contributors. |

## Closeout Status

| Priority | Gap | Status | Exit proof |
| --- | --- | --- | --- |
| P0 | Nested schema strictness pass | Closed | Focused tests prove runtime report, readiness, replay gate, training case, promotion gate, and audit payloads reject unvalidated top-level extras where appropriate. |
| P0 | Public export completeness audit | Closed | `tests/test_transformer_torch_backend_training_exports.py` proves the phase-critical training export catalog is explicit, public, and duplicate-free. |
| P1 | Real-runtime evidence refresh | Closed with blocked-runtime evidence | Runtime preflight and training parity attempt both report `blocked_runtime_unavailable` because PyTorch is not installed; the written-attempt audit still passes as `artifact_set_valid`. |
| P1 | Phase exit checklist | Closed | `PYTORCH_PARITY_PHASE_EXIT_CHECKLIST.md` records gates, non-claims, commands, blocked-runtime reason, and the next roadmap lane. |
| P2 | SRP watchlist cleanup | Closed for touched files | The compact audit schema was extracted to a focused helper; touched source and test files remain under the practical 250-line target. |

## Recommended Next Steps

1. Run the final bounded validation set for this phase.
2. Open a PR for the PyTorch parity evidence phase.
3. Return to the broader roadmap: tokenizer quality, transformer quality,
   corpus growth, and model-quality evaluation.

## Stop Rule

The current PyTorch parity evidence phase should stop when:

- focused PyTorch parity tests pass;
- full Python discovery passes;
- docs and marketing builds pass;
- candidate, attempt, audit, runtime, replay-gate, and promotion-gate schemas
  reject known drift paths;
- the public optional backend surface exposes the phase-critical contracts;
- the phase exit checklist clearly says PyTorch remains optional, unpromoted,
  and fixture-evidence-only.

After that point, additional PyTorch work should be opened as a new phase:
general trainer promotion, performance training runs, or hardware acceleration.

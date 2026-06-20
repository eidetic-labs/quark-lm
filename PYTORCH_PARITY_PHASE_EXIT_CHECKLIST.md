# PyTorch Parity Phase Exit Checklist

Last checked: 2026-06-20.

This checklist closes the current optional PyTorch parity evidence phase without
promoting PyTorch as QuarkLM's trainer. Scalar Python remains the canonical
dependency-free reference implementation.

## Gate status

| Gate | Status | Evidence |
| --- | --- | --- |
| Nested schema strictness | Passed | Runtime report, readiness, replay gate, training case, promotion gate, and compact audit validators reject known extra-key drift paths. |
| Public export completeness | Passed | Training public exports are checked against an explicit phase-critical catalog and aggregate backend exports reject duplicates. |
| Runtime preflight | Blocked in this environment | `python3 src/transformer_torch_runtime_report.py --requested-device cpu --requested-dtype float64 --output build/torch_runtime_report_float64.json` exited `1` with `blocked_runtime_unavailable`. |
| Training parity attempt | Blocked by runtime, artifact-valid | `python3 src/transformer_torch_training_parity_attempt_cli.py --output-dir build/torch_training_parity_attempt_float64 --requested-device cpu --requested-dtype float64` exited `1` with `blocked_runtime_unavailable`. |
| Written-attempt audit | Passed | `python3 src/transformer_torch_training_parity_attempt_cli.py --output-dir build/torch_training_parity_attempt_float64 --verify-existing` exited `0` with `artifact_set_valid`. |
| Promotion boundary | Held | The attempt reports `training_backend_not_promoted` and `promoted_training_backend: false`. |

## Runtime result

PyTorch is not installed in this environment:

- runtime status: `blocked_runtime_unavailable`;
- failed runtime checks: `runtime_available`, `runtime_kind`, `dtype_available`;
- next action: `install_real_pytorch_runtime`;
- closed-world flags remain clean: no pretrained weights, pretrained tokenizer,
  external embeddings, learned assets, or training data were imported.

This is an acceptable phase result because the runtime path produced a valid
blocked-runtime artifact set and the compact audit accepted the written attempt.
It is not real PyTorch training evidence.

## Non-claims

This phase does not claim:

- PyTorch is a promoted training backend;
- PyTorch runs count as model-quality evidence;
- the transformer branch-diversity issue is solved;
- tokenizer optimization is complete;
- a general PyTorch trainer exists.

## Validation commands

Run these before opening the phase PR:

```bash
python3 -m unittest \
  tests.test_transformer_torch_backend_training_exports \
  tests.test_transformer_torch_backend_public_audit \
  tests.test_transformer_torch_training_parity_attempt_audit_schema

python3 -m unittest \
  tests.test_transformer_torch_training_parity_attempt_audit_schema \
  tests.test_transformer_torch_training_parity_attempt_audit_validation \
  tests.test_transformer_torch_training_parity_attempt_audit

python3 -m compileall src tests
git diff --check
```

The broader release validation remains the normal full Python discovery, docs
build, marketing build, and shared-state JSON validation.

## Next roadmap lane

After this phase, return to model-quality work:

1. tokenizer quality and closed-world subword evidence;
2. transformer quality and branch-diversity repair;
3. corpus growth with governed admission;
4. evaluation evidence that separates retrieval memory from weight
   consolidation.

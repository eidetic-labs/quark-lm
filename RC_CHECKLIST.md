# QuarkLM RC Checklist

Last reviewed: 2026-06-15.

Use this checklist before tagging or announcing any QuarkLM release candidate.
It is designed to keep the release honest about the difference between the
audited research prototype and the not-yet-promoted transformer language model.

## 1. Choose The RC Track

Before running commands, name the track:

- **Research Prototype RC:** the closed-world self-improvement system is
  reproducible, auditable, documented, and clear about what is not yet promoted.
- **Language Model RC:** the from-scratch transformer itself passes neural
  promotion gates and can answer admitted corpus evals without hidden candidate
  selection.

Current recommendation: pursue **Research Prototype RC** first.

## 2. Required Clean Verification

Run from a clean checkout after dependencies are installed:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
npm run sites:build
python3 -m json.tool sites/shared/current-state.json >/dev/null
```

Expected current baseline:

- Python discovered suite passes.
- Docusaurus docs build passes; Read the Docs is responsible for publishing it.
- Marketing build passes through `npm run sites:build`; GitHub Pages is
  responsible for publishing it.
- Shared state JSON validates.

## 3. Required Evidence Artifacts

Research Prototype RC requires evidence for:

- promoted responder run: `runs/self-improve-v0.42/`;
- latest transformer screen:
  `runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`;
- corpus and training boundary checks;
- candidate quarantine checks;
- deterministic closed-world verifier checks;
- training recipe and constraint-first promotion reports;
- docs and marketing current-state alignment.

Language Model RC additionally requires:

- passing `branch_diversity_target`;
- non-collapsed multi-target branch profiles;
- target-token coverage floors met;
- direct-answer evals accepted without hidden candidate selection;
- retention and unknown-policy checks passing for the neural learner.

## 4. Forbidden RC Claims

Do not claim:

- that QuarkLM is a production language model;
- that retrieval success is neural weight learning;
- that the transformer is promoted while `branch_diversity_target` fails;
- that v0.115 solved branch routing;
- that the project has proven "world's first" status.

Allowed current claim:

- QuarkLM is an experimental closed-world research prototype with a reproducible
  admitted-corpus learning loop, exact retrieval/responder evidence, and an
  unpromoted from-scratch transformer whose next blocker is branch routing.

## 5. Release Surface Review

Before RC tagging, verify these agree:

- `README.md`;
- `STATUS.md`;
- `RC_SPEC.md`;
- `RC_GAP_AUDIT.md`;
- this checklist;
- `sites/shared/current-state.json`;
- `sites/DEPLOYMENT.md`;
- `.readthedocs.yaml`;
- `.github/workflows/deploy-marketing.yml`;
- Docusaurus Learn and Operate docs;
- standalone marketing page.

## 6. Decision Record

Record the final decision before tagging:

- RC track:
- version or tag:
- latest commit:
- verification commands:
- promoted evidence:
- unpromoted evidence:
- known blockers:
- forbidden claims reviewed:
- release decision:

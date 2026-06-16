# QuarkLM

QuarkLM is a tiny research prototype for a closed-world language model: random
weights, no pretrained tokenizer, no external embeddings, and learning only
from an explicitly admitted corpus.

Tagline: Big idea. Tiny package.

The repository slug is `quark-lm`. The Python import path is still
`closed_world_lm` until a dedicated package migration is promoted.

## Current Status

QuarkLM is not a production assistant. It is an experimental self-improvement
loop that separates corpus admission, retrieval memory, training candidates,
guarded weight updates, evaluation, and promotion.

Current release-candidate posture:

- **Research Prototype RC:** near. The admitted-corpus learning loop is
  reproducible, auditable, documented, and honest about rejected evidence.
- **Language Model RC:** not ready. The from-scratch transformer still fails
  `branch_diversity_target` after v0.115.

See `RC_SPEC.md`, `RC_GAP_AUDIT.md`, and `RC_CHECKLIST.md` before tagging or
announcing a release candidate.

## Core Boundary

QuarkLM currently allows:

- human-authored seed glossary, grammar, stories, self facts, and admitted
  memories
- corpus-derived curriculum and probes
- a character tokenizer trained only on admitted text
- random-initialized learned components
- corpus-only retrieval memory
- guarded weight updates accepted or rejected by evidence

QuarkLM currently forbids:

- pretrained weights
- pretrained tokenizers
- external embeddings
- unledgered training data
- treating retrieval success as neural weight learning

The central loop is:

```text
new lesson -> corpus -> retrieval memory -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

## Quickstart

Run commands from the repository root.

```bash
PYTHONPATH=src python3 -m closed_world_lm.curriculum --output build
PYTHONPATH=src python3 -m closed_world_lm.respond --eval --json runs/smoke/respond.json
PYTHONPATH=src python3 -m closed_world_lm.answer_model train --run runs/answer-smoke
PYTHONPATH=src python3 -m closed_world_lm.answer_decoder train --run runs/decoder-smoke
```

For the full command set, transformer screens, and release discipline, use the
Docusaurus docs.

## Documentation

- Docs: `https://docs.quark-lm.eidetic-labs.com`
- Marketing: `https://quark-lm.eidetic-labs.com`
- Project overview: `sites/docs/docs/learn/project-overview.md`
- Model philosophy: `sites/docs/docs/learn/language-model.md`
- Self-improvement loop: `sites/docs/docs/learn/self-improvement-loop.md`
- Current evidence: `sites/docs/docs/learn/current-evidence.mdx`
- Release candidate readiness: `sites/docs/docs/operate/release-candidate.md`
- Deployment: `sites/DEPLOYMENT.md`

The Docusaurus docs are hosted by Read the Docs. The standalone marketing page
is hosted by GitHub Pages.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `corpus/` | Ledgered closed-world source data. |
| `src/closed_world_lm/` | Model, curriculum, responder, training, retrieval, verifier, and eval code. |
| `tests/` | Focused regression coverage for the prototype. |
| `runs/` | Versioned local run evidence, ignored by git. |
| `sites/docs/` | Docusaurus documentation source. |
| `sites/marketing/` | Standalone static marketing site. |
| `sites/shared/current-state.json` | Shared state consumed by docs and marketing. |

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
npm run sites:build
python3 -m json.tool sites/shared/current-state.json >/dev/null
```

## License

MIT. See `LICENSE`.
